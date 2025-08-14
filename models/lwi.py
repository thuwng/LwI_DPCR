import logging
import numpy as np
import torch
import os
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from utils.data_manager import DummyDataset
from utils.inc_net import IncrementalNet, CosineIncrementalNet, Drift_Estimator, ALClassifier
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from torchvision import datasets, transforms
from utils.autoaugment import CIFAR10Policy
from copy import deepcopy

# -------------------------------
# Helper functions for LwI fusion
# -------------------------------

def extract_channel_reprs_from_state(state_dict, layer_keys=None, device='cpu'):
    """
    Given a state_dict (OrderedDict), extract per-layer per-out-channel representation vectors.
    Returns dict: layer_key -> tensor (n_out_channels, D)
    layer_keys: list of state_dict keys to consider (like 'model.features.0.weight' or 'layer1.0.conv1.weight').
    If None: choose keys that look like conv or linear weights.
    """
    reprs = {}
    for k, v in state_dict.items():
        if layer_keys is not None and k not in layer_keys:
            continue
        # only consider weights (not biases)
        if k.endswith('.weight'):
            W = v.cpu()
            if W.ndim == 4:
                # conv weight: [out_c, in_c, kh, kw] -> per out channel vector
                out_c = W.shape[0]
                vecs = W.view(out_c, -1).clone()  # (out_c, in_c*kh*kw)
                reprs[k] = vecs.to(device)
            elif W.ndim == 2:
                out_c = W.shape[0]
                vecs = W.clone()  # (out_c, in_c)
                reprs[k] = vecs.to(device)
            else:
                # skip other shapes
                continue
    return reprs

def compute_cost_matrix(E_old, E_new, metric='euclid'):
    """
    E_old: (n_old, d), E_new: (n_new, d)
    return cost matrix C (n_old, n_new)
    """
    # convert to float tensors
    A = E_old
    B = E_new
    # compute squared euclidean distances: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    with torch.no_grad():
        an = (A**2).sum(dim=1).unsqueeze(1)  # (n_old,1)
        bn = (B**2).sum(dim=1).unsqueeze(0)  # (1,n_new)
        C = an + bn - 2.0 * (A @ B.t())
        # ensure non-negative numerical
        C = torch.clamp(C, min=0.0)
        # optional: take sqrt to get euclidean
        C = torch.sqrt(C + 1e-12)
    return C

def greedy_match_from_cost(C):
    """
    Greedy 1-1 matching: returns perm indices mapping old_idx -> new_idx.
    C: torch tensor (n_old, n_new), minimize cost.
    If n_old != n_new, we will create mapping for min(n_old, n_new) pairs.
    Returns:
        perm_old2new: length n_old, with -1 for unmatched old channels (if n_old>n_new)
    """
    n_old, n_new = C.shape
    C_np = C.cpu().numpy()
    assigned_new = set()
    perm = -1 * np.ones(n_old, dtype=np.int64)
    # Greedy: pick smallest cost pair iteratively
    # Build list of (cost, old, new), sort ascending
    idxs = np.dstack(np.unravel_index(np.argsort(C_np.ravel()), (n_old, n_new)))[0]
    # iterate sorted pairs
    for old_idx, new_idx in idxs:
        if perm[old_idx] == -1 and new_idx not in assigned_new:
            perm[old_idx] = int(new_idx)
            assigned_new.add(int(new_idx))
        # early stop if all assigned
        if len(assigned_new) >= min(n_old, n_new):
            break
    return perm  # -1 means unassigned

def apply_permutation_and_fuse_state(state_old, state_new, layer_permutations, k=0.6, device='cpu'):
    """
    state_old/state_new: state_dict-like mappings {key: tensor}
    layer_permutations: dict key->perm array mapping old_out_idx -> new_out_idx (or -1)
    k: fusion coefficient (weight for old)
    Returns fused_state dict that can be loaded into model_new.
    Strategy:
      - For each weight key present in both states and in perms:
          - For conv weight [out,in,kH,kW]: permute output channels by new indices; permute input channels if prev layer had permutation
      - If key not in permutations: fallback fuse = k*old + (1-k)*new (elementwise)
    Note: We try to apply previous-layer permutations to input channels when available.
    """
    fused = {}
    state_old = {k: v.cpu() for k, v in state_old.items()}
    state_new = {k: v.cpu() for k, v in state_new.items()}
    keys = set(list(state_old.keys()) + list(state_new.keys()))
    # Build reverse mapping: for layer order we attempt to apply P_prev for input channels.
    # Assumption: permutations keys correspond to weight keys; for simplicity we apply output permutation only.
    for k in keys:
        Wo = state_old.get(k, None)
        Wn = state_new.get(k, None)
        if (Wo is not None) and (Wn is not None) and (k in layer_permutations) and Wo.ndim in (2,4):
            perm = layer_permutations[k]  # numpy array length n_old -> new_idx or -1
            # Build index mapping for output channels in new ordering.
            # We want to align old outputs to new outputs: create reorder_old such that old_aligned[new_pos] = old[old_pos]
            n_old = Wo.shape[0]
            n_new = Wn.shape[0]
            # create aligned_old with same shape as Wn
            aligned_old = torch.zeros_like(Wn)
            # for each old_idx -> new_idx
            for old_idx in range(n_old):
                new_idx = int(perm[old_idx]) if perm is not None and old_idx < len(perm) else -1
                if new_idx >= 0 and new_idx < n_new:
                    aligned_old[new_idx] = Wo[old_idx]
                else:
                    # if unmatched, we may place old channel into first available empty slot (or zero)
                    # we'll skip (remaining zeros)
                    pass
            # fused = k*aligned_old + (1-k)*Wn
            Wf = k * aligned_old + (1.0 - k) * Wn
            fused[k] = Wf.to(device)
        else:
            # fallback: if both exist fuse elementwise (shapes must match)
            if (Wo is not None) and (Wn is not None) and Wo.shape == Wn.shape:
                fused[k] = (k * Wo + (1.0 - k) * Wn).to(device)
            elif (Wn is not None):
                fused[k] = Wn.to(device)
            else:
                fused[k] = Wo.to(device)
    return fused

def select_weight_keys_for_matching(state_dict):
    """
    Heuristic selection of keys to match: choose those weight keys that are conv or linear weights,
    but skip classifier final layer weights if present (e.g., 'fc.weight' or 'heads').
    """
    keys = []
    for k, v in state_dict.items():
        if k.endswith('.weight'):
            # skip classifier final head often named 'fc.weight' or contains 'heads'
            if ('head' in k) or ('fc' in k and ('weight' in k) and (v.ndim == 2 and v.shape[0] < 1000 and v.shape[1] < 1024 and 'model' not in k)):
                # heuristic: still include earlier fc if needed; keep it simple: include everything except last heads
                pass
            # include conv/linear weights generally
            keys.append(k)
    # You may want to filter out final classifier by name patterns:
    filtered = [k for k in keys if ('heads' not in k) and ('classifier' not in k)]
    return filtered

def perform_lwi_fusion(old_model, new_model, args, device):
    """
    Orchestrates matching and fusion between old_model and new_model.
    - Extract state_dicts
    - choose matching keys
    - compute representations
    - matching with greedy assign
    - apply permutation & fuse
    - returns fused_state_dict (which should be loaded into new_model)
    """
    # Save state dicts
    state_old = {k: v.detach().cpu().clone() for k, v in old_model.state_dict().items()}
    state_new = {k: v.detach().cpu().clone() for k, v in new_model.state_dict().items()}

    # choose keys to match
    match_keys = select_weight_keys_for_matching(state_old)

    # extract per-layer reprs
    reprs_old = extract_channel_reprs_from_state(state_old, layer_keys=match_keys, device=device)
    reprs_new = extract_channel_reprs_from_state(state_new, layer_keys=match_keys, device=device)

    layer_permutations = {}
    for key in match_keys:
        Eo = reprs_old.get(key, None)
        En = reprs_new.get(key, None)
        if Eo is None or En is None:
            continue
        C = compute_cost_matrix(Eo, En)  # (n_old, n_new)
        perm = greedy_match_from_cost(C)  # numpy array length n_old -> new_idx or -1
        layer_permutations[key] = perm

    k = float(args.get("ensemble_step", 0.6)) if isinstance(args, dict) else float(getattr(args, 'ensemble_step', 0.6))
    fused_state = apply_permutation_and_fuse_state(state_old, state_new, layer_permutations, k=k, device=device)
    return fused_state

# -------------------------------
# End helper functions
# -------------------------------

init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 160]
init_lr_decay = 0.1
init_weight_decay = 0.0005

# cifar100
epochs = 100
lrate = 0.05
milestones = [45, 90]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2
lamda = 10

# Tiny-ImageNet200
# epochs = 100
# lrate = 0.001
# milestones = [45, 90]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay = 2e-4
# num_workers = 8
# T = 2
# lamda = 10

# imagenet100
# epochs = 100
# lrate = 0.05
# milestones = [45, 90]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay = 2e-4
# num_workers = 8
# T = 2
# lamda = 5


# fine-grained dataset
# init_lr = 0.01
# lrate = 0.005
# lamda = 20

# refer to supplementary materials for other dataset training settings

EPSILON = 1e-8

class LwI(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        if self.args["dataset"] == "imagenet100" or self.args["dataset"] == "imagenet1000":
            epochs = 100
            lrate = 0.05
            milestones = [45, 90]
            lrate_decay = 0.1
            batch_size = 128
            weight_decay = 2e-4
            num_workers = 8
            T = 2
            lamda = 5
            self.num_per_class = 1300
        elif self.args["dataset"] == "tinyimagenet200":
            epochs = 100
            lrate = 0.001
            milestones = [45, 90]
            lrate_decay = 0.1
            batch_size = 128
            weight_decay = 2e-4
            num_workers = 8
            T = 2
            lamda = 10
        print("Number of samples per class:{}".format(self.num_per_class))
        if self.args["dataset"] == "cub200":
            init_lr = 0.1
            lrate = 0.05
            lamda = 20
            self.num_per_class = 30
        if self.args["cosine"]:
            self._network = CosineIncrementalNet(args, False)
        else:
            self._network = IncrementalNet(args, False)

        self._protos = []
        self.al_classifier = None
        if self.args["DPCR"]:
            self._covs = []
            self._projectors = []

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        if not self.args['resume']:
            if not os.path.exists(self.args["model_dir"]):
                os.makedirs(self.args["model_dir"])
            self.save_checkpoint("{}".format(self.args["model_dir"]))
        

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        if self.args['dataset'] == "cifar100":
            self.data_manager._train_trsf = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63/255),
                CIFAR10Policy(),
                transforms.ToTensor(),
            ]
        elif self.args['dataset'] == "tinyimagenet200":
            self.data_manager._train_trsf = [
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.ToPILImage()
            ]
        elif self.args['dataset'] == "imagenet100" or self.args['dataset'] == "cub200":
            self.data_manager._train_trsf = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.ToPILImage()
            ]
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        if self.args["cosine"]:
            self._network.update_fc(self._total_classes, self._cur_task)
        else:
            self._network.update_fc(self._total_classes)

        if self.al_classifier == None:
            self.al_classifier = ALClassifier(512, self._total_classes, 0, self._device,args=self.args).to(self._device)
            for name, param in self.al_classifier.named_parameters():
                param.requires_grad = False
        else:
            self.al_classifier.augment_class(data_manager.get_task_size(self._cur_task))
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        self.shot = None
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            shot=self.shot
        )
        # self.train_dataset = train_dataset
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        resume = self.args['resume']  # set resume=True to use saved checkpoints
        if self._cur_task == 0:
            if resume:
                print("Loading checkpoint: {}{}_model.pth.tar".format(self.args["model_dir"], self._total_classes))
                self._network.load_state_dict(torch.load("{}{}_model.pth.tar".format(self.args["model_dir"], self._total_classes))["state_dict"], strict=False)
            self._network.to(self._device)
            if hasattr(self._network, "module"):
                self._network_module_ptr = self._network.module
            if not resume:
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=init_lr, weight_decay=init_weight_decay)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay)
                self._init_train(train_loader, test_loader, optimizer, scheduler)

            self._network.eval()
            pbar = tqdm(enumerate(train_loader), desc='Analytic Learning Phase=' + str(self._cur_task),
                             total=len(train_loader),
                             unit='batch')
            cov = torch.zeros(self.al_classifier.fe_size, self.al_classifier.fe_size).to(self._device)
            crs_cor = torch.zeros(self.al_classifier.fc.weight.size(1), self._total_classes).to(self._device)
            with torch.no_grad():
                for i, (_, inputs, targets) in pbar:
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    out_backbone = self._network(inputs)["features"]
                    out_fe, pred = self.al_classifier(out_backbone)
                    label_onehot = F.one_hot(targets, self._total_classes).float()
                    cov += torch.t(out_fe) @ out_fe
                    crs_cor += torch.t(out_fe) @ (label_onehot)
            self.al_classifier.cov = self.al_classifier.cov + cov
            self.al_classifier.R = self.al_classifier.R + cov
            self.al_classifier.Q = self.al_classifier.Q + crs_cor
            R_inv = torch.inverse(self.al_classifier.R.cpu()).to(self._device)
            Delta = R_inv @ self.al_classifier.Q

            self.al_classifier.fc.weight = torch.nn.parameter.Parameter(
                    F.normalize(torch.t(Delta.float()), p=2, dim=-1))
            self._build_protos()
        else:
            resume = self.args['resume']
            if resume:
                print("Loading checkpoint: {}{}_model.pth.tar".format(self.args["model_dir"], self._total_classes))
                self._network.load_state_dict(torch.load("{}{}_model.pth.tar".format(self.args["model_dir"], self._total_classes))["state_dict"], strict=False)
            self._network.to(self._device)
            if hasattr(self._network, "module"):
                self._network_module_ptr = self._network.module
            if self._old_network is not None:
                self._old_network.to(self._device)
            if not resume:
                optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
                self._update_representation(train_loader, test_loader, optimizer, scheduler)
            self._build_protos()                

            if hasattr(self, '_old_network') and self._old_network is not None:
                try:
                    # Save teacher old copy for KD
                    teacher_old = deepcopy(self._old_network).to(self._device)
                    teacher_old.eval()
                    # Perform LwI fusion: returns fused_state_dict to load into self._network
                    fused_state = perform_lwi_fusion(self._old_network, self._network, self.args, device=self._device)
                    # Load fused state into self._network (overwrite parameters)
                    # careful about strict - keep same keys
                    # convert fused_state (cpu tensors) -> device
                    fused_state_device = {k: v.to(self._device) for k, v in fused_state.items()}
                    # load into model (non-strict to avoid small mismatches)
                    self._network.load_state_dict(fused_state_device, strict=False)
                    logging.info("LwI fusion applied: loaded fused weights into model_new.")
                except Exception as e:
                    logging.exception("LwI fusion failed, falling back to original model. Error: {}".format(e))
            else:
                teacher_old = None       
                    
            if self.args["DPCR"]:
                print('Using DPCR')
                self._network.eval()
                self.projector = Drift_Estimator(512,False,self.args)
                self.projector.to(self._device)
                for name, param in self.projector.named_parameters():
                    param.requires_grad = False
                self.projector.eval()
                cov_pwdr = self.projector.rg_tssp * torch.eye(self.projector.fe_size).to(self._device)
                crs_cor_pwdr = torch.zeros(self.projector.fe_size, self.projector.fe_size).to(self._device)

                crs_cor_new = torch.zeros(self.al_classifier.fc.weight.size(1), self._total_classes).to(self._device)
                cov_new = torch.zeros(self.projector.fe_size, self.projector.fe_size).to(self._device)
                with torch.no_grad():
                    for i, (_, inputs, targets) in enumerate(train_loader):
                        inputs, targets = inputs.to(self._device), targets.to(self._device)
                        feats_old = self._old_network(inputs)["features"]
                        feats_new = self._network(inputs)["features"]
                        cov_pwdr += torch.t(feats_old) @ feats_old
                        cov_new += torch.t(feats_new) @ feats_new
                        crs_cor_pwdr += torch.t(feats_old) @ (feats_new)
                        label_onehot = F.one_hot(targets, self._total_classes).float()
                        crs_cor_new += torch.t(feats_new) @ (label_onehot)
                self.projector.cov = cov_pwdr
                self.projector.Q = crs_cor_pwdr
                R_inv = torch.inverse(cov_pwdr.cpu()).to(self._device)
                Delta = R_inv @ crs_cor_pwdr
                self.projector.fc.weight = torch.nn.parameter.Parameter(torch.t(Delta.float()))

                cov_prime = torch.zeros(self.al_classifier.fe_size, self.al_classifier.fe_size).to(self._device)
                Q_prime = torch.zeros(self.al_classifier.fe_size, self.al_classifier.num_classes).to(self._device)

                for class_idx in range(0, self._known_classes):
                    W = self.projector.get_weight() @ self._projectors[class_idx]
                    cov_idx = self._covs[class_idx]
                    cov_prime_idx = torch.t(W) @ cov_idx @ W
                    label = class_idx
                    label_onehot = F.one_hot(torch.tensor(label).long().to(self._device), self._total_classes).float()
                    cor_prime_idx = self.num_per_class * (torch.t(W) @ torch.t(
                        self._protos[class_idx].view(1, self.al_classifier.fe_size))) @ label_onehot.view(1, self._total_classes)
                    cov_prime += cov_prime_idx
                    Q_prime += cor_prime_idx
                    self._covs[class_idx] = cov_prime_idx
                    self._projectors[class_idx] = self.get_projector_svd(cov_prime_idx)
                    self._protos[class_idx] = self._protos[class_idx] @ W

                R_prime = cov_prime + self.al_classifier.gamma * torch.eye(self.al_classifier.fe_size).to(self._device)
                self.al_classifier.cov = cov_prime + cov_new
                self.al_classifier.Q = Q_prime + crs_cor_new
                self.al_classifier.R = R_prime+ cov_new
                R_inv = torch.inverse(self.al_classifier.R.cpu()).to(self._device)
                Delta = R_inv @ self.al_classifier.Q
                self.al_classifier.fc.weight = torch.nn.parameter.Parameter(
                        F.normalize(torch.t(Delta.float()), p=2, dim=-1))





    # SVD for calculating the W_c
    def get_projector_svd(self, raw_matrix, all_non_zeros=True):
        V, S, VT = torch.svd(raw_matrix)
        if all_non_zeros:
            non_zeros_idx = torch.where(S > 0)[0]
            left_eign_vectors = V[:, non_zeros_idx]

        else:
            left_eign_vectors = V[:, :512]
        projector = left_eign_vectors @ torch.t(left_eign_vectors)
        return projector

    def _build_protos(self):
        if self.args["DPCR"]:
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx + 1),
                                                                           source='train',
                                                                           mode='test', shot=self.shot, ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)  # vectors.mean(0)
                cov = np.dot(np.transpose(vectors),vectors)
                self._protos.append(torch.tensor(class_mean).to(self._device))
                self._covs.append(torch.tensor(cov).to(self._device))
                self._projectors.append(self.get_projector_svd(self._covs[class_idx]))


    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 25 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):

        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                fake_targets = targets - self._known_classes
                loss_clf = F.cross_entropy(
                    logits[:, self._known_classes :], fake_targets
                )
                loss_kd = _KD_loss(
                    logits[:, : self._known_classes],
                    self._old_network(inputs)["logits"],
                    T,
                )

                loss = lamda * loss_kd + loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                with torch.no_grad():
                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 25 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

