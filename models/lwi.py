import logging
import numpy as np
import torch
import os
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet, CosineIncrementalNet, Drift_Estimator, ALClassifier
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from torchvision import transforms
from utils.autoaugment import CIFAR10Policy
from scipy.optimize import linear_sum_assignment
import math

init_epoch = 2  # test
init_lr = 0.1
init_milestones = [60, 120, 160]
init_lr_decay = 0.1
init_weight_decay = 0.0005

# cifar100
epochs = 100
lrate = 0.05
milestones = [45, 90]
lrate_decay = 0.1
batch_size = 64  # Giảm từ 128 để giảm RAM
weight_decay = 2e-4
num_workers = 0  # Tắt workers để giảm RAM
T = 2
lamda = 10
k = 0.5  # fuse blend

EPSILON = 1e-8


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


class LwI(BaseLearner):
    """
    Bản đã vá để tránh OOM và fix shape mismatch khi fuse weight.
    """

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # hyper
        self.tau = args.get("tau", 0.1)
        self.E = args.get("E", 1000)
        self.num_per_class = args.get("num_per_class", 500)

        # dataset-specific
        ds = self.args.get("dataset", "").lower()
        if ds in ["imagenet100", "imagenet1000"]:
            self.num_per_class = 1300
        elif ds == "tinyimagenet200":
            self.num_per_class = 500
        elif ds == "cub200":
            self.num_per_class = 30

        print(f"Number of samples per class:{self.num_per_class}")

        # backbone
        if self.args.get("cosine", False):
            self._network = CosineIncrementalNet(args, False)
        else:
            self._network = IncrementalNet(args, False)
        self._network.split_layers()

        self._old_network = None
        # đổi sang dict để index theo class id
        self._protos = {}
        self._covs = {}
        self._projectors = {}

        # AL classifier
        self.al_classifier = ALClassifier(512, 0, 0, self._device, args=self.args).to(self._device)
        for _, p in self.al_classifier.named_parameters():
            p.requires_grad = False

        self._known_classes = 0

    # ----------------------- life-cycle -----------------------

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        if not self.args.get('resume', False):
            if not os.path.exists(self.args["model_dir"]):
                os.makedirs(self.args["model_dir"])
            # hợp nhất trọng số giữa old và new (align kênh)
            self._fuse_weights()
            self.save_checkpoint("{}".format(self.args["model_dir"]))

    # ----------------------- weight fusion -----------------------

    @torch.no_grad()
    def _fuse_weights(self):
        """
        Align & fuse theo chiều out_features/out_channels.
        """
        if self._old_network is None or self._network is None:
            return

        for layer in range(2, 5):  # các layer đã split
            W_old = self._old_network.get_layer_weights(layer)
            W_new = self._network.get_layer_weights(layer)

            if W_old is None or W_new is None:
                continue

            W_old = W_old.to(dtype=torch.float32)
            W_new = W_new.to(dtype=torch.float32)

            W_old_2d, shape_old = self._flatten_out_first(W_old)
            W_new_2d, shape_new = self._flatten_out_first(W_new)

            n_out_old, feat_old = W_old_2d.shape
            n_out_new, feat_new = W_new_2d.shape

            if feat_old != feat_new:
                logging.warning(
                    f"[Fuse] skip layer {layer}: feature dim mismatch old({feat_old}) vs new({feat_new})"
                )
                continue

            R = self._compute_similarity(W_old_2d, W_new_2d, batch_size=1024, use_cpu=True)
            P = self._compute_permutation_matrix(R, layer)
            P_left = P.t().contiguous().to(dtype=torch.float32)

            if P_left.shape != (n_out_new, n_out_old):
                raise ValueError(f"P_left shape invalid: {P_left.shape}, expected {(n_out_new, n_out_old)}")

            device_for_matmul = torch.device("cpu")  # Chuyển matmul sang CPU
            aligned_old = torch.matmul(P_left.to(device_for_matmul), W_old_2d.to(device_for_matmul))

            if layer == 2:
                W_fused_2d = aligned_old
            else:
                W_fused_2d = k * aligned_old + (1 - k) * W_new_2d.to(device_for_matmul)

            W_fused = W_fused_2d.view(*shape_new).to(self._device)
            self._old_network.set_layer_weights(layer, W_fused)

            # Giải phóng bộ nhớ
            del R, P, P_left, W_old_2d, W_new_2d, aligned_old, W_fused_2d
            torch.cuda.empty_cache()

    def _flatten_out_first(self, W):
        """
        Đưa trọng số về (n_out, -1) theo chuẩn PyTorch:
        - Linear: (out_features, in_features) -> (out_features, in_features)
        - Conv: (out_channels, in_channels, kH, kW) -> (out_channels, in_channels*kH*kW)
        - BatchNorm/others 1D: (num_features,) -> (num_features, 1)
        Trả về (W_2d, original_shape)
        """
        if W.dim() == 2:
            shape = W.shape
            return W, shape
        elif W.dim() >= 3:
            n_out = W.shape[0]
            # Lưu shape ban đầu để reshape lại: (n_out, c, h, w, ...)
            orig_shape = (n_out, *W.shape[1:])
            return W.view(n_out, -1), orig_shape
        elif W.dim() == 1:
            n_out = W.shape[0]
            return W.view(n_out, 1), (n_out,)
        else:
            raise ValueError(f"Unsupported weight dim: {W.shape}")

    def _compute_similarity(self, W_old, W_new, batch_size=1024, use_cpu=True):
        """
        Tính cosine similarity giữa các hàng của W_old và W_new theo batch.
        Trả về ma trận R kích thước (n_out_old, n_out_new).
        """
        W_old_2d = W_old.view(W_old.size(0), -1).float()
        W_new_2d = W_new.view(W_new.size(0), -1).float()

        device_calc = torch.device("cpu")  # Luôn dùng CPU để tránh VRAM spike
        W_old_norm = F.normalize(W_old_2d.to(device_calc), p=2, dim=1)
        W_new_norm = F.normalize(W_new_2d.to(device_calc), p=2, dim=1)

        n_old = W_old_norm.size(0)
        sims_parts = []
        for start in range(0, n_old, batch_size):
            end = min(start + batch_size, n_old)
            part = torch.matmul(W_old_norm[start:end], W_new_norm.t())
            sims_parts.append(part.cpu())  # Lưu trên CPU
            del part  # Xóa ngay để giải phóng RAM
            torch.cuda.empty_cache()  # Giải phóng VRAM nếu có
        R = torch.cat(sims_parts, dim=0)
        del sims_parts, W_old_norm, W_new_norm  # Giải phóng bộ nhớ
        return R

    def _compute_permutation_matrix(self, R, layer):
        """
        Trả về P kích thước (n_old, n_new).
        Với tau < 0.1 -> Hungarian; ngược lại -> Sinkhorn.
        """
        if R.numel() == 0 or R.dim() != 2:
            raise ValueError(f"Input R invalid: {R.shape}")

        cost = -R
        n_old, n_new = R.shape

        if self.tau < 0.1:
            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            P = torch.zeros((n_old, n_new), device=torch.device("cpu"), dtype=torch.float32)
            P[row_ind, col_ind] = 1.0
            return P
        else:
            a = torch.full((n_old,), 1.0 / n_old, dtype=torch.float32, device=torch.device("cpu"))
            b = torch.full((n_new,), 1.0 / n_new, dtype=torch.float32, device=torch.device("cpu"))
            M = (-R).to(dtype=torch.float32)
            P = self.sinkhorn_torch(M=M, a=a, b=b, lambda_sh=1.0, numItermax=1000, stopThr=1e-4, cuda=False)
            return P

    def sinkhorn_torch(self, M, a, b, lambda_sh, numItermax=1000, stopThr=1e-4, cuda=False):
        """
        Trả về P ~ argmin <P,M> s.t. P1=a, P^T1=b, P>=0
        """
        device = torch.device("cpu")  # Luôn dùng CPU
        M = M.to(dtype=torch.float32, device=device)
        a = a.to(dtype=torch.float32, device=device)
        b = b.to(dtype=torch.float32, device=device)

        with torch.no_grad():
            K = torch.exp(-lambda_sh * M).clamp_min(1e-12)
            u = torch.ones_like(a) / a.size(0)
            v = torch.ones_like(b) / b.size(0)

            err = float('inf')
            for it in range(int(numItermax)):
                Kv = K @ v
                Kv = Kv.clamp_min(1e-12)
                u = a / Kv

                KTu = K.t() @ u
                KTu = KTu.clamp_min(1e-12)
                v = b / KTu

                if it % 20 == 0:
                    b_est = v * (K.t() @ u)
                    err = torch.norm(b_est - b, p=float('inf')).item()
                    if err < stopThr:
                        break

            P = (u[:, None] * K) * v[None, :]
            del K, u, v  # Giải phóng tensor tạm
            torch.cuda.empty_cache()
        return P

    # ----------------------- training -----------------------

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1

        # augment hợp lý: ToTensor cuối cùng
        ds = self.args.get('dataset', '').lower()
        if ds == "cifar100":
            self.data_manager._train_trsf = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63/255),
                CIFAR10Policy(),
                transforms.ToTensor(),
            ]
        elif ds == "tinyimagenet200":
            self.data_manager._train_trsf = [
                transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        elif ds in ["imagenet100", "cub200"]:
            self.data_manager._train_trsf = [
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]

        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        if self.args.get("cosine", False):
            self._network.update_fc(self._total_classes, self._cur_task)
        else:
            self._network.update_fc(self._total_classes)

        if self.al_classifier is None:
            self.al_classifier = ALClassifier(512, self._total_classes, 0, self._device, args=self.args).to(self._device)
            for _, p in self.al_classifier.named_parameters():
                p.requires_grad = False
        else:
            self.al_classifier.augment_class(data_manager.get_task_size(self._cur_task))

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self.shot = None
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            shot=self.shot
        )
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        resume = self.args.get('resume', False)

        if self._cur_task == 0:
            if resume:
                checkpoint_path = f"{self.args['model_dir']}{self._total_classes}_model.pth.tar"
                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found!")
                print(f"Loading checkpoint: {checkpoint_path}")
                self._network.load_state_dict(torch.load(checkpoint_path)["state_dict"], strict=False)

            self._network.to(self._device)
            if hasattr(self._network, "module"):
                self._network_module_ptr = self._network.module

            if not resume:
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=init_lr, weight_decay=init_weight_decay)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay)
                self._init_train(train_loader, test_loader, optimizer, scheduler)

            # AL closed-form
            self._network.eval()
            cov = torch.zeros(self.al_classifier.fe_size, self.al_classifier.fe_size, device=self._device)
            crs_cor = torch.zeros(self.al_classifier.fe_size, self._total_classes, device=self._device)
            with torch.no_grad():
                for _, (_, inputs, targets) in enumerate(tqdm(train_loader, desc='Analytic Learning Phase=0', unit='batch')):
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    out_backbone = self._network(inputs)["features"]
                    out_fe, _ = self.al_classifier(out_backbone)
                    label_onehot = F.one_hot(targets, self._total_classes).float()
                    cov += out_fe.t() @ out_fe
                    crs_cor += out_fe.t() @ label_onehot

            self.al_classifier.cov = self.al_classifier.cov + cov
            self.al_classifier.R = self.al_classifier.R + cov
            self.al_classifier.Q = self.al_classifier.Q + crs_cor

            R_inv = torch.inverse(self.al_classifier.R.cpu()).to(self._device)
            Delta = R_inv @ self.al_classifier.Q
            self.al_classifier.fc.weight = torch.nn.Parameter(F.normalize(Delta.t().float(), p=2, dim=-1))

            self._build_protos()
        else:
            if resume:
                checkpoint_path = f"{self.args['model_dir']}{self._total_classes}_model.pth.tar"
                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found!")
                print(f"Loading checkpoint: {checkpoint_path}")
                self._network.load_state_dict(torch.load(checkpoint_path)["state_dict"], strict=False)

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

            # -------- DPCR/Drift Estimation (giữ nguyên logic, thêm guard shape) --------
            if self.args.get("DPCR", False):
                print('Using DPCR')
                if self.al_classifier is None:
                    raise ValueError("ALClassifier is not initialized!")

                self._network.eval()
                self.projector = Drift_Estimator(512, False, self.args).to(self._device)
                for _, p in self.projector.named_parameters():
                    p.requires_grad = False
                self.projector.eval()

                cov_pwdr = self.projector.rg_tssp * torch.eye(self.projector.fe_size, device=self._device)
                crs_cor_pwdr = torch.zeros(self.projector.fe_size, self.projector.fe_size, device=self._device)

                crs_cor_new = torch.zeros(self.al_classifier.fe_size, self._total_classes, device=self._device)
                cov_new = torch.zeros(self.projector.fe_size, self.projector.fe_size, device=self._device)

                with torch.no_grad():
                    for _, (_, inputs, targets) in enumerate(train_loader):
                        inputs, targets = inputs.to(self._device), targets.to(self._device)
                        feats_old = self._old_network(inputs)["features"]
                        feats_new = self._network(inputs)["features"]
                        cov_pwdr += feats_old.t() @ feats_old
                        cov_new += feats_new.t() @ feats_new
                        crs_cor_pwdr += feats_old.t() @ feats_new
                        label_onehot = F.one_hot(targets, self._total_classes).float()
                        crs_cor_new += feats_new.t() @ label_onehot

                self.projector.cov = cov_pwdr
                self.projector.Q = crs_cor_pwdr
                R_inv = torch.inverse(cov_pwdr.cpu()).to(self._device)
                Delta = R_inv @ crs_cor_pwdr
                self.projector.fc.weight = torch.nn.Parameter(Delta.t().float())

                cov_prime = torch.zeros(self.al_classifier.fe_size, self.al_classifier.fe_size, device=self._device)
                Q_prime = torch.zeros(self.al_classifier.fe_size, self.al_classifier.num_classes, device=self._device)

                for class_idx in range(0, self._known_classes):
                    if class_idx not in self._projectors or class_idx not in self._covs or class_idx not in self._protos:
                        # nếu thiếu, bỏ qua class này
                        continue

                    W = self.projector.get_weight() @ self._projectors[class_idx]
                    cov_idx = self._covs[class_idx]
                    cov_prime_idx = W.t() @ cov_idx @ W
                    label_onehot = F.one_hot(torch.tensor(class_idx, device=self._device), self._total_classes).float()
                    cor_prime_idx = self.num_per_class * (W.t() @ self._protos[class_idx].view(-1, 1)) @ label_onehot.view(1, self._total_classes)

                    cov_prime += cov_prime_idx
                    Q_prime += cor_prime_idx

                    # cập nhật từng class
                    self._covs[class_idx] = cov_prime_idx
                    self._projectors[class_idx] = self.get_projector_svd(cov_prime_idx)
                    self._protos[class_idx] = (self._protos[class_idx] @ W).contiguous()

                R_prime = cov_prime + self.al_classifier.gamma * torch.eye(self.al_classifier.fe_size, device=self._device)
                self.al_classifier.cov = cov_prime + cov_new
                self.al_classifier.Q = Q_prime + crs_cor_new
                self.al_classifier.R = R_prime + cov_new

                R_inv = torch.inverse(self.al_classifier.R.cpu()).to(self._device)
                Delta = R_inv @ self.al_classifier.Q
                self.al_classifier.fc.weight = torch.nn.Parameter(F.normalize(Delta.t().float(), p=2, dim=-1))

    def get_projector_svd(self, raw_matrix, all_non_zeros=True):
        """
        Dùng eigh cho ma trận đối xứng dương (cov).
        """
        # đảm bảo đối xứng
        A = 0.5 * (raw_matrix + raw_matrix.t())
        # trị riêng, vector riêng: A = V diag(S) V^T
        S, V = torch.linalg.eigh(A)  # S tăng dần
        if all_non_zeros:
            non_zeros_idx = torch.where(S > 1e-12)[0]
            if non_zeros_idx.numel() == 0:
                left_vectors = V
            else:
                left_vectors = V[:, non_zeros_idx]
        else:
            r = min(512, V.shape[1])
            left_vectors = V[:, -r:]  # các trị riêng lớn nhất
        projector = left_vectors @ left_vectors.t()
        return projector

    def _build_protos(self):
        """
        Tạo proto/cov/projector cho các class mới, lưu vào dict theo class id.
        """
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source='train',
                mode='test',
                shot=self.shot,
                ret_data=True
            )
            idx_loader = DataLoader(idx_dataset, batch_size=64, shuffle=False, num_workers=0)  # Giảm batch_size, tắt workers
            fe = self.al_classifier.fe_size
            cov_sum = torch.zeros(fe, fe, device=self._device)
            proto_sum = torch.zeros(fe, device=self._device)
            count = 0

            with torch.no_grad():
                for _, (_, inputs, _) in enumerate(idx_loader):
                    inputs = inputs.to(self._device)
                    feats = self._network(inputs)["features"]  # [batch, fe]
                    cov_sum += feats.t() @ feats  # Tích lũy covariance
                    proto_sum += feats.sum(dim=0)  # Tích lũy mean
                    count += feats.size(0)
                    del feats  # Giải phóng ngay
                    torch.cuda.empty_cache()

            class_mean = proto_sum / count
            cov_t = cov_sum / count

            if class_mean.shape[0] != fe:
                logging.warning(f"[Protos] feature size mismatch proto:{class_mean.shape[0]} vs fe:{fe}")
                if class_mean.shape[0] > fe:
                    class_mean = class_mean[:fe]
                    cov_t = cov_t[:fe, :fe]
                else:
                    pad = fe - class_mean.shape[0]
                    class_mean = F.pad(class_mean, (0, pad))
                    cov_pad = torch.zeros((fe, fe), device=self._device)
                    cov_pad[:cov_t.shape[0], :cov_t.shape[1]] = cov_t
                    cov_t = cov_pad

            self._protos[class_idx] = class_mean
            self._covs[class_idx] = cov_t
            self._projectors[class_idx] = self.get_projector_svd(cov_t)
            del cov_sum, proto_sum  # Giải phóng bộ nhớ
            torch.cuda.empty_cache()

    # ----------------------- epoch loops -----------------------

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for _, (_, inputs, targets) in enumerate(train_loader):
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
                info = f"Task {self._cur_task}, Epoch {epoch+1}/{init_epoch} => Loss {losses/len(train_loader):.3f}, Train_accy {train_acc:.2f}, Test_accy {test_acc:.2f}"
            else:
                info = f"Task {self._cur_task}, Epoch {epoch+1}/{init_epoch} => Loss {losses/len(train_loader):.3f}, Train_accy {train_acc:.2f}"
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                fake_targets = targets - self._known_classes
                loss_clf = F.cross_entropy(logits[:, self._known_classes:], fake_targets)

                loss_kd = torch.tensor(0.0, device=self._device, dtype=torch.float32)
                loss_ot = torch.tensor(0.0, device=self._device, dtype=torch.float32)

                if self._old_network is not None:
                    loss_kd = _KD_loss(logits[:, :self._known_classes], self._old_network(inputs)["logits"], T)

                    # OT alignment (tuỳ codebase của bạn)
                    try:
                        aligned_layers, _ = get_wassersteinized_layers_modularized(self.args, self._device,
                                                                                   [self._old_network, self._network])
                        for old_param, new_like in zip(self._old_network.parameters(), aligned_layers):
                            if old_param.shape == new_like.shape:
                                loss_ot += F.mse_loss(old_param, new_like)
                    except Exception as e:
                        logging.warning(f"[OT] skipped due to: {e}")

                loss = lamda * loss_kd + loss_clf + 0.1 * loss_ot

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
                info = f"Task {self._cur_task}, Epoch {epoch+1}/{epochs} => Loss {losses/len(train_loader):.3f}, Train_accy {train_acc:.2f}, Test_accy {test_acc:.2f}"
            else:
                info = f"Task {self._cur_task}, Epoch {epoch+1}/{epochs} => Loss {losses/len(train_loader):.3f}, Train_accy {train_acc:.2f}"
            prog_bar.set_description(info)

        logging.info(info)

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]