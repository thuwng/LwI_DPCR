import logging
import numpy as np
import torch
import os
import psutil
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
import traceback

# ---------------- hyperparams (same semantics as before) ----------------
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
batch_size = 16  # mặc định (cấu hình lấy từ JSON)
weight_decay = 2e-4
num_workers = 0
T = 2
lamda = 10
k = 0.5  # fuse blend

EPSILON = 1e-8

# ---------------- helper utils ----------------
def print_mem(tag=""):
    if torch.cuda.is_available():
        print(f"[MEM] {tag} Allocated={torch.cuda.memory_allocated()/1e9:.3f} GB, Reserved={torch.cuda.memory_reserved()/1e9:.3f} GB")
    else:
        print(f"[MEM] {tag} (no CUDA)")


def safe_cuda_empty():
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _KD_loss(pred, soft, T):
    # both pred and soft must be on same device
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

# Thiết lập logging để ghi ra console và file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),  # Ghi log ra file
        logging.StreamHandler()  # In log ra console
    ]
)

# ---------------- main class ----------------
class LwI(BaseLearner):
    """
    Giữ nguyên logic DPCR nhưng đảm bảo:
    - tất cả ma trận tích lũy lớn ở trên CPU
    - _fuse_weights chạy hoàn toàn trên CPU và set weights cho old_network trên CPU
    - tránh DataParallel/tránh duplicate model trên nhiều GPU
    - thêm debug memory logging (bật bằng args['debug']=True)
    """

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        # device chọn GPU nếu có, mặc định cuda:0
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._cpu = torch.device("cpu")

        # AMP dtype
        self._amp_enabled = bool(args.get("amp", True) and torch.cuda.is_available())
        self._amp_dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16

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
        # LƯU TRỮ TRÊN CPU: protos, covs, projectors -> tránh giữ trên GPU
        self._protos = {}
        self._covs = {}
        self._projectors = {}

        # AL classifier - đặt trên device (GPU) bởi vì inference/weight cuối cần trên GPU
        self.al_classifier = ALClassifier(512, 0, 0, self._device, args=self.args).to(self._device)
        for _, p in self.al_classifier.named_parameters():
            p.requires_grad = False

        self._known_classes = 0

        # Option flags
        self.no_data_parallel = args.get("no_data_parallel", True)
        self.do_ot_alignment = args.get("do_ot_alignment", False)
        self.debug_mode = args.get("debug", False)

    # ---------------- lifecycle ----------------
    def after_task(self):
        logging.info("[DEBUG] Starting after_task")
        # copy old network and keep it on CPU to save VRAM
        self._old_network = self._network.copy().freeze()
        try:
            self._old_network.to(self._cpu)
            logging.info("[DEBUG] Moved old_network to CPU")
        except Exception as e:
            logging.warning(f"[DEBUG] Failed to move old_network to CPU: {str(e)}")

        # free GPU cache
        safe_cuda_empty()

        self._known_classes = self._total_classes
        if not self.args.get('resume', False):
            if not os.path.exists(self.args["model_dir"]):
                logging.info(f"[DEBUG] Creating model directory: {self.args['model_dir']}")
                os.makedirs(self.args["model_dir"])
            if self.debug_mode:
                print_mem("before _fuse_weights in after_task")
            logging.info("[DEBUG] Calling _fuse_weights")
            try:
                self._fuse_weights()
                logging.info("[DEBUG] _fuse_weights completed")
            except Exception as e:
                logging.error(f"[ERROR] Failed in _fuse_weights: {str(e)}")
                logging.error(f"[ERROR] Traceback: {traceback.format_exc()}")
                raise
            if self.debug_mode:
                print_mem("after _fuse_weights in after_task")
            logging.info("[DEBUG] Saving checkpoint")
            try:
                self.save_checkpoint("{}".format(self.args["model_dir"]))
                logging.info("[DEBUG] Checkpoint saved")
            except Exception as e:
                logging.error(f"[ERROR] Failed to save checkpoint: {str(e)}")
                logging.error(f"[ERROR] Traceback: {traceback.format_exc()}")
                raise
        safe_cuda_empty()
        logging.info("[DEBUG] after_task completed")

    # ---------------- weight similarity (CPU, blockwise) ----------------
    @torch.no_grad()
    def _compute_similarity(self, W_old, W_new, batch_size=128, use_cpu=True):
        import numpy as _np
        import os
        import psutil

        logging.info("[DEBUG] [Similarity] Starting _compute_similarity")
        # Đảm bảo ma trận trên CPU và contiguous
        try:
            W_old_2d = W_old.view(W_old.size(0), -1).float().cpu().contiguous()
            W_new_2d = W_new.view(W_new.size(0), -1).float().cpu().contiguous()
            logging.info(f"[DEBUG] [Similarity] Weights reshaped: W_old_2d.shape={W_old_2d.shape}, W_new_2d.shape={W_new_2d.shape}")
        except Exception as e:
            logging.error(f"[ERROR] [Similarity] Failed to reshape weights: {str(e)}")
            logging.error(f"[ERROR] [Similarity] Traceback: {traceback.format_exc()}")
            raise

        # Gán n_old và n_new ngay từ đầu dựa trên shape sau reshape
        n_old = W_old_2d.size(0)
        n_new = W_new_2d.size(0)
        logging.info(f"[DEBUG] [Similarity] Initial sizes: n_old={n_old}, n_new={n_new}")

        # Ghi log thông tin layer
        layer_debug = getattr(self, "_last_fuse_layer", None)
        if layer_debug is not None:
            self._log_layer_info(layer_debug, W_old_2d, W_new_2d)
            logging.info(f"[DEBUG] [Similarity] Logged layer info for layer {layer_debug}")

        # Kiểm tra shape bất thường và xử lý trực tiếp nếu cần
        if W_old_2d.shape[1] <= 1 or W_new_2d.shape[1] <= 1:
            logging.warning(f"[DEBUG] [Similarity] Unusual feature dimension (<=1) detected: W_old_2d.shape={W_old_2d.shape}, W_new_2d.shape={W_new_2d.shape}")
            logging.info("[DEBUG] [Similarity] Computing R directly in RAM with batching")
            try:
                W_old_norm = F.normalize(W_old_2d, p=2, dim=1)
                W_new_norm = F.normalize(W_new_2d, p=2, dim=1)
                R = torch.zeros(n_old, n_new, device=self._cpu)
                for i in range(0, n_old, batch_size):
                    end = min(i + batch_size, n_old)
                    R[i:end, :] = W_old_norm[i:end] @ W_new_norm.t()
                logging.info(f"[DEBUG] [Similarity] Direct computation completed, R shape={R.shape}")
                return R.cpu().numpy()
            except Exception as e:
                logging.error(f"[ERROR] [Similarity] Failed direct computation: {str(e)}")
                logging.error(f"[ERROR] [Similarity] Traceback: {traceback.format_exc()}")
                raise

        # Chuẩn hóa ma trận
        logging.info("[DEBUG] [Similarity] Starting normalization")
        try:
            W_old_norm = F.normalize(W_old_2d, p=2, dim=1)
            W_new_norm = F.normalize(W_new_2d, p=2, dim=1)
            logging.info("[DEBUG] [Similarity] Weights normalized")
        except Exception as e:
            logging.error(f"[ERROR] [Similarity] Failed to normalize weights: {str(e)}")
            logging.error(f"[ERROR] [Similarity] Traceback: {traceback.format_exc()}")
            raise

        matrix_size_bytes = n_old * n_new * 4
        matrix_size_gb = matrix_size_bytes / (1024 ** 3)
        logging.info(f"[DEBUG] [Similarity] n_old={n_old}, n_new={n_new}, R_shape={(n_old, n_new)}, Estimated R size: {matrix_size_gb:.2f} GB")

        # Kiểm tra dung lượng đĩa và RAM
        tmpdir = "/kaggle/working" if os.path.exists("/kaggle/working") else "/tmp"
        try:
            disk_usage = psutil.disk_usage(tmpdir)
            ram_usage = psutil.virtual_memory()
            logging.info(f"[DEBUG] [Similarity] Disk usage at {tmpdir}: Free={disk_usage.free / (1024 ** 3):.2f} GB, Total={disk_usage.total / (1024 ** 3):.2f} GB")
            logging.info(f"[DEBUG] [Similarity] RAM usage: Free={ram_usage.free / (1024 ** 3):.2f} GB, Total={ram_usage.total / (1024 ** 3):.2f} GB")
            if matrix_size_gb > disk_usage.free / (1024 ** 3):
                logging.error(f"[ERROR] [Similarity] Insufficient disk space for memmap: Required {matrix_size_gb:.2f} GB, Available {disk_usage.free / (1024 ** 3):.2f} GB")
                raise RuntimeError("Insufficient disk space for memmap")
            if matrix_size_gb > ram_usage.free / (1024 ** 3):
                logging.warning("[DEBUG] [Similarity] RAM may not be sufficient for direct computation, using memmap")
        except Exception as e:
            logging.error(f"[ERROR] [Similarity] Failed to check resources: {str(e)}")
            logging.error(f"[ERROR] [Similarity] Traceback: {traceback.format_exc()}")
            raise

        # Tạo file memmap
        logging.info(f"[DEBUG] [Similarity] Using tmpdir: {tmpdir}")
        try:
            if not os.path.exists(tmpdir):
                logging.info(f"[DEBUG] [Similarity] Creating directory: {tmpdir}")
                os.makedirs(tmpdir, exist_ok=True)
        except Exception as e:
            logging.error(f"[ERROR] [Similarity] Failed to create directory {tmpdir}: {str(e)}")
            logging.error(f"[ERROR] [Similarity] Traceback: {traceback.format_exc()}")
            raise
        fname = os.path.join(tmpdir, f"similarity_R_{os.getpid()}_{int(torch.randint(0, int(1e9), (1,)).item())}.dat")
        logging.info(f"[DEBUG] [Similarity] Memmap file path: {fname}")

        R_memmap = None
        try:
            logging.info(f"[DEBUG] [Similarity] Creating memmap file: {fname}")
            R_memmap = _np.memmap(fname, dtype='float32', mode='w+', shape=(n_old, n_new))
            logging.info(f"[DEBUG] [Similarity] Memmap file created: {fname}")
            W_new_np = W_new_norm.numpy()
            for start in range(0, n_old, batch_size):
                end = min(start + batch_size, n_old)
                logging.info(f"[DEBUG] [Similarity] Processing batch {start}:{end} of {n_old}")
                try:
                    block_old = W_old_norm[start:end].numpy()
                    block_sim = block_old.dot(W_new_np.T)
                    R_memmap[start:end, :] = block_sim.astype('float32')
                    R_memmap.flush()
                    logging.info(f"[DEBUG] [Similarity] Batch {start}:{end} computed and written to memmap")
                    del block_old, block_sim
                except MemoryError as me:
                    logging.error(f"[ERROR] [Similarity] MemoryError in batch {start}:{end}: {str(me)}")
                    logging.error(f"[ERROR] [Similarity] Traceback: {traceback.format_exc()}")
                    raise
                except Exception as e:
                    logging.error(f"[ERROR] [Similarity] Failed to process batch {start}:{end}: {str(e)}")
                    logging.error(f"[ERROR] [Similarity] Traceback: {traceback.format_exc()}")
                    raise
            logging.info(f"[DEBUG] [Similarity] Memmap computation completed")
            return R_memmap
        except Exception as e:
            logging.error(f"[ERROR] [Similarity] Error in _compute_similarity: {str(e)}")
            logging.error(f"[ERROR] [Similarity] Traceback: {traceback.format_exc()}")
            raise
        finally:
            if R_memmap is not None and hasattr(R_memmap, 'filename') and os.path.exists(R_memmap.filename):
                try:
                    file_size = os.path.getsize(R_memmap.filename) / (1024 ** 3)
                    logging.info(f"[DEBUG] [Similarity] Removing memmap file: {R_memmap.filename}, size={file_size:.2f} GB")
                    os.remove(R_memmap.filename)
                except Exception as e:
                    logging.warning(f"[DEBUG] [Similarity] Failed to remove memmap file {R_memmap.filename}: {str(e)}")
            del W_old_norm, W_new_norm
            safe_cuda_empty()
            logging.info("[DEBUG] [Similarity] Cleared memory after _compute_similarity")

    # ---------------- weight fusion ----------------
    @torch.no_grad()
    def _fuse_weights(self):
        if self._old_network is None or self._network is None:
            logging.info("[DEBUG] [Fuse] Skipping _fuse_weights: old_network or network is None")
            return

        HUNGARIAN_MAX = self.args.get("hungarian_max", 3000)
        prev_P = None  # align chuỗi convs

        for layer in range(2, 5):
            logging.info(f"[DEBUG] [Fuse] Processing layer {layer}")
            W_old = self._old_network.get_layer_weights(layer)
            W_new = self._network.get_layer_weights(layer)

            if W_old is None or W_new is None:
                logging.warning(f"[DEBUG] [Fuse] Skipping layer {layer}: W_old or W_new is None")
                continue

            is_deep = (layer >= 4)

            # Chuyển ma trận sang CPU
            logging.info(f"[DEBUG] [Fuse] Moving weights to CPU for layer {layer}")
            try:
                W_old = W_old.detach().float().to(self._cpu)
                W_new = W_new.detach().float().to(self._cpu)
                logging.info(f"[DEBUG] [Fuse] Weights moved to CPU for layer {layer}")
            except Exception as e:
                logging.error(f"[ERROR] [Fuse] Failed to move weights to CPU for layer {layer}: {str(e)}")
                logging.error(f"[ERROR] [Fuse] Traceback: {traceback.format_exc()}")
                raise

            orig_shape_old = W_old.shape
            if prev_P is not None and len(orig_shape_old) == 4:
                logging.info(f"[DEBUG] [Fuse] Aligning input channels with prev_P for layer {layer}")
                try:
                    nout, nin, h, w = orig_shape_old
                    W_old_resh = W_old.view(nout, nin, -1)
                    if prev_P.shape[0] == nin:
                        W_old_resh_aligned = torch.einsum('oik, ij -> ojk', W_old_resh, prev_P)
                        W_old = W_old_resh_aligned.view(nout, prev_P.shape[1], h, w)
                        logging.info(f"[DEBUG] [Fuse] Input channels aligned for layer {layer}")
                    else:
                        logging.warning(f"[DEBUG] [Fuse] Skipping alignment for layer {layer}: prev_P shape {prev_P.shape} mismatch with nin {nin}")
                except Exception as e:
                    logging.error(f"[ERROR] [Fuse] Failed to align input channels for layer {layer}: {str(e)}")
                    logging.error(f"[ERROR] [Fuse] Traceback: {traceback.format_exc()}")
                    raise

            logging.info(f"[DEBUG] [Fuse] Flattening weights for layer {layer}")
            try:
                W_old_2d, shape_old = self._flatten_out_first(W_old)
                W_new_2d, shape_new = self._flatten_out_first(W_new)
                logging.info(f"[DEBUG] [Fuse] Weights flattened: W_old_2d.shape={W_old_2d.shape}, W_new_2d.shape={W_new_2d.shape}")
            except Exception as e:
                logging.error(f"[ERROR] [Fuse] Failed to flatten weights for layer {layer}: {str(e)}")
                logging.error(f"[ERROR] [Fuse] Traceback: {traceback.format_exc()}")
                raise

            n_out_old, feat_old = W_old_2d.shape
            n_out_new, feat_new = W_new_2d.shape

            if feat_old != feat_new:
                logging.warning(f"[DEBUG] [Fuse] Skipping layer {layer}: feature dim mismatch old({feat_old}) vs new({feat_new})")
                continue

            if self.debug_mode:
                print_mem(f"fuse_weights layer {layer} - before similarity")

            self._last_fuse_layer = layer
            R = None
            try:
                logging.info(f"[DEBUG] [Fuse] Starting similarity computation for layer {layer}")
                R = self._compute_similarity(W_old_2d, W_new_2d, batch_size=512, use_cpu=True)
                logging.info(f"[DEBUG] [Fuse] Similarity matrix R computed for layer {layer}, shape={R.shape}")

                logging.info(f"[DEBUG] [Fuse] Computing permutation matrix for layer {layer}")
                P = self._compute_permutation_matrix(R, layer, is_deep=is_deep)
                logging.info(f"[DEBUG] [Fuse] Permutation matrix P computed for layer {layer}, shape={P.shape}")

                P_left = P.t().contiguous().to(dtype=torch.float32, device=self._cpu)
                if P_left.shape != (n_out_new, n_out_old):
                    logging.error(f"[ERROR] [Fuse] P_left shape invalid for layer {layer}: {P_left.shape}, expected {(n_out_new, n_out_old)}")
                    raise ValueError(f"P_left shape invalid: {P_left.shape}, expected {(n_out_new, n_out_old)}")
                logging.info(f"[DEBUG] [Fuse] P_left computed for layer {layer}, shape={P_left.shape}")

                logging.info(f"[DEBUG] [Fuse] Computing fused weights for layer {layer}")
                aligned_old = torch.matmul(P_left, W_old_2d)
                if layer == 2:
                    W_fused_2d = aligned_old
                else:
                    W_fused_2d = k * aligned_old + (1 - k) * W_new_2d
                logging.info(f"[DEBUG] [Fuse] Fused weights computed for layer {layer}")

                W_fused = W_fused_2d.view(*shape_new).to(self._cpu)
                self._old_network.set_layer_weights(layer, W_fused)
                logging.info(f"[DEBUG] [Fuse] Set fused weights for layer {layer}")
                prev_P = P
            except Exception as e:
                logging.error(f"[ERROR] [Fuse] Failed during weight fusion for layer {layer}: {str(e)}")
                logging.error(f"[ERROR] [Fuse] Traceback: {traceback.format_exc()}")
                raise
            finally:
                if R is not None and hasattr(R, 'filename'):
                    try:
                        if os.path.exists(R.filename):
                            file_size = os.path.getsize(R.filename) / (1024 ** 3)
                            logging.info(f"[DEBUG] [Fuse] Removing memmap file: {R.filename}, size={file_size:.2f} GB")
                            os.remove(R.filename)
                        else:
                            logging.info(f"[DEBUG] [Fuse] Memmap file {R.filename} does not exist")
                    except Exception as e:
                        logging.warning(f"[DEBUG] [Fuse] Failed to remove memmap file {getattr(R,'filename','?')}: {str(e)}")
                del R
                safe_cuda_empty()
                logging.info(f"[DEBUG] [Fuse] Cleared memory after processing layer {layer}")

            if self.debug_mode:
                print_mem(f"fuse_weights layer {layer} - after processing")

        if self.debug_mode:
            print_mem("after _fuse_weights")
            disk_usage = psutil.disk_usage('/kaggle/working' if os.path.exists('/kaggle/working') else '/tmp')
            logging.info(f"[DEBUG] [Fuse] Final disk usage: Free={disk_usage.free / (1024 ** 3):.2f} GB, Total={disk_usage.total / (1024 ** 3):.2f} GB")

    def _flatten_out_first(self, W):
        logging.info(f"[DEBUG] [Flatten] Input weight shape: {W.shape}")
        try:
            if W.dim() == 2:
                shape = W.shape
                logging.info(f"[DEBUG] [Flatten] Output shape: {shape}")
                return W, shape
            elif W.dim() >= 3:
                n_out = W.shape[0]
                orig_shape = (n_out, *W.shape[1:])
                flattened = W.view(n_out, -1)
                logging.info(f"[DEBUG] [Flatten] Flattened to shape: {flattened.shape}, original shape: {orig_shape}")
                return flattened, orig_shape
            elif W.dim() == 1:
                n_out = W.shape[0]
                flattened = W.view(n_out, 1)
                logging.info(f"[DEBUG] [Flatten] Flattened to shape: {flattened.shape}")
                return flattened, (n_out,)
            else:
                logging.error(f"[ERROR] [Flatten] Unsupported weight dim: {W.shape}")
                raise ValueError(f"Unsupported weight dim: {W.shape}")
        except Exception as e:
            logging.error(f"[ERROR] [Flatten] Failed to flatten weights: {str(e)}")
            logging.error(f"[ERROR] [Flatten] Traceback: {traceback.format_exc()}")
            raise

    def _log_layer_info(self, layer, W_old_2d, W_new_2d):
        try:
            print(f"[LAYER INFO] layer={layer} W_old_2d.shape={tuple(W_old_2d.shape)} W_new_2d.shape={tuple(W_new_2d.shape)}")
        except Exception:
            pass

    def _approx_permutation_from_memmap(self, R, layer):
        import numpy as _np

        logging.info(f"[DEBUG] [ApproxPerm] Starting _approx_permutation_from_memmap for layer {layer}")
        n_old, n_new = R.shape
        best_vals = -1e9 * _np.ones((n_new,), dtype='float32')
        best_idxs = -1 * _np.ones((n_new,), dtype='int64')

        BLOCK = 1024
        for start in range(0, n_old, BLOCK):
            end = min(start + BLOCK, n_old)
            logging.info(f"[DEBUG] [ApproxPerm] Processing batch {start}:{end} of {n_old} for layer {layer}")
            try:
                block = R[start:end, :]  # (block_size, n_new)
                block_max = block.max(axis=0)
                block_arg = block.argmax(axis=0)

                mask = block_max > best_vals
                if mask.any():
                    best_vals[mask] = block_max[mask]
                    best_idxs[mask] = (start + block_arg[mask])
                logging.info(f"[DEBUG] [ApproxPerm] Batch {start}:{end} processed for layer {layer}")
            except Exception as e:
                logging.error(f"[ERROR] [ApproxPerm] Failed to process batch {start}:{end} for layer {layer}: {str(e)}")
                logging.error(f"[ERROR] [ApproxPerm] Traceback: {traceback.format_exc()}")
                raise

        logging.info(f"[DEBUG] [ApproxPerm] Forming candidate list for layer {layer}")
        try:
            candidates = []
            for new_idx in range(n_new):
                old_idx = int(best_idxs[new_idx])
                val = float(best_vals[new_idx])
                if old_idx >= 0:
                    candidates.append((val, new_idx, old_idx))

            candidates.sort(reverse=True, key=lambda x: x[0])

            assigned_old = set()
            assigned_new = set()
            pairs = []
            for val, new_idx, old_idx in candidates:
                if old_idx in assigned_old or new_idx in assigned_new:
                    continue
                assigned_old.add(old_idx)
                assigned_new.add(new_idx)
                pairs.append((old_idx, new_idx))
                if len(assigned_old) >= min(n_old, n_new):
                    break

            P = torch.zeros((n_old, n_new), device=self._cpu, dtype=torch.float32)
            for old_idx, new_idx in pairs:
                if 0 <= old_idx < n_old and 0 <= new_idx < n_new:
                    P[old_idx, new_idx] = 1.0
            logging.info(f"[DEBUG] [ApproxPerm] Permutation matrix formed for layer {layer}, P shape={P.shape}")
            return P
        except Exception as e:
            logging.error(f"[ERROR] [ApproxPerm] Failed to form permutation matrix for layer {layer}: {str(e)}")
            logging.error(f"[ERROR] [ApproxPerm] Traceback: {traceback.format_exc()}")
            raise

    def _compute_permutation_matrix(self, R, layer, is_deep=False):
        import numpy as _np

        logging.info(f"[DEBUG] [Permutation] Starting _compute_permutation_matrix for layer {layer}")
        # Lấy kích thước không ép copy
        try:
            if isinstance(R, (_np.memmap, _np.ndarray)):
                n_old, n_new = R.shape
            elif torch.is_tensor(R):
                n_old, n_new = R.shape
                R = R.cpu().numpy()
            else:
                logging.error(f"[ERROR] [Permutation] Unsupported R type: {type(R)}")
                raise ValueError(f"Unsupported R type: {type(R)}")
            logging.info(f"[DEBUG] [Permutation] R shape: ({n_old}, {n_new})")
        except Exception as e:
            logging.error(f"[ERROR] [Permutation] Failed to get R shape: {str(e)}")
            logging.error(f"[ERROR] [Permutation] Traceback: {traceback.format_exc()}")
            raise

        if n_old == 0 or n_new == 0:
            logging.error(f"[ERROR] [Permutation] Input R invalid: ({n_old}, {n_new})")
            raise ValueError(f"Input R invalid: {(n_old, n_new)}")

        HUNGARIAN_MAX = self.args.get("hungarian_max", 3000)

        if min(n_old, n_new) <= HUNGARIAN_MAX:
            logging.info(f"[DEBUG] [Permutation] Using Hungarian algorithm for layer {layer}")
            try:
                cost = -R
                row_ind, col_ind = linear_sum_assignment(cost)
                P = torch.zeros((n_old, n_new), device=self._cpu, dtype=torch.float32)
                P[row_ind, col_ind] = 1.0
                logging.info(f"[DEBUG] [Permutation] Hungarian completed for layer {layer}, P shape={P.shape}")
                return P
            except Exception as e:
                logging.warning(f"[ERROR] [Permutation] Hungarian failed at layer {layer} ({n_old}x{n_new}): {str(e)}")
                logging.warning(f"[ERROR] [Permutation] Traceback: {traceback.format_exc()}")
                logging.info(f"[DEBUG] [Permutation] Falling back to approx matching for layer {layer}")

        if self.debug_mode:
            logging.warning(f"[DEBUG] [Permutation] Using approx matching for layer {layer} ({n_old}x{n_new})")
        try:
            P = self._approx_permutation_from_memmap(R, layer)
            logging.info(f"[DEBUG] [Permutation] Approx permutation completed for layer {layer}, P shape={P.shape}")
            return P
        except Exception as e:
            logging.error(f"[ERROR] [Permutation] Failed in approx permutation for layer {layer}: {str(e)}")
            logging.error(f"[ERROR] [Permutation] Traceback: {traceback.format_exc()}")
            raise

    # ---------------- training ----------------
    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1

        # augment transforms
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
        bs = self.args.get("batch_size", batch_size)
        nw = self.args.get("num_workers", num_workers)

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            shot=self.shot
        )
        self.train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

        # avoid DataParallel if requested
        if not self.no_data_parallel and len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        # ensure network (current) on GPU for training
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module

        self._train(self.train_loader, self.test_loader)

        # if wrapped, unwrap
        if not self.no_data_parallel and len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        resume = self.args.get('resume', False)

        if self._cur_task == 0:
            if resume:
                checkpoint_path = f"{self.args['model_dir']}{self._total_classes}_model.pth.tar"
                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found!")
                print(f"Loading checkpoint: {checkpoint_path}")
                self._network.load_state_dict(torch.load(checkpoint_path, map_location=self._device)["state_dict"], strict=False)

            if not resume:
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=init_lr, weight_decay=init_weight_decay)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay)
                self._init_train(train_loader, test_loader, optimizer, scheduler)

            # --- AL closed-form: do accumulation on CPU to save VRAM ---
            if self.debug_mode:
                print_mem("before AL closed-form (task 0)")

            cov = torch.zeros(self.al_classifier.fe_size, self.al_classifier.fe_size, device=self._cpu)
            crs_cor = torch.zeros(self.al_classifier.fe_size, self._total_classes, device=self._cpu)
            with torch.inference_mode():
                for _, (_, inputs, targets) in enumerate(tqdm(train_loader, desc='Analytic Learning Phase=0', unit='batch')):
                    inputs_cpu = inputs  # dataloader yields CPU tensors
                    inputs_gpu = inputs_cpu.to(self._device, non_blocking=True)
                    # AMP reduce VRAM
                    if self._amp_enabled and torch.cuda.is_available():
                        with torch.amp.autocast(device_type="cuda", dtype=self._amp_dtype):
                            out_backbone = self._network(inputs_gpu)["features"]
                    else:
                        out_backbone = self._network(inputs_gpu)["features"]

                    out_backbone = out_backbone.to(torch.float32)
                    out_fe, _ = self.al_classifier(out_backbone)
                    out_fe_cpu = out_fe.detach().cpu()
                    label_onehot = F.one_hot(targets, self._total_classes).float()
                    cov += out_fe_cpu.t() @ out_fe_cpu
                    crs_cor += out_fe_cpu.t() @ label_onehot
                    del out_backbone, out_fe, out_fe_cpu, label_onehot, inputs_gpu
                    safe_cuda_empty()

            # keep accumulators on CPU, compute inverse on CPU
            self.al_classifier.cov = (self.al_classifier.cov.cpu() + cov).cpu()
            self.al_classifier.R = (self.al_classifier.R.cpu() + cov).cpu()
            self.al_classifier.Q = (self.al_classifier.Q.cpu() + crs_cor).cpu()

            R_inv = torch.inverse(self.al_classifier.R).to(self._cpu)  # CPU
            Delta = R_inv @ self.al_classifier.Q  # CPU
            # normalize and set fc weight on GPU
            self.al_classifier.fc.weight = torch.nn.Parameter(F.normalize(Delta.t().float(), p=2, dim=-1).to(self._device))

            self._build_protos()
            safe_cuda_empty()

        else:
            if resume:
                checkpoint_path = f"{self.args['model_dir']}{self._total_classes}_model.pth.tar"
                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found!")
                print(f"Loading checkpoint: {checkpoint_path}")
                self._network.load_state_dict(torch.load(checkpoint_path, map_location=self._device)["state_dict"], strict=False)

            # network already moved to GPU by incremental_train
            if self._old_network is not None:
                # ensure old network stays on CPU (we will forward on CPU)
                try:
                    self._old_network.to(self._cpu)
                except Exception:
                    pass

            if not resume:
                optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
                self._update_representation(train_loader, test_loader, optimizer, scheduler)

            self._build_protos()

            # DPCR
            if self.args.get("DPCR", False):
                if self.debug_mode:
                    print_mem("before DPCR accumulation")
                print('Using DPCR')

                # projector trên CPU
                self.projector = Drift_Estimator(512, False, self.args).to(self._cpu)
                for _, p in self.projector.named_parameters():
                    p.requires_grad = False
                self.projector.eval()

                cov_pwdr = (self.projector.rg_tssp * torch.eye(self.projector.fe_size, device=self._cpu)).float()
                crs_cor_pwdr = torch.zeros(self.projector.fe_size, self.projector.fe_size, device=self._cpu)
                crs_cor_new = torch.zeros(self.al_classifier.fe_size, self._total_classes, device=self._cpu)
                cov_new = torch.zeros(self.projector.fe_size, self.projector.fe_size, device=self._cpu)

                with torch.inference_mode():
                    for _, (_, inputs, targets) in enumerate(tqdm(self.train_loader, desc='DPCR accumulation', unit='batch')):
                        inputs_cpu = inputs
                        inputs_gpu = inputs_cpu.to(self._device, non_blocking=True)
                        if self._amp_enabled and torch.cuda.is_available():
                            with torch.amp.autocast(device_type="cuda", dtype=self._amp_dtype):
                                feats_new_gpu = self._network(inputs_gpu)["features"]
                        else:
                            feats_new_gpu = self._network(inputs_gpu)["features"]
                        feats_new = feats_new_gpu.detach().cpu()
                        del feats_new_gpu, inputs_gpu
                        safe_cuda_empty()

                        # old net trên CPU
                        if self._old_network is not None:
                            feats_old = self._old_network(inputs_cpu)["features"].detach().cpu()
                        else:
                            feats_old = torch.zeros_like(feats_new, device=self._cpu)

                        cov_pwdr += feats_old.t() @ feats_old
                        cov_new += feats_new.t() @ feats_new
                        crs_cor_pwdr += feats_old.t() @ feats_new
                        label_onehot = F.one_hot(targets, self._total_classes).float()
                        crs_cor_new += feats_new.t() @ label_onehot

                        del feats_old, feats_new, label_onehot
                        safe_cuda_empty()

                # set projector weights (CPU)
                self.projector.cov = cov_pwdr
                self.projector.Q = crs_cor_pwdr
                R_inv = torch.inverse(cov_pwdr)
                Delta = R_inv @ crs_cor_pwdr
                self.projector.fc.weight = torch.nn.Parameter(Delta.t().float())

                # aggregate primes (CPU)
                cov_prime = torch.zeros(self.al_classifier.fe_size, self.al_classifier.fe_size, device=self._cpu)
                Q_prime = torch.zeros(self.al_classifier.fe_size, self.al_classifier.num_classes, device=self._cpu)

                for class_idx in range(0, self._known_classes):
                    if class_idx not in self._projectors or class_idx not in self._covs or class_idx not in self._protos:
                        continue

                    W = self.projector.get_weight().cpu() @ self._projectors[class_idx].cpu()
                    cov_idx = self._covs[class_idx].cpu()
                    cov_prime_idx = W.t() @ cov_idx @ W
                    label_onehot = F.one_hot(torch.tensor(class_idx, device=self._cpu), self._total_classes).float()
                    cor_prime_idx = self.num_per_class * (W.t() @ self._protos[class_idx].view(-1, 1).cpu()) @ label_onehot.view(1, self._total_classes)

                    cov_prime += cov_prime_idx
                    Q_prime += cor_prime_idx

                    # cập nhật lưu trữ trên CPU
                    self._covs[class_idx] = cov_prime_idx.cpu()
                    self._projectors[class_idx] = self.get_projector_svd(cov_prime_idx.cpu())
                    self._protos[class_idx] = (self._protos[class_idx].cpu() @ W).contiguous().cpu()

                    del W, cov_idx, cov_prime_idx, cor_prime_idx, label_onehot
                    safe_cuda_empty()

                R_prime = cov_prime + (self.al_classifier.gamma * torch.eye(self.al_classifier.fe_size, device=self._cpu))
                self.al_classifier.cov = (cov_prime + cov_new).cpu()
                self.al_classifier.Q = (Q_prime + crs_cor_new).cpu()
                self.al_classifier.R = (R_prime + cov_new).cpu()

                R_inv = torch.inverse(self.al_classifier.R).to(self._cpu)
                Delta = R_inv @ self.al_classifier.Q
                self.al_classifier.fc.weight = torch.nn.Parameter(F.normalize(Delta.t().float(), p=2, dim=-1).to(self._device))

                safe_cuda_empty()
                if self.debug_mode:
                    print_mem("after DPCR")

    def get_projector_svd(self, raw_matrix, all_non_zeros=True):
        # operate on CPU
        A = raw_matrix.cpu()
        A = 0.5 * (A + A.t())
        S, V = torch.linalg.eigh(A)
        if all_non_zeros:
            non_zeros_idx = torch.where(S > 1e-12)[0]
            if non_zeros_idx.numel() == 0:
                left_vectors = V
            else:
                left_vectors = V[:, non_zeros_idx]
        else:
            r = min(512, V.shape[1])
            left_vectors = V[:, -r:]
        projector = left_vectors @ left_vectors.t()
        return projector  # CPU

    def _build_protos(self):
        """
        Build protos/covs/projectors per class và giữ trên CPU (tiết kiệm VRAM).
        """
        bs_local = min(64, self.args.get("batch_size", batch_size))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source='train',
                mode='test',
                shot=self.shot,
                ret_data=True
            )
            idx_loader = DataLoader(idx_dataset, batch_size=bs_local, shuffle=False, num_workers=0, pin_memory=True)
            fe = self.al_classifier.fe_size
            cov_sum = torch.zeros(fe, fe, device=self._cpu)
            proto_sum = torch.zeros(fe, device=self._cpu)
            count = 0

            with torch.inference_mode():
                for _, (_, inputs, _) in enumerate(idx_loader):
                    inputs_cpu = inputs
                    inputs_gpu = inputs_cpu.to(self._device, non_blocking=True)
                    if self._amp_enabled and torch.cuda.is_available():
                        with torch.amp.autocast(device_type="cuda", dtype=self._amp_dtype):
                            feats_gpu = self._network(inputs_gpu)["features"]
                    else:
                        feats_gpu = self._network(inputs_gpu)["features"]
                    feats = feats_gpu.detach().cpu()
                    del feats_gpu, inputs_gpu
                    safe_cuda_empty()

                    cov_sum += feats.t() @ feats
                    proto_sum += feats.sum(dim=0)
                    count += feats.size(0)
                    del feats
                    safe_cuda_empty()

            if count == 0:
                continue

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
                    cov_pad = torch.zeros((fe, fe), device=self._cpu)
                    cov_pad[:cov_t.shape[0], :cov_t.shape[1]] = cov_t
                    cov_t = cov_pad

            self._protos[class_idx] = class_mean.cpu()
            self._covs[class_idx] = cov_t.cpu()
            self._projectors[class_idx] = self.get_projector_svd(cov_t.cpu())
            del cov_sum, proto_sum
            safe_cuda_empty()

        if self.debug_mode:
            print_mem("after _build_protos")

    # ---------------- epoch loops ----------------
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                if self._amp_enabled and torch.cuda.is_available():
                    with torch.amp.autocast(device_type="cuda", dtype=self._amp_dtype):
                        logits = self._network(inputs)["logits"]
                        loss = F.cross_entropy(logits, targets)
                else:
                    logits = self._network(inputs)["logits"]
                    loss = F.cross_entropy(logits, targets)

                optimizer.zero_grad(set_to_none=True)
                if self._amp_enabled:
                    loss.backward()
                else:
                    loss.backward()
                optimizer.step()
                losses += loss.item()

                with torch.no_grad():
                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    total += len(targets)

                del logits, preds, inputs, targets
                safe_cuda_empty()

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
                inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)

                if self._amp_enabled and torch.cuda.is_available():
                    with torch.amp.autocast(device_type="cuda", dtype=self._amp_dtype):
                        outputs = self._network(inputs)
                        logits = outputs["logits"]
                else:
                    outputs = self._network(inputs)
                    logits = outputs["logits"]

                fake_targets = targets - self._known_classes
                loss_clf = F.cross_entropy(logits[:, self._known_classes:], fake_targets)

                loss_kd = torch.tensor(0.0, device=self._device, dtype=torch.float32)
                loss_ot = torch.tensor(0.0, device=self._device, dtype=torch.float32)

                if self._old_network is not None:
                    # compute old_logits on CPU, rồi đưa sang GPU cho KD
                    inputs_cpu = inputs.detach().cpu()
                    with torch.inference_mode():
                        old_logits_cpu = self._old_network(inputs_cpu)["logits"].detach().cpu()
                    old_logits_gpu = old_logits_cpu.to(self._device, non_blocking=True)
                    loss_kd = _KD_loss(logits[:, :self._known_classes], old_logits_gpu, T)
                    del old_logits_cpu, old_logits_gpu
                    safe_cuda_empty()

                    # optional OT alignment (cẩn thận bộ nhớ)
                    if self.do_ot_alignment:
                        try:
                            from utils.ot_align import get_wassersteinized_layers_modularized  # nếu có
                            aligned_layers, _ = get_wassersteinized_layers_modularized(self.args, self._device,
                                                                                       [self._old_network, self._network])
                            for old_param, new_like in zip(self._old_network.parameters(), aligned_layers):
                                if old_param.shape == new_like.shape:
                                    loss_ot += F.mse_loss(old_param.to(self._device, non_blocking=True), new_like)
                            del aligned_layers
                        except Exception as e:
                            logging.warning(f"[OT] skipped due to: {e}")
                            if self.debug_mode:
                                traceback.print_exc()

                loss = lamda * loss_kd + loss_clf + 0.1 * loss_ot

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses += loss.item()

                with torch.no_grad():
                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    total += len(targets)

                del outputs, logits, preds, inputs, targets
                safe_cuda_empty()

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
