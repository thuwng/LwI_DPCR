import torch

def isnan(x):
    return x != x

class GroundMetric:
    def __init__(self, params, not_squared=False):
        self.params = params
        # Giá trị mặc định nếu không có trong params
        self.ground_metric_type = params.get('ground_metric', 'cosine')  # Mặc định là cosine
        self.ground_metric_normalize = params.get('ground_metric_normalize', 'none')  # Mặc định không normalize
        self.reg = params.get('reg', 1.0)  # Hệ số regularization mặc định
        self.mem_eff = params.get('ground_metric_eff', False)  # Mặc định không dùng memory efficiency
        if hasattr(params, 'not_squared'):
            self.squared = not params.not_squared
        else:
            self.squared = not not_squared  # Mặc định squared = True
        self.device = params.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.debug = params.get('debug', False)  # Mặc định không debug
        self.clip_max = params.get('clip_max', 100.0)  # Giá trị tối đa mặc định
        self.clip_min = params.get('clip_min', 0.0)  # Giá trị tối thiểu mặc định
        self.dist_normalize = params.get('dist_normalize', False)  # Mặc định không normalize distance
        self.geom_ensemble_type = params.get('geom_ensemble_type', 'none')  # Mặc định không ensemble
        self.normalize_wts = params.get('normalize_wts', False)  # Mặc định không normalize weights
        self.act_num_samples = params.get('act_num_samples', 1.0)  # Mặc định 1.0
        self.clip_gm = params.get('clip_gm', True)  # Mặc định áp dụng clipping

    def _clip(self, ground_metric_matrix):
        if self.debug:
            print("before clipping", ground_metric_matrix.data)
        percent_clipped = (float((ground_metric_matrix >= self.reg * self.clip_max).long().sum().item()) / ground_metric_matrix.numel()) * 100
        print("percent_clipped is (assumes clip_min = 0): ", percent_clipped)
        setattr(self.params, 'percent_clipped', percent_clipped)
        ground_metric_matrix.clamp_(min=self.reg * self.clip_min, max=self.reg * self.clip_max)
        if self.debug:
            print("after clipping", ground_metric_matrix.data)
        return ground_metric_matrix

    def _normalize(self, ground_metric_matrix):
        if self.ground_metric_normalize == "log":
            ground_metric_matrix = torch.log1p(ground_metric_matrix)
        elif self.ground_metric_normalize == "max":
            max_val = ground_metric_matrix.max()
            print("Normalizing by max of ground metric, which is ", max_val)
            ground_metric_matrix = ground_metric_matrix / (max_val + 1e-8)  # Tránh chia cho 0
        elif self.ground_metric_normalize == "median":
            median_val = ground_metric_matrix.median()
            print("Normalizing by median of ground metric, which is ", median_val)
            ground_metric_matrix = ground_metric_matrix / (median_val + 1e-8)
        elif self.ground_metric_normalize == "mean":
            mean_val = ground_metric_matrix.mean()
            print("Normalizing by mean of ground metric, which is ", mean_val)
            ground_metric_matrix = ground_metric_matrix / (mean_val + 1e-8)
        elif self.ground_metric_normalize == "none":
            return ground_metric_matrix
        else:
            raise NotImplementedError(f"Normalize type '{self.ground_metric_normalize}' not implemented.")
        return ground_metric_matrix

    def _sanity_check(self, ground_metric_matrix):
        assert not (ground_metric_matrix < 0).any(), "Ground metric contains negative values."
        assert not isnan(ground_metric_matrix).any(), "Ground metric contains NaN values."

    def _cost_matrix_xy(self, x, y, p=2, squared=True):
        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        c = torch.sum((torch.abs(x_col - y_lin)) ** p, dim=2)
        if not squared:
            c = c ** (1.0 / p)
        return c

    def _pairwise_distances(self, x, y=None, squared=True):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y = x
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        dist = torch.clamp(dist, min=0.0)
        dist = dist / self.act_num_samples

        if not squared:
            dist = dist ** 0.5
        return dist

    def _get_euclidean(self, coordinates, other_coordinates=None):
        if other_coordinates is None:
            if self.mem_eff:
                matrix = self._pairwise_distances(coordinates, squared=self.squared)
            else:
                matrix = self._cost_matrix_xy(coordinates, coordinates, squared=self.squared)
        else:
            if self.mem_eff:
                matrix = self._pairwise_distances(coordinates, other_coordinates, squared=self.squared)
            else:
                matrix = self._cost_matrix_xy(coordinates, other_coordinates, squared=self.squared)
        return matrix

    def _normed_vecs(self, vecs, eps=1e-9):
        norms = torch.norm(vecs, dim=-1, keepdim=True)
        return vecs / (norms + eps)

    def _get_cosine(self, coordinates, other_coordinates=None):
        if other_coordinates is None:
            normed_coords = self._normed_vecs(coordinates)
            matrix = 1 - torch.matmul(normed_coords, normed_coords.t())
        else:
            normed_x = self._normed_vecs(coordinates)
            normed_y = self._normed_vecs(other_coordinates)
            matrix = 1 - torch.matmul(normed_x, normed_y.t())
        return matrix.clamp_(min=0)

    def _get_angular(self, coordinates, other_coordinates=None):
        # Triển khai cơ bản cho angular (dựa trên cosine với điều chỉnh góc)
        if other_coordinates is None:
            normed_coords = self._normed_vecs(coordinates)
            matrix = torch.acos(torch.clamp(torch.matmul(normed_coords, normed_coords.t()), -1.0, 1.0))
        else:
            normed_x = self._normed_vecs(coordinates)
            normed_y = self._normed_vecs(other_coordinates)
            matrix = torch.acos(torch.clamp(torch.matmul(normed_x, normed_y.t()), -1.0, 1.0))
        return matrix

    def get_metric(self, coordinates, other_coordinates=None):
        get_metric_map = {
            'euclidean': self._get_euclidean,
            'cosine': self._get_cosine,
            'angular': self._get_angular,
        }
        if self.ground_metric_type not in get_metric_map:
            raise ValueError(f"ground_metric_type '{self.ground_metric_type}' not supported.")
        return get_metric_map[self.ground_metric_type](coordinates, other_coordinates)

    def process(self, coordinates, other_coordinates=None):
        if self.geom_ensemble_type == 'wts' and self.normalize_wts:
            coordinates = self._normed_vecs(coordinates)
            if other_coordinates is not None:
                other_coordinates = self._normed_vecs(other_coordinates)

        ground_metric_matrix = self.get_metric(coordinates, other_coordinates)
        if self.debug:
            if other_coordinates is not None:
                print("other_coordinates is ", other_coordinates)
            print("ground_metric_matrix is ", ground_metric_matrix)

        self._sanity_check(ground_metric_matrix)
        ground_metric_matrix = self._normalize(ground_metric_matrix)
        self._sanity_check(ground_metric_matrix)

        if self.clip_gm:
            ground_metric_matrix = self._clip(ground_metric_matrix)
        self._sanity_check(ground_metric_matrix)

        if self.debug:
            print("ground_metric_matrix at the end is ", ground_metric_matrix)
        return ground_metric_matrix