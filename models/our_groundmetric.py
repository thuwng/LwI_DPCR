import torch

def isnan(x):
    return x != x

class GroundMetric:  

    def __init__(self, params, not_squared = False):
        self.params = params
        self.ground_metric_type = params.ground_metric
        self.ground_metric_normalize = params.ground_metric_normalize
        self.reg = params.reg   # what is reg?
        if hasattr(params, 'not_squared'):
            self.squared = not params.not_squared
        else:
            # so by default squared will be on!
            self.squared = not not_squared
        self.mem_eff = params.ground_metric_eff     # what is mem_eff?

    def _clip(self, ground_metric_matrix):

        if self.params.debug:
            print("before clipping", ground_metric_matrix.data)

        percent_clipped = (float((ground_metric_matrix >= self.reg * self.params.clip_max).long().sum().data) \
                           / ground_metric_matrix.numel()) * 100    # what is numel()?
        print("percent_clipped is (assumes clip_min = 0) ", percent_clipped)
        setattr(self.params, 'percent_clipped', percent_clipped)
        # will keep the M' = M/reg in range clip_min and clip_max
        ground_metric_matrix.clamp_(min=self.reg * self.params.clip_min,
                                             max=self.reg * self.params.clip_max)
        if self.params.debug:
            print("after clipping", ground_metric_matrix.data)
        return ground_metric_matrix

    def _normalize(self, ground_metric_matrix):

        if self.ground_metric_normalize == "log":
            ground_metric_matrix = torch.log1p(ground_metric_matrix)
        elif self.ground_metric_normalize == "max":
            print("Normalizing by max of ground metric and which is ", ground_metric_matrix.max())
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.max()
        elif self.ground_metric_normalize == "median":
            print("Normalizing by median of ground metric and which is ", ground_metric_matrix.median())
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.median()
        elif self.ground_metric_normalize == "mean":
            print("Normalizing by mean of ground metric and which is ", ground_metric_matrix.mean())
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.mean()
        elif self.ground_metric_normalize == "none":
            return ground_metric_matrix
        else:
            raise NotImplementedError

        return ground_metric_matrix

    def _sanity_check(self, ground_metric_matrix):
        # print('ground',ground_metric_matrix)
        assert not (ground_metric_matrix < 0).any()
        assert not (isnan(ground_metric_matrix).any())

    def _cost_matrix_xy(self, x, y, p=2, squared = True):
        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        c = torch.sum(((torch.abs(x_col - y_lin))) ** p, 2)
        # print('c',c)
        if not squared:
            c = c ** (1/2)
        if self.params.dist_normalize:
            assert NotImplementedError
        return c


    def _pairwise_distances(self, x, y=None, squared=True):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = (x_norm + y_norm - 2.0 * torch.mm(x, y_t))
        dist = torch.clamp(dist, min=0.0)
        dist = dist/self.params.act_num_samples

        if not squared:
            dist = dist ** (1/2)

        return dist

    def _get_euclidean(self, coordinates, other_coordinates=None):
        if other_coordinates is None:
            matrix = torch.norm(
                coordinates.view(coordinates.shape[0], 1, coordinates.shape[1]) \
                - coordinates, p=2, dim=2
            )
        else:
            if self.mem_eff:
                matrix = self._pairwise_distances(coordinates, other_coordinates, squared=self.squared)
            else:
                matrix = self._cost_matrix_xy(coordinates, other_coordinates, squared = self.squared)

        return matrix

    def _normed_vecs(self, vecs, eps=1e-9):
        norms = torch.norm(vecs, dim=-1, keepdim=True)
        return vecs / (norms + eps)

    def _get_cosine(self, coordinates, other_coordinates=None):
        if other_coordinates is None:
            matrix = coordinates / torch.norm(coordinates, dim=1, keepdim=True)
            matrix = 1 - matrix @ matrix.t()    # "@" symbol means matrix multiplication
        else:
            matrix = 1 - torch.div(
                coordinates @ other_coordinates.t(),
                torch.norm(coordinates, dim=1).view(-1, 1) @ torch.norm(other_coordinates, dim=1).view(1, -1)
            )
        return matrix.clamp_(min=0)

    def _get_angular(self, coordinates, other_coordinates=None):
        pass

    def get_metric(self, coordinates, other_coordinates=None):
        get_metric_map = {
            'euclidean': self._get_euclidean,
            'cosine': self._get_cosine,
            'angular': self._get_angular,
        }
        return get_metric_map[self.ground_metric_type](coordinates, other_coordinates)

    def process(self, coordinates, other_coordinates=None):
        if self.params.geom_ensemble_type == 'wts' and self.params.normalize_wts:
            coordinates = self._normed_vecs(coordinates)
            if other_coordinates is not None:
                other_coordinates = self._normed_vecs(other_coordinates)

        ground_metric_matrix = self.get_metric(coordinates, other_coordinates)
        # print('ground',ground_metric_matrix)
        if self.params.debug:
            if other_coordinates is not None:
                print("other_coordinates is ", other_coordinates)
            print("ground_metric_matrix is ", ground_metric_matrix)

        self._sanity_check(ground_metric_matrix)

        ground_metric_matrix = self._normalize(ground_metric_matrix)
        # print('ground1', ground_metric_matrix)
        self._sanity_check(ground_metric_matrix)

        if self.params.clip_gm:
            ground_metric_matrix = self._clip(ground_metric_matrix)

        self._sanity_check(ground_metric_matrix)

        if self.params.debug:
            print("ground_metric_matrix at the end is ", ground_metric_matrix)

        return ground_metric_matrix