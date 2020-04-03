import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# this class is taken from https://github.com/naver/r2d2/blob/master/nets/ap_loss.py
class APLoss (nn.Module):
    """ differentiable AP loss, through quantization.

        Input: (N, M)   values in [min, max]
        label: (N, M)   values in {0, 1}

        Returns: list of query AP (for each n in {1..N})
                 Note: typically, you want to minimize 1 - mean(AP)
    """
    def __init__(self, nq=25, min=0, max=1):
        nn.Module.__init__(self)
        assert isinstance(nq, int) and 2 <= nq <= 100
        self.nq = nq
        self.min = min
        self.max = max
        gap = max - min
        assert gap > 0

        # init quantizer = non-learnable (fixed) convolution
        self.quantizer = q = nn.Conv1d(1, 2*nq, kernel_size=1, bias=True)
        a = (nq-1) / gap
        #1st half = lines passing to (min+x,1) and (min+x+1/a,0) with x = {nq-1..0}*gap/(nq-1)
        q.weight.data[:nq] = -a
        q.bias.data[:nq] = torch.from_numpy(a*min + np.arange(nq, 0, -1)) # b = 1 + a*(min+x)
        #2nd half = lines passing to (min+x,1) and (min+x-1/a,0) with x = {nq-1..0}*gap/(nq-1)
        q.weight.data[nq:] = a
        q.bias.data[nq:] = torch.from_numpy(np.arange(2-nq, 2, 1) - a*min) # b = 1 - a*(min+x)
        # first and last one are special: just horizontal straight line
        q.weight.data[0] = q.weight.data[-1] = 0
        q.bias.data[0] = q.bias.data[-1] = 1

    def compute_AP(self, x, label):
        BS, N, M = x.shape

        # quantize all predictions
        # TODO: make to work over batch. Currently only BS=1
        q = self.quantizer(x.squeeze(0).unsqueeze(1))
        q = torch.min(q[:,:self.nq], q[:,self.nq:]).clamp(min=0) # N x Q x M

        nbs = q.sum(dim=-1) # number of samples  N x Q = c
        rec = (q * label.view(N,1,M).float()).sum(dim=-1) # nb of correct samples = c+ N x Q
        prec = rec.cumsum(dim=-1) / (1e-16 + nbs.cumsum(dim=-1)) # precision
        rec /= rec.sum(dim=-1).unsqueeze(1) # norm in [0,1]

        ap = (prec * rec).sum(dim=-1) # per-image AP
        return ap

    def forward(self, x, label):
        assert x.shape == label.shape # N x M
        return self.compute_AP(x, label)

# this class is inspired by PixelAPLoss from https://github.com/naver/r2d2/blob/master/nets/ap_loss.py
class PixelAPLoss(nn.Module):
    """
    Computes the pixel-wise AP loss
    """
    def __init__(self, nq=20, sampler=None, num_negative_samples=80):
        nn.Module.__init__(self)
        self.aploss = APLoss(nq, min=0, max=1)
        self.sampler = sampler
        self.num_negative_samples = num_negative_samples

    def compute_scores(self, descriptors1, descriptors2, indices_1, indices_2):
        selected_descriptors_1 = descriptors1[:, indices_1, :]
        selected_descriptors_2 = descriptors2[:, indices_2, :]

        # crazily enough, if there is only one element to index_select into
        # above, then the first dimension is collapsed down, and we end up
        # with shape [D,], where we want [1,D]
        # this unsqueeze fixes that case
        if len(indices_1) == 1:
            selected_descriptors_1 = selected_descriptors_1.unsqueeze(0)
            selected_descriptors_2 = selected_descriptors_2.unsqueeze(0)

        cosine_distance = (selected_descriptors_1 * selected_descriptors_2).sum(-1)
        return cosine_distance

    def combine_scores(self, positive_scores, negative_scores):
        scores = torch.cat((positive_scores, negative_scores), dim=-1)
        ground_truth = scores.new_zeros(scores.shape, dtype=torch.uint8)
        ground_truth[:, :, :positive_scores.shape[2]] = 1
        return scores, ground_truth


    def get_indieces_from_points_and_offsets(self, matches, offsets):
        offsetted_points = matches[:,None] + offsets
        return offsetted_points.clamp(0, 480 * 640 - 1)

    def forward(self, descriptors1, descriptors2, matches_1, matches_2):
        non_matches_2 = self.get_indieces_from_points_and_offsets(matches_2, self.sampler.get_samples(self.num_negative_samples))
        matches_1 = matches_1.unsqueeze(-1)
        matches_2 = matches_2.unsqueeze(-1)

        positive_scores = self.compute_scores(descriptors1, descriptors2, matches_1, matches_2)
        negative_scores = self.compute_scores(descriptors1, descriptors2, matches_1, non_matches_2)
        scores, ground_truth = self.combine_scores(positive_scores, negative_scores)

        # compute pixel-wise AP
        ap_score = self.aploss(scores, ground_truth)

        # [WIP]
        # this line shuld be changed if you want get more funky with ap loss
        # for instance if you want to add reliabiliy map do sth like:
        # 1 - ap_score * rel
        ap_loss = 1 - ap_score
        return ap_loss.mean()


# this class is inspired by Ngh2Sampler from https://github.com/naver/r2d2/blob/master/nets/sampler.py
class RingSampler(nn.Module):
    """
    Class for sampling non-correspondence.
    Points are being drawn from the ring around true match
    Radius is defined in pixel units.
    """
    def __init__(self, inner_radius=10, outter_radius=12):
        nn.Module.__init__(self)
        self.inner_radius = inner_radius
        self.outter_radius = outter_radius
        self.sample_offsets()

    def sample_offsets(self, image_width=640):
        inner_r2 = self.inner_radius**2
        outer_r2 = self.outter_radius**2
        neg = []
        for j in range(-self.outter_radius-1, self.outter_radius+1):
            for i in range(-self.outter_radius-1, self.outter_radius+1):
                d2 = i*i + j*j
                if inner_r2 <= d2 <= outer_r2:
                    neg.append(i * image_width + j)

        self.register_buffer('negative_offsets', torch.LongTensor(neg))

    def get_samples(self, num_samples=None):
        if num_samples is None:
            return self.negative_offsets
        if num_samples < 0:
            raise Exception('Number of samples must be positive')
        num_offsets = len(self.negative_offsets)
        indices = torch.randperm(num_offsets)[:num_samples]
        return self.negative_offsets[indices]
