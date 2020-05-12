import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple


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
    def __init__(self, nq=20, sampler=None, num_samples=80, ap_threshold=0.5):
        assert 0 <= ap_threshold <= 1
        nn.Module.__init__(self)
        self.aploss = APLoss(nq, min=0, max=1)
        self.sampler = sampler
        self.num_samples = num_samples
        self._ap_threshold = ap_threshold

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

        l2_distance = torch.norm(selected_descriptors_1 - selected_descriptors_2, p=2, dim=-1)
        return 1 - l2_distance / 2 # truncate to [0, 1] space

    def combine_scores(self, positive_scores, negative_scores):
        scores = torch.cat((positive_scores, negative_scores), dim=-1)
        ground_truth = scores.new_zeros(scores.shape, dtype=torch.uint8)
        ground_truth[:, :, :positive_scores.shape[2]] = 1
        return scores, ground_truth

    def forward(self, descriptors1, descriptors2, dataset_item):
        matches_1, matches_2 = dataset_item.matches_a, dataset_item.matches_b
        non_matches_2 = self.sampler.get_samples(self.num_samples, dataset_item)

        matches_1 = matches_1.unsqueeze(-1)
        matches_2 = matches_2.unsqueeze(-1)

        positive_scores = self.compute_scores(descriptors1, descriptors2, matches_1, matches_2)
        negative_scores = self.compute_scores(descriptors1, descriptors2, matches_1, non_matches_2)
        scores, ground_truth = self.combine_scores(positive_scores, negative_scores)

        # compute pixel-wise AP
        ap_score = self.aploss(scores, ground_truth)
        return ap_score

    def get_loss(self, descriptors1, descriptors2, dataset_item):
        ap_score = self(descriptors1, descriptors2, dataset_item)
        ap_loss = (1 - ap_score).mean()
        return ap_loss

    def get_loss_with_reliability(
            self, descriptors1, descriptors2, dataset_item, reliability1, reliability2):
        ap_score = self(descriptors1, descriptors2, dataset_item)
        ap_loss = (1 - ap_score).mean()
        reliability1 = reliability1[:, dataset_item.matches_a]
        reliability2 = reliability2[:, dataset_item.matches_b]
        average_reliability = (reliability1 + reliability2) / 2
        ap_loss_with_reliability = (1 - ap_score * average_reliability
                                    - self._ap_threshold * (1 - average_reliability)).mean()
        return ap_loss, ap_loss_with_reliability