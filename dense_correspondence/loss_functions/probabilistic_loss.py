import torch
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDatasetDataType


class ProbabilisticLoss:
    def __init__(self, image_shape, config=None):
        self.type = "pixelwise_contrastive"
        self.image_width = image_shape[1]
        self.image_height = image_shape[0]
        assert config is not None
        self._config = config
        self._debug_data = dict()
        self._debug = False

    @property
    def debug(self):
        return self._debug

    @property
    def config(self):
        return self._config

    @debug.setter
    def debug(self, value):
        self._debug = value

    @property
    def debug_data(self):
        return self._debug_data

    def get_loss(self, match_type,
                 image_a_pred, image_b_pred,
                 reliability_a, reliability_b,
                 matches_a, matches_b,
                 masked_non_matches_a, masked_non_matches_b,
                 background_non_matches_a, background_non_matches_b,
                 blind_non_matches_a, blind_non_matches_b):
        matches_unnormalized_distribution = self.get_matches_distribution(image_a_pred, image_b_pred,
                                                                          reliability_a, reliability_b,
                                                                          matches_a, matches_b)
        masked_non_matches_unnormalized_distribution = self.get_non_matches_distribution(image_a_pred, image_b_pred,
                                                                                         reliability_a, reliability_b,
                                                                                         masked_non_matches_a,
                                                                                         masked_non_matches_b)
        background_non_matches_unnormalized_distribution = self.get_non_matches_distribution(image_a_pred, image_b_pred,
                                                                                             reliability_a,
                                                                                             reliability_b,
                                                                                             background_non_matches_a,
                                                                                             background_non_matches_b)
        blind_non_matche_unnormalized_distribution = self.get_non_matches_distribution(image_a_pred, image_b_pred,
                                                                                       reliability_a, reliability_b,
                                                                                       blind_non_matches_a,
                                                                                       blind_non_matches_b)
        unnormalized_distribution = torch.cat((matches_unnormalized_distribution,
                                               masked_non_matches_unnormalized_distribution,
                                               background_non_matches_unnormalized_distribution,
                                               blind_non_matche_unnormalized_distribution), dim=1)

        normalization_constant = unnormalized_distribution.sum(dim=1)
        distribution = unnormalized_distribution / normalization_constant
        loss = (-torch.log(distribution)).mean(dim=1).mean()
        return loss

    def get_matches_distribution(self, image_a_pred, image_b_pred,
                                 reliability_a, reliability_b,
                                 matches_a, matches_b):
        matches_a_descriptors = torch.index_select(image_a_pred, 1, matches_a)
        matches_b_descriptors = torch.index_select(image_b_pred, 1, matches_b)
        matches_a_reliability = torch.index_select(reliability_a, 1, matches_a)
        matches_b_reliability = torch.index_select(reliability_b, 1, matches_b)

        if len(matches_a) == 1:
            print("unsqueeze")
            matches_a_descriptors = matches_a_descriptors.unsqueeze(0)
            matches_b_descriptors = matches_b_descriptors.unsqueeze(0)
            assert False

        matching_score = (matches_a_descriptors * matches_b_descriptors).sum(dim=2).clamp(min=0)
        average_reliability = (matches_a_reliability + matches_b_reliability) / 2
        unnormalized_distribution = torch.exp(matching_score / average_reliability)
        return unnormalized_distribution

    def get_non_matches_distribution(self, image_a_pred, image_b_pred,
                                  reliability_a, reliability_b,
                                  non_matches_a, non_matches_b):
        non_matches_a_descriptors = torch.index_select(image_a_pred, 1, non_matches_a)
        non_matches_b_descriptors = torch.index_select(image_b_pred, 1, non_matches_b)
        non_matches_a_reliability = torch.index_select(reliability_a, 1, non_matches_a)
        non_matches_b_reliability = torch.index_select(reliability_b, 1, non_matches_b)

        matching_score = (non_matches_a_descriptors * non_matches_b_descriptors).sum(dim=2).clamp(min=0)
        average_reliability = (non_matches_a_reliability + non_matches_b_reliability) / 2
        unnormalized_distribution = torch.exp((1 - matching_score) / average_reliability)
        return unnormalized_distribution