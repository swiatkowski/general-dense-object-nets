import torch.nn.functional as F


class RepeatabilityLoss:
    @staticmethod
    def get_loss(repeatability_a, repeatability_b, dataset_item):
        repeatability_a = repeatability_a[:, dataset_item.matches_a]
        repeatability_b = repeatability_b[:, dataset_item.matches_b]
        repeatability_a = repeatability_a.squeeze()
        repeatability_b = repeatability_b.squeeze()
        cosine_similarity = F.cosine_similarity(repeatability_a, repeatability_b, dim=0)
        cosine_loss = 1 - cosine_similarity
        peaky_loss_a = RepeatabilityLoss.peaky_loss(repeatability_a)
        peaky_loss_b = RepeatabilityLoss.peaky_loss(repeatability_b)
        peaky_loss = (peaky_loss_a + peaky_loss_b) / 2
        loss = cosine_loss + peaky_loss
        return loss, cosine_loss, peaky_loss

    @staticmethod
    def peaky_loss(repeatability):
        return 1 - repeatability.max() + repeatability.mean()
