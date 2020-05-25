import numpy as np
import matplotlib.pyplot as plt


class ReliabilityStatistics:
    def __init__(self, name_prefix, create_histogram=False):
        self.name_prefix = name_prefix
        self.create_histogram = create_histogram
        self.reliability_maps = []

    def add_from_matches(self, reliability, matches):
        reliability = reliability.squeeze()
        reliability = reliability[matches]
        reliability = reliability.data.cpu().numpy()
        self.reliability_maps.append(reliability)

    def add_from_mask(self, reliability, mask):
        reliability = reliability.squeeze()
        reliability = reliability[mask.nonzero()]
        self.reliability_maps.append(reliability)

    def log(self, logger, x):
        values = np.concatenate(self.reliability_maps)
        stats = self._compute_stats(values)
        for stat_name, value in stats.items():
            name = '{}_{}'.format(self.name_prefix, stat_name)
            logger.log(name, x, value)

        if self.create_histogram:
            figure = self._compute_histogram(values, x)
            name = '{}_histogram'.format(self.name_prefix)
            logger.log(name, x, figure, type='image')
            plt.close(figure)

    @staticmethod
    def _compute_stats(values):
        stats = dict()
        stats['min'] = values.min()
        stats['max'] = values.max()
        stats['mean'] = values.mean()
        stats['std'] = values.std()
        return stats

    @staticmethod
    def _compute_histogram(values, x):
        figure, ax = plt.subplots()
        ax.hist(values, bins=100, range=(0, 1))
        ax.set_title(str(x))
        return figure
