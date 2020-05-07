from collections import namedtuple


class ReliabilityStatistics:
    Statistics = namedtuple('Statistics', ['min', 'max', 'mean', 'std'])

    def __init__(self, name_prefix, create_histograms=False):
        self.name_prefix = name_prefix
        self.create_histograms = create_histograms
        self.stats = []
        self.histograms = []

    def add_from_matches(self, reliability, matches):
        reliability = reliability[:, matches]
        min = reliability.min().item()
        max = reliability.max().item()
        mean = reliability.mean().item()
        std = reliability.std().item()
        self.stats.append(self.Statistics(min, max, mean, std))

    def add_from_mask(self, reliability, mask):
        reliability = reliability[mask.nonzero()]
        min = reliability.min()
        max = reliability.max()
        mean = reliability.mean()
        std = reliability.std()
        self.stats.append(self.Statistics(min, max, mean, std))

    def log(self, logger, x):
        for suffix, stats in enumerate(self.stats, 1):
            for stat_name, value in stats._asdict().items():
                name = '{}_{}_{}'.format(self.name_prefix, stat_name, suffix)
                logger.log(name, x, value)