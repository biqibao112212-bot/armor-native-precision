"""Training group selection with explicit distance probability.

Balancing every (distance, view, light, crop-size) group directly overweights
ranges with more distinct crop sizes. Distance balancing first chooses a range,
then a group inside it. No held-out rows or per-frame truth enter inference.
"""
from collections import defaultdict

class GroupSelector:
    def __init__(self, keys, mode='groups'):
        if mode not in ('groups', 'distance'):
            raise ValueError(mode)
        self.keys = sorted(keys)
        if not self.keys:
            raise ValueError('empty training groups')
        self.mode = mode
        grouped = defaultdict(list)
        for key in self.keys:
            grouped[key[0]].append(key)
        self.by_range = dict(grouped)
        self.ranges = sorted(grouped)

    def draw(self, rng):
        keys = self.keys
        if self.mode == 'distance':
            band = self.ranges[int(rng.integers(len(self.ranges)))]
            keys = self.by_range[band]
        return keys[int(rng.integers(len(keys)))]

    def range_probabilities(self):
        return {str(k): (1 / len(self.ranges) if self.mode == 'distance'
                         else len(v) / len(self.keys))
                for k, v in self.by_range.items()}
