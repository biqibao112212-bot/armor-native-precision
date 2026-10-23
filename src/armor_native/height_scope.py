"""Scope and sampling for the user's camera-to-armor vertical height envelope."""
from collections import defaultdict
HEIGHT_LIMIT_M = .4
TYPICAL_HEIGHT_M = .15
HEIGHT_WEIGHTS = {0: .70, 1: .15, 2: .15}
HEIGHT_NAMES = {0: "within_15cm", 1: "camera_15_to_40cm_above", 2: "camera_15_to_40cm_below"}

def height_region(height):
    h = float(height)
    if abs(h) > HEIGHT_LIMIT_M + 1e-9:
        raise ValueError("object outside signed camera-minus-armor height scope")
    return 0 if abs(h) <= TYPICAL_HEIGHT_M else (1 if h > 0 else 2)

def height_pool(rows):
    result = {}
    for row in rows:
        for obj in row["objects"]:
            key = (obj["range_bin"], min(2, int(max(0, obj["facing_cos"]) * 3)),
                   row["light"], obj["native_size"], height_region(obj["height_difference_m"]))
            result.setdefault(key, {}).setdefault(row["session"], []).append((row, obj))
    return result

class DistanceHeightSelector:
    """Uniform distances, 70% typical height and 15% per height extreme sign.

    Missing height strata are explicitly renormalized within that distance.
    View/light/size groups and sessions are balanced by the caller, as before.
    """
    def __init__(self, keys):
        self.by_range = defaultdict(lambda: defaultdict(list))
        for key in sorted(keys):
            if len(key) != 5 or key[-1] not in HEIGHT_WEIGHTS:
                raise ValueError("expected five-part range/view/light/size/height key")
            self.by_range[key[0]][key[-1]].append(key)
        self.ranges = sorted(self.by_range)
        if not self.ranges:
            raise ValueError("empty training groups")

    def probabilities(self):
        result = {}
        for band in self.ranges:
            regions = sorted(self.by_range[band])
            total = sum(HEIGHT_WEIGHTS[b] for b in regions)
            result[str(band)] = {str(b): HEIGHT_WEIGHTS[b]/total for b in regions}
        return result

    def range_probabilities(self):
        return {str(b): 1/len(self.ranges) for b in self.ranges}

    def draw(self, rng):
        band = self.ranges[int(rng.integers(len(self.ranges)))]
        regions = sorted(self.by_range[band])
        total = sum(HEIGHT_WEIGHTS[b] for b in regions)
        region = regions[int(rng.choice(len(regions), p=[HEIGHT_WEIGHTS[b]/total for b in regions]))]
        keys = self.by_range[band][region]
        return keys[int(rng.integers(len(keys)))]
