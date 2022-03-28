import numpy as np


class DistanceService:
    def get_smallest_distance_index(self, arr):
        return np.argmin(arr)

    def check_if_distance_is_small(self, distance):
        return distance < 1