import numpy as np


class DistanceService:
    def get_smallest_distance_index(self, arr):
        return np.argmin(arr)

    def check_if_distance_is_small(self, distance):
        return bool(distance < 1)

    def arr_consists_of_ones(self, arr):
        filter_arr = arr <= 1
        return arr[filter_arr].size == 0