import numpy as np

class PSignature:
    def __init__(self, bins, assigned_points=[]):
        self.bins = bins
        self.bin_dict = {}
        for _bin in bins:
            self.bin_dict[_bin.id] = _bin
        self.assigned_points = assigned_points
        self.id = list([_bin.id for _bin in bins])
        self.parent = False
        
    def get_support(self):
        return len(self.assigned_points)
    
    def add_bin(self, _bin):
        self.bins.append(_bin)
        self.assigned_points.append(_bin.assigned_support)
        self.id.append(_bin.index)
        self.bin_dict[_bin.id] = _bin
    
    def reevaluate_assigned_points(self, _bin, current_dim):
        evaluated_points = []
        current_interval = _bin.interval
        for point in self.assigned_points:
            if point[current_dim] > current_interval.start and point[current_dim] <= current_interval.end:
                evaluated_points.append(point)
        return evaluated_points
    
    def get_means(self):
        return np.average(np.array(self.assigned_points), axis = 0)
    
    def __str__(self):
        return f"NumBins: {len(self.bins)}, NumPoints: {len(self.assigned_points)}, Parent: {self.parent}"
    
    def __repr__(self):
        return self.__str__()