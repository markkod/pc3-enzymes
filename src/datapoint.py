class DataPoint:
    # Class that represents a point in the dataset
    # It keeps track of its coordinates, as well as of cluster to 
    # which it is assigned.
    def __init__(self, coords):
        self.coords = coords
        self.assigned_clusters = []
        
    def __eq__(self, other):
        if len(other.coords) != len(self.coords):
            return False
        for i, x in enumerate(self.coords):
            if x - other.coords[i] > 1e-9:
                return False
        return True