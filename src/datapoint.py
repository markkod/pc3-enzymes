class DataPoint:
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