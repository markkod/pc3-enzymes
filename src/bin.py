class Bin:
    # A class representing a bin
    # A bin is a segment of the dataset in one dimension
    # In the first steps of the algorithm the bins are distributed uniformly
    # and later they get merged according to a statistical test
    def __init__(self, index, interval, dim):
        self.interval = interval
        self.marked = False
        self.support = 0
        self.index = index
        self.merge_with = []
        self.assigned_points = []
        self.dimension = dim
        self.id = 'd{}b{}'.format(dim, index)
        
    def add_point(self, point):
        """Adds a point to the bin. Keeps track of the support set
        size
        
        Arguments:
        point -- new point that is added to the bin
        """
        
        self.support += 1
        self.assigned_points.append(point)
        
    def get_width(self):
        """Returns the width of the bin
        
        Returns: numeric value representing tte width of the bin in space
        """

        return self.interval.length
    
    def __str__(self):
        return 'Interval: {0}; Marked: {1}; Support: {2}; Index: {3}; \
            Merge-with {4}; # of assigned points {5}; Dimension: {6}'.format(
            self.interval,
            self.marked,
            self.support,
            self.index,
            self.merge_with,
            len(self.assigned_points),
            self.dimension
        )
    
    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self.id == other.id