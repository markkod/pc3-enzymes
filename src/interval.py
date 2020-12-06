class Interval:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.length = end - start
    
    def __str__(self):
        return 'Start: {0}; End: {1}; Length {2}'.format(self.start, self.end, self.length)