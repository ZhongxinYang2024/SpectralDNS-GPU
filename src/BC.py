class BoundaryCondition:
    def __init__(self, type='None'):
        self.type = type
    
    def apply(self, state):
        if self.type == 'None':
            pass