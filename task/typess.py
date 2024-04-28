import enum



    
    
class LossType(enum.Enum):
    L1 = enum.auto()
    L2 = enum.auto()
    SmoothL1 = enum.auto()
    CrossEntropy = enum.auto()
