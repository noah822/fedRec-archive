from enum import Enum

class STATE(Enum):
    AUDIO = 0
    IMAGE = 1
    BOTH = 2

class LossMode(Enum):
    SelfContra = 0
    CrossContra = 1
    HybridContra = 2