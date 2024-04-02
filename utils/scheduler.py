from abc import abstractmethod, ABC
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple

class HparamScheduler(ABC):
    @abstractmethod
    def step(self) -> float:
        pass

    @abstractmethod
    def get_value(self, round: int) -> float:
        pass

class SineClimbing(HparamScheduler):
    def __init__(self,
                 min_value: float,
                 max_value: float,
                 num_round: int,
                 warm_up_round: int=0
                ):
        super(SineClimbing, self).__init__()
        self._cur_round = 0
        self.min_value, self.max_value = min_value, max_value
        self.total_round = num_round
        self.warm_up_round = warm_up_round
    
    def step(self) -> float:
        value = self.get_value(self._cur_round)
        self._cur_round += 1
        return value

    def get_value(self, round: int) -> float:
        '''
        Annealing from min value to max value according to sine 
        function

        set value as min_value during warm up round
        '''
        if round < self.warm_up_round:
            penalty = 0
        else:
            base = np.pi / 2
            anneal_round = round - self.warm_up_round
            penalty = np.sin(base * (anneal_round / self.total_round))
        cur_value = self.min_value + penalty * (self.max_value - self.min_value)
        return cur_value



class Constant(HparamScheduler):
    def __init__(self, value):
        super(Constant, self).__init__()
        self.value = value
        
    def step(self) -> float:
        return self.value
    def get_value(self, round: int) -> float:
        return self.value

class Heaviside(HparamScheduler):
    def __init__(self,
                 value: float, 
                 turning_round: int,
                 floor: float=.0):
        super(Heaviside, self).__init__()
        self.value = value
        self.turning_round = turning_round
        self.floor = floor

        self._cur_round = 0
    def step(self):
        value = self.get_value(self._cur_round)
        self._cur_round += 1
        return value
    def get_value(self, round: int) -> float:
        if round < self.turning_round:
            return self.floor
        else:
            return self.value

        
        
class ChainedScheduler(HparamScheduler):
    def __init__(self,
                 chains: List[HparamScheduler],
                 milestones: List[int]
                ):
        super(ChainedScheduler, self).__init__()
        self.chains = chains
        self.milestones = milestones
        self._cur_round = 0

        assert len(chains) == len(milestones) + 1
    
    def step(self) -> float:
        value = self.get_value(self._cur_round)
        self._cur_round += 1
        return value

    def get_value(self, round: int) -> float:
        sche_idx, lvalue, _ = ChainedScheduler._switch(self.milestones, round)
        climbing_round = round - lvalue
        return self.chains[sche_idx].get_value(climbing_round)

    @staticmethod
    def _switch(intervals: List[int],
                value: int) -> Tuple[int, int, int]:
        '''
        determine which interval given value falls into
        by default, 0 is excluded from intervals input
        for example,
        intervals = [3, 4], denotes two intervals: [0, 3) & [3, 4]
        - return tuple
          (interval_idx, lvalue, rvalue)
          where interval_idx counts from 0
        '''
        _interval = [0] + intervals + [1 << 30]
        interval_tuples = []
        for i in range(len(_interval)-1):
            lvalue, rvalue = _interval[i:i+2]
            interval_tuples.append((lvalue, rvalue))

        
        for idx, (lvalue, rvalue) in enumerate(interval_tuples):
            if value >= lvalue and value < rvalue:
                return idx, lvalue, rvalue
        
