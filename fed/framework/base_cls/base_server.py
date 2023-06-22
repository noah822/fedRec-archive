from abc import ABC, abstractmethod

class Server(ABC):
    def __init__(self):
        super(ABC, self).__init__()
    
    @abstractmethod
    def test(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def aggregate(self, *args, **kwargs):
        pass
    
