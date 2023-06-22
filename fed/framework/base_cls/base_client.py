from abc import ABC, abstractmethod
class Client():
    def __init__(self):
        super(Client, self).__init__()
        
    @abstractmethod
    def train(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def test(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def get_data(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def set_data(self, *args, **kwargs):
        pass
    
    