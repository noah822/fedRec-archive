from fed import framework

class Client(framework.standard.Client):
    def __init__(self):
        super().__init__()
        self.mave = None
        self.audio_autoencoder = None
        self.visual_autoencoder = None
    
    def set_data(self, download):
        generation_net_state_dict = download['generation']
        autoencoder = download['autoencoder']
    
    def get_data(self):
        return None
    
    def train(self):
        pass
    
    def test(self):
        pass
        


class Server(framework.standard.Server):
    def __init__(self, public):
        super().__init__()