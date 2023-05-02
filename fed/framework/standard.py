from . import base_cls
from typing import List

class Client(base_cls.Client):
    def __init__(self):
        super().__init__()

class Server(base_cls.Server):
    def __init__(self):
        super().__init__()
        
        
        
class Framework(base_cls.Framework):
    def __init__(self,
        clients: List[Client],
        client_train_config,
        server: Server,
        num_round: int,
        initial_model = None,
        save_path = 'framework',
        logging_path = 'logs'
    ):
        self.clients = clients
        self.server = server
        self.num_round = num_round
        self.num_clients = len(clients)
        self.initial_model = initial_model
        self.logging_path = 'logs'
        
        # initialize server and clients
    
    def run(self):
        distribution = self.initial_model
        for cur_round in range(self.num_round):
            # distribute model to client
            if distribution is not None:
                for client in self.clients:
                    client.set_data(distribution)
                
            upload_data = []
            
            # client training
            with open(self.logging_path, 'a') as logger:
                for idx, client in enumerate(self.clients):
                    _train_res = client.train()
                    upload_data.append(client.get_data())
                logger.write(
                    f'client[{idx}/{self.clients}] train loss: {_train_res:3f}\n'
                )
            
            distribution = self.server.aggregate(upload_data)
    
                
            
                