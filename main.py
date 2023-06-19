from fed.server import Server
import flwr as fl
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as MP

from experiments.fed.dataset import (
    prepare_client_loader,
    get_default_aug
)
from experiments.ssl.model import get_backbone
from model import BYOL
from fed.client import Client


if __name__ == '__main__':
    SEED = 123456
    NUM_CLIENTS = 10

    client_dataloaders = prepare_client_loader(
        NUM_CLIENTS, SEED
    )
    
    def setup_client(cid: str):
        dataloader = client_dataloaders[int(cid)]
        augmentation, _ = get_default_aug()
        backbone = get_backbone('resnet18')
        model = BYOL(
            backbone,
            512,
            1024,
            0.98,
            256,
            1024
        )
        client = Client(
            cid, model,
            dataloader,
            augmentaion=augmentation
        )
        return client
                        
    # writer = SummaryWriter()

    strategy = Server(
        num_client=5,
        local_epoch=5,
        # tensorboard_writer=writer,
        # cluster_size=5
    )

    fl.simulation.start_simulation(
        strategy=strategy,
        client_fn=setup_client,
        num_clients=5,
        client_resources={"num_cpus" : 2},
        config=fl.server.ServerConfig(num_rounds=2),
        ray_init_args={"include_dashboard": True}
    )

