from fed.server import Server
from fed.client import setup_client
import flwr as fl

strategy = Server()

fl.simulation.start_simulation(
    strategy=strategy,
    client_fn=setup_client,
    num_clients=2,
    client_resources={"num_cpus" : 2},
    config=fl.server.ServerConfig(num_rounds=2)
)
