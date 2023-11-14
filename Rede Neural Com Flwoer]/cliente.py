from centralizado import load_data, load_model, train, test
from collections import OrderedDict

import torch
import flwr as fl

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model


modelo = load_model()
trainloader, testloader = load_data()

class FlowerClient(fl.client.numpy_client):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in modelo.state_dict().items()]

    def fit(self, parameters, config):
        set_parameters(modelo, parameters)
        train(modelo, trainloader, epoch=1)
        return self.get_parameters({}), len(trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        set_parameters(modelo, parameters)
        perda, acuracia = test(modelo, testloader)
        return float(perda), len(testloader.dataset), {"acuracia": float(acuracia)}
    


fl.client.start_numpy_client(
    server_address="localhost:8080", client=FlowerClient()
)