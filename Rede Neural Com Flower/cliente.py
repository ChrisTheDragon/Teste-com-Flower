from centralizado import load_data, load_model, train, test, ConvolutionalModel
from collections import OrderedDict
import torch
import flwr as fl

def set_parameters(model: ConvolutionalModel, parameters):
    """
    Define os parâmetros do modelo com base em uma lista de parâmetros fornecida.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model

def load_model_from_file(file_path):
    """
    Carrega o modelo a partir de um arquivo.
    """
    modelo = load_model()  # Use a função adequada para criar o modelo inicial
    modelo.load_state_dict(torch.load(file_path))
    return modelo

# Verifica se há um modelo treinado anteriormente
try:
    modelo = load_model_from_file("modelo_treinado.pth")
    print("Modelo treinado encontrado. Carregando...")
except FileNotFoundError:
    modelo = load_model()
    print("Nenhum modelo treinado encontrado. Inicializando um novo modelo.")

trainloader, testloader = load_data()

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        """
        Retorna os parâmetros do modelo.
        """
        return [val.cpu().numpy() for _, val in modelo.state_dict().items()]

    def fit(self, parameters, config):
        """
        Realiza o treinamento do modelo com base nos parâmetros fornecidos.
        """
        set_parameters(modelo, parameters)
        train(modelo, trainloader, epoch=5)
        
        # Salvar o modelo após o treinamento
        torch.save(modelo.state_dict(), "modelo_treinado.pth")
        
        return self.get_parameters({}), len(trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        """
        Avalia o modelo com base nos parâmetros fornecidos.
        """
        set_parameters(modelo, parameters)
        perda, acuracia = test(modelo, testloader)
        return float(perda), len(testloader.dataset), {"acuracia": float(acuracia)}

# Código para iniciar o cliente Flower
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080", 
    client=FlowerClient()
)
