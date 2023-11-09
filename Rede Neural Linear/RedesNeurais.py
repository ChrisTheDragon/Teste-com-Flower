import torch.nn as nn
import torch.distributions.uniform as urand
from torch.utils.data import Dataset


class LineNetwork(nn.Module):
    """
    Classe que define a rede neural com uma camada linear.
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 1)
        )
    
    def forward(self, x):
        """
        Define o fluxo de dados na rede neural.
        """
        return self.layers(x)


class MultiLayerNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        """
        Define o fluxo de dados na rede neural.
        """
        return self.layers(x)


class DatasetAlgebrico(Dataset):
    """
    Classe que define o dataset para treinamento da rede neural.
    """
    def __init__(self, funcao, intervalo, n_amostras):
        """
        Inicializa o dataset com amostras aleatórias.
        """
        X = urand.Uniform(intervalo[0], intervalo[1]).sample([n_amostras])
        self.data = [(x, funcao(x)) for x in X]
    
    def __len__(self):
        """
        Retorna o tamanho do dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retorna uma amostra do dataset.
        """
        return self.data[idx]

