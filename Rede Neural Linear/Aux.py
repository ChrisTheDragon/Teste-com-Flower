import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import RedesNeurais as RN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(modelo, dataloader: DataLoader, funcPerda, optimizer):
    """
    Função de treinamento da rede neural.

    Args:
        modelo (RN.LineNetwork): Modelo da rede neural.
        dataloader (DataLoader): Dataloader com os dados de treinamento.
        funcPerda (nn.MSELoss): Função de perda.
        optimizer (torch.optim.SGD): Otimizador.

    Returns:
        float: Média das perdas.
    """
    modelo.train()
    cumloss = 0.0
    
    for X, y in dataloader:
        X, y = X.unsqueeze(1).float().to(device), y.unsqueeze(1).float().to(device)
        
        pred = modelo(X)
        loss = funcPerda(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        cumloss += loss.item()
    
    return cumloss/len(dataloader)

def test(modelo, dataloader: DataLoader, funcPerda):
    """
    Função de teste da rede neural.

    Args:
        modelo (RN.LineNetwork): Modelo da rede neural.
        dataloader (DataLoader): Dataloader com os dados de teste.
        funcPerda (nn.MSELoss): Função de perda.

    Returns:
        float: Média das perdas.
    """
    modelo.eval()
    
    cumloss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.unsqueeze(1).float().to(device), y.unsqueeze(1).float().to(device)
            
            pred = modelo(X)
            loss = funcPerda(pred, y)
            cumloss += loss.item()
    
    return cumloss/len(dataloader)

def plt_comparacao(f, model, intervalo=(-10, 10), n_amostras=10):
    """
    Função para plotar a comparação entre a função original e a função gerada pela rede neural.

    Args:
        f (function): Função original.
        model (RN.LineNetwork): Modelo da rede neural.
        intervalo (tuple, optional): Intervalo de valores para plotar. Defaults to (-10, 10).
        n_amostras (int, optional): Número de amostras para plotar. Defaults to 10.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.grid(True, which='both')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    
    samples = np.linspace(intervalo[0], intervalo[1], n_amostras)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(samples).unsqueeze(1).float().to(device))
        
    ax.plot(samples, list(map(f, samples)), 'o', label='Resultado Esperado')
    ax.plot(samples, pred.cpu(), label='Resultado do Modelo')
    plt.legend()
    plt.show()
