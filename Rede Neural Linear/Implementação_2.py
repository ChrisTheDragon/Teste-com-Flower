# Importando os módulos necessários
import Aux as AT
import RedesNeurais as RN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from math import cos

# Definindo a função que será usada para gerar os dados de treino e teste
linha = lambda x: cos(x/2)
intervalo = (-10, 10)
nAmostras_treino = 1000
nAmostras_teste = 100

# Criando os datasets de treino e teste
train_dataset = RN.DatasetAlgebrico(linha, intervalo, nAmostras_treino)
test_dataset = RN.DatasetAlgebrico(linha, intervalo, nAmostras_teste)

# Criando os dataloaders de treino e teste
train_dataloader = DataLoader(train_dataset, nAmostras_treino, shuffle=True)
test_dataloader = DataLoader(test_dataset, nAmostras_treino, shuffle=True)

# Verificando se há disponibilidade de GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Criando a rede neural
modelo = RN.MultiLayerNetwork().to(device)

# Definindo a função de perda (loss function)
# Neste caso, usaremos o Erro Quadrático Médio (MSE)
funcPerda = nn.MSELoss()

# Definindo o otimizador
# Neste caso, usaremos o Stochastic Gradient Descent (SGD)
optimize = torch.optim.SGD(modelo.parameters(), lr=1e-3)

# Rodando o treinamento
for epoch in range(20001):
    train_loss = AT.train(modelo, train_dataloader, funcPerda, optimize)
    if epoch % 100 == 0:
        print(f"Epoch: {epoch+1} - Train Loss: {train_loss}")
        AT.plt_comparacao(linha, modelo, intervalo, nAmostras_teste)

# Testando o modelo
test_loss = AT.test(modelo, test_dataloader, funcPerda)
print(f"Teste Loss: {test_loss}")