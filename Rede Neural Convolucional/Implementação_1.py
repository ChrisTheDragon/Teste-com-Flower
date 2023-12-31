import torch
from torch import nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import Redes_Neurais as M
import aux
import seaborn as sn
import matplotlib.pyplot as plt

# Define o diretório onde os dados estão armazenados e as categorias possíveis
DATA_DIR = '/home/gercom2/Documentos/Redes Neurais/Teste com Flower/Rede Neural Convolucional/cifar-10-batches-py'
CATEGORIES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Carrega os dados de treino e teste
cifar10_train = CIFAR10(DATA_DIR, train=True, download=True)
cifar10_test = CIFAR10(DATA_DIR, train=False, download=True)

# Define a transformação que será aplicada nas imagens
prep_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(
            (0.4914, 0.4822, 0.4465), #img_mean
            (0.2470, 0.2435, 0.2616)  #img_std
        ) 
    ]
)

# Aplica a transformação nas imagens de treino e teste
tensor_train = CIFAR10(DATA_DIR, train=True, download=False, transform=prep_transform)
tensor_test = CIFAR10(DATA_DIR, train=False, download=False, transform=prep_transform)

# Empilha as imagens de treino para calcular a média e o desvio padrão
imgs = torch.stack([img for img, _ in tensor_train], dim=3)
#imgs_mean = imgs.view(3, -1).mean(dim=1)
#imgs_std = imgs.view(3, -1).std(dim=1)

# Define o tamanho do batch e cria os dataloaders de treino e teste
batch_size = 64
train_loader = DataLoader(tensor_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(tensor_test, batch_size=batch_size, shuffle=False)

# Verifica se há GPU disponível e define o dispositivo
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Instancia o modelo, o otimizador e a função de perda
model = M.MLPClassifier().to(device)
optmizer = torch.optim.SGD(model.parameters(), lr=1e-3)
lossfunc = nn.CrossEntropyLoss()

# Define o número de épocas e as listas para armazenar as perdas de treino e teste
epochs = 31
train_losses = []
test_losses = []

# Loop de treinamento
for t in range(epochs):
    train_loss = aux.train(model, train_loader, lossfunc, optmizer)
    train_losses.append(train_loss)
    if t % 10 == 0:
        print(f"Epoch {t} - Loss: {train_loss:.4f}")
    test_loss = aux.test(model, test_loader, lossfunc)
    test_losses.append(test_loss)
    
# Plota as curvas de aprendizado
losses = {"Train loss": train_losses, "Test loss": test_losses}
aux.plot_losses(losses)

# Calcula a matriz de confusão e plota o heatmap
confusion_matrix = aux.evaluate_accuracy(model, test_loader, CATEGORIES)
plt.figure(figsize=(12, 12))
sn.set(font_scale=1.4)
sn.heatmap(confusion_matrix.tolist(), annot=True, annot_kws={"size": 16}, fmt='d')