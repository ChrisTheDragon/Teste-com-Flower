import os
import torch
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import Redes_Neurais as M
import aux


DATA_DIR = '/home/gercom2/Documentos/Redes Neurais/Teste com Flower/Rede Neural Convolucional/cifar-10-batches-py'
CATEGORIES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def get_path(relpath):
    return os.path.join('/home/gercom2/Documentos/Redes Neurais/Teste com Flower/Rede Neural Convolucional', relpath)


def main():
    # Define a transformação que será aplicada nas imagens
    prep_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                (0.4914, 0.4822, 0.4465),  # img_mean
                (0.2470, 0.2435, 0.2616)  # img_std
            )
        ]
    )

    # Carrega os dados de treino e teste
    tensor_train = CIFAR10(DATA_DIR, train=True, download=False, transform=prep_transform)
    tensor_test = CIFAR10(DATA_DIR, train=False, download=False, transform=prep_transform)

    # Define o tamanho do batch e cria os dataloaders
    batch_size = 64
    train_loader = DataLoader(tensor_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(tensor_test, batch_size=batch_size, shuffle=False)

    # Verifica se há GPU disponível
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define o modelo, otimizador e função de perda
    model = M.ConvolutionalModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss()

    # Define o número de épocas e as listas para armazenar as perdas de treino e teste
    epochs = 41
    train_losses = []
    test_losses = []

    # Treina o modelo
    for t in range(epochs):
        train_loss = aux.train(model, train_loader, loss_func, optimizer)
        train_losses.append(train_loss)
        if t % 2 == 0:
            print(f"Epoch {t} - Loss: {train_loss:.4f}")
        test_loss = aux.test(model, test_loader, loss_func)
        test_losses.append(test_loss)

    # Salva os pesos do modelo treinado
    torch.save(model.state_dict(), '/home/gercom2/Documentos/Redes Neurais/Teste com Flower/Rede Neural Convolucional/model.pth')


if __name__ == '__main__':
    main()
