import torch
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch import nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ConvolutionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.convLayers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.linearLayers = nn.Sequential(
            nn.Linear(32 * 6 * 6, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.convLayers(x)
        x = torch.flatten(x, 1)
        return self.linearLayers(x)


def train(modelo, dataloader, epoch):
    funcPerda = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(modelo.parameters(), lr=1e-3)
    
    for _ in range(epoch):
        for imgs, labels in dataloader:
            optimizer.zero_grad()
            funcPerda(modelo(imgs.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(modelo, dataloader):
    funcPerda = nn.CrossEntropyLoss()
    correto, total, perda = 0, 0, 0.0
    
    with torch.no_grad():
        for imgs, labels in dataloader:
            output = modelo(imgs.to(DEVICE))
            perda += funcPerda(output, labels.to(DEVICE)).item()
            total += labels.size(0)
            correto += (torch.max(output.data, 1)[1] == labels.to(DEVICE)).sum().item()
    
    return perda/len(dataloader.dataset), correto/total


def load_data():
    prep_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                (0.4914, 0.4822, 0.4465),  # img_mean
                (0.2470, 0.2435, 0.2616)  # img_std
            )
        ]
    )
    
    trainset = CIFAR10(root='./data', train=True, download=True, transform=prep_transform)
    testset = CIFAR10(root='./data', train=False, download=True, transform=prep_transform)
    
    return DataLoader(trainset, batch_size=64, shuffle=True), DataLoader(testset)


def load_model():
    return ConvolutionalModel().to(DEVICE)


if __name__ == '__main__':
    trainloader, testloader = load_data()
    model = load_model()
    train(model, trainloader, 5)
    perda, acuracia = test(model, testloader)
    print(f"Perda: {perda:.5f} - Acc: {acuracia:.3f}")