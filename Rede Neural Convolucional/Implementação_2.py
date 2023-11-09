import torch
from torch import nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import Redes_Neurais as M
import aux
from PIL import Image
import os
import seaborn as sn
import matplotlib.pyplot as plt

DATA_DIR = '/home/gercom2/Documentos/Redes Neurais/Teste com Flower/Rede Neural Convolucional/cifar-10-batches-py'
CATEGORIES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def get_path(relpath):
  return os.path.join('/home/gercom2/Documentos/Redes Neurais/Teste com Flower/Rede Neural Convolucional', relpath)

cifar10_train = CIFAR10(DATA_DIR, train=True, download=True)
cifar10_test = CIFAR10(DATA_DIR, train=False, download=True)

#aux.lookat_dataset(cifar10_train)

prep_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(
            (0.4914, 0.4822, 0.4465), #img_mean
            (0.2470, 0.2435, 0.2616)  #img_std
        ) 
    ]
)

tensor_train = CIFAR10(DATA_DIR, train=True, download=False, transform=prep_transform)
tensor_test = CIFAR10(DATA_DIR, train=False, download=False, transform=prep_transform)

imgs = torch.stack([img for img, _ in tensor_train], dim=3)
#imgs_mean = imgs.view(3, -1).mean(dim=1)
#imgs_std = imgs.view(3, -1).std(dim=1)

batch_size = 64
train_loader = DataLoader(tensor_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(tensor_test, batch_size=batch_size, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = M.ConvolutionalModel().to(device)
optmizer = torch.optim.SGD(model.parameters(), lr=1e-3)
lossfunc = nn.CrossEntropyLoss()

epochs = 31
train_losses = []
test_losses = []

for t in range(epochs):
    train_loss = aux.train(model, train_loader, lossfunc, optmizer)
    train_losses.append(train_loss)
    if t % 5 == 0:
        print(f"Epoch {t} - Loss: {train_loss:.4f}")
    test_loss = aux.test(model, test_loader, lossfunc)
    test_losses.append(test_loss)
    
conv_confusion_matrix = aux.evaluate_accuracy(model, test_loader, CATEGORIES)

#img_passaro = Image.open('passaro.jpeg')
img_carro = Image.open(get_path("carro.jpg"))

prep_transform = T.Compose(
    [
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize(
            (0.4914, 0.4822, 0.4465), #img_mean
            (0.2470, 0.2435, 0.2616)  #img_std
        ) 
    ]
)

img_tensor = prep_transform(img_carro)

plt.imshow(img_tensor.permute(1,2,0))

batch = img_tensor.unsqueeze(0).to(device)

output = model(batch)

probs = torch.nn.functional.softmax(output, dim=1) *100
prob_dict = {}
for i, classname in enumerate(CATEGORIES):
    prob = probs[0][i].item()
    print(f"{classname} probabilidade: {prob:.2f}")
    prob_dict[classname] = [prob]