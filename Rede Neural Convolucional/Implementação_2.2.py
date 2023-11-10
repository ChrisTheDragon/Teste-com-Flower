import torch
import Redes_Neurais as M
from PIL import Image
import os
import torchvision.transforms as T
import matplotlib.pyplot as plt
import pandas as pd



CATEGORIES = ['aviao','carro','passaro','gato','veado','cachorro','sapo','cavalo','navio','caminhao']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = M.ConvolutionalModel().to(device)
model.load_state_dict(torch.load('/home/gercom2/Documentos/Redes Neurais/Teste com Flower/Rede Neural Convolucional/model.pth'))
#print(model.eval())

def get_path(relpath):
  return os.path.join('/home/gercom2/Documentos/Redes Neurais/Teste com Flower/Rede Neural Convolucional', relpath)

img_passaro = Image.open(get_path('passaro.jpeg'))
img_carro = Image.open(get_path("carro.jpg"))
img_cavalo = Image.open(get_path("cavalo.jpg"))

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

img_tensor = prep_transform(img_cavalo)

plt.imshow(img_tensor.permute(1,2,0))

batch = img_tensor.unsqueeze(0).to(device)

output = model(batch)

probs = torch.nn.functional.softmax(output, dim=1) *100
prob_dict = {}
for i, classname in enumerate(CATEGORIES):
    prob = probs[0][i].item()
    print(f"{classname} probabilidade: {prob:.2f}")
    prob_dict[classname] = [prob]
    
df_prob = pd.DataFrame.from_dict(prob_dict)
df_prob.plot(kind='bar', figsize=(10, 6))