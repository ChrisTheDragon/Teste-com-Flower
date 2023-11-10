import os
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

import Redes_Neurais as M

# Lista de categorias
CATEGORIES = ['aviao', 'carro', 'passaro', 'gato', 'veado', 'cachorro', 'sapo', 'cavalo', 'navio', 'caminhao']

# Verifica se há GPU disponível
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Carrega o modelo treinado
model = M.ConvolutionalModel().to(device)
model.load_state_dict(torch.load('/home/gercom2/Documentos/Redes Neurais/Teste com Flower/Rede Neural Convolucional/model.pth'))

# Função para obter o caminho completo do arquivo
def get_path(relpath):
    return os.path.join('/home/gercom2/Documentos/Redes Neurais/Teste com Flower/Rede Neural Convolucional', relpath)

# Carrega as imagens
img_passaro = Image.open(get_path('passaro.jpeg'))
img_carro = Image.open(get_path("carro.jpg"))
img_carro2 = Image.open(get_path('carro_2.jpg'))
img_cavalo = Image.open(get_path("cavalo.jpg"))
img_aviao = Image.open(get_path("aviao.jpg"))

# Transformação para preparar as imagens para a entrada na rede neural
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

# Aplica a transformação na imagem
img_tensor = prep_transform(img_aviao)

# Exibe a imagem
plt.imshow(img_tensor.permute(1,2,0))

# Cria um batch com a imagem
batch = img_tensor.unsqueeze(0).to(device)

# Executa a rede neural com a imagem
output = model(batch)

# Calcula as probabilidades de cada categoria
probs = torch.nn.functional.softmax(output, dim=1) * 100
prob_dict = {}
for i, classname in enumerate(CATEGORIES):
    prob = probs[0][i].item()
    print(f"{classname} probabilidade: {prob:.2f}")
    prob_dict[classname] = [prob]

# Cria um dataframe com as probabilidades
df_prob = pd.DataFrame.from_dict(prob_dict)

# Exibe um gráfico de barras com as probabilidades
df_prob.plot(kind='bar', figsize=(10, 6))
