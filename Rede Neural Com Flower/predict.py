import torch
from centralizado import load_model
from PIL import Image
from torchvision import transforms

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CLASSES = ['aviao', 'carro', 'passaro', 'gato', 'veado', 'cachorro', 'sapo', 'cavalo', 'navio', 'caminhao']

# Função para carregar e processar uma única imagem
def process_image(image_path):
    # Carregar a imagem usando a PIL
    image = Image.open(image_path).convert("RGB")
    
    # Aplicar transformações
    prep_transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Redimensionar para o tamanho esperado
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),  # img_mean
            (0.2470, 0.2435, 0.2616)   # img_std
        )
    ])
    
    # Aplicar transformações à imagem
    image = prep_transform(image).unsqueeze(0)  # Adicionar dimensão do lote (batch dimension)
    
    return image

# Função para realizar a predição usando o modelo treinado
def predict_image(model, image_path):
    model.eval()  # Modo de avaliação
    
    # Processar a imagem
    input_image = process_image(image_path)
    
    # Enviar a imagem para o dispositivo (CPU ou GPU)
    input_image = input_image.to(DEVICE)
    
    # Fazer a predição
    with torch.no_grad():
        output = model(input_image)
    
    # Obter a classe prevista
    _, predicted_class = torch.max(output, 1)
    
    return predicted_class.item()

def predict_prob(model, image_path):
    model.eval()  # Modo de avaliação
    
    # Processar a imagem
    input_image = process_image(image_path)
    
    # Enviar a imagem para o dispositivo (CPU ou GPU)
    input_image = input_image.to(DEVICE)
    
    output = model(input_image)
    
    probs = torch.nn.functional.softmax(output, dim=1) * 100
    prob_dict = {}
    for i, classname in enumerate(CLASSES):
        prob = probs[0][i].item()
        print(f"{classname} probabilidade: {prob:.2f}")
        prob_dict[classname] = [prob]
    

# Função para carregar o modelo a partir de um arquivo
def load_model_from_file(file_path):
    modelo = load_model()  # Use a função adequada para criar o modelo inicial
    modelo.load_state_dict(torch.load(file_path))
    return modelo

# Exemplo de uso
if __name__ == '__main__':
    # Substitua 'caminho/para/sua/imagem.jpg' pelo caminho real da sua imagem
    image_path = '/home/gercom2/Documentos/Redes Neurais/Teste com Flower/Rede Neural Com Flower/Imagens/veado.jpg'
    
    # Carregar modelo treinado
    trained_model = load_model_from_file("modelo_treinado.pth")
    
    # Fazer a predição
    op = input('Digite 1 para predição de classe ou 2 para predição de probabilidade: ')
    if op == '1':
        predicted_class = predict_image(trained_model, image_path)
        print(f"A imagem é da classe: {CLASSES[predicted_class]}")
    elif op == '2':
        predict_prob(trained_model, image_path)
    else:
        print('Opção inválida!')
    
    #predicted_class = predict_image(trained_model, image_path)
    #predicted_class = predict_prob(trained_model, image_path)
    
    #print(f"A imagem é da classe: {CLASSES[predicted_class]}")
