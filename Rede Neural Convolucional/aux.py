import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(modelo, dataloader: DataLoader, funcPerda, optimizer):
    """
    Função de treinamento da rede neural.

    Args:
        modelo: Modelo da rede neural.
        dataloader (DataLoader): Dataloader com os dados de treinamento.
        funcPerda: Função de perda.
        optimizer: Otimizador.

    Returns:
        float: Média das perdas.
    """
    modelo.train()
    cumloss = 0.0
    
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        pred = modelo(imgs)
        loss = funcPerda(pred, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        cumloss += loss.item()
    
    return cumloss/len(dataloader)


def test(modelo, dataloader: DataLoader, funcPerda):
    """
    Função de teste da rede neural.

    Args:
        modelo: Modelo da rede neural.
        dataloader (DataLoader): Dataloader com os dados de teste.
        funcPerda: Função de perda.

    Returns:
        float: Média das perdas.
    """
    modelo.eval()
    
    cumloss = 0.0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            pred = modelo(imgs)
            loss = funcPerda(pred, labels)
            cumloss += loss.item()
    
    return cumloss/len(dataloader)


def lookat_dataset(dataset, istensor=False):
    figure = plt.figure(figsize=(8, 8))
    rows, cols = 2, 2
    for i in range(1, 5):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        if istensor:
            plt.imshow(img)
            #plt.imshow(img.permute(1, 2, 0))
        else:
            plt.imshow(img)
    plt.show()


def plot_losses(losses: dict):
    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()
    for loss_name, loss_values in losses.items():
        ax.plot(loss_values, label=loss_name)
    ax.legend(fontsize='16')
    ax.set_xlabel('Iteration', fontsize='16')
    ax.set_ylabel('Loss', fontsize='16')
    ax.set_title('Losses vs Iterations', fontsize='16')
   
 
def make_confusion_matrix(model, loader, n_classes):
  confusion_matrix = torch.zeros(n_classes, n_classes, dtype=torch.int64)
  with torch.no_grad():
    for i, (imgs, labels) in enumerate(loader):
      imgs = imgs.to(device)
      labels = labels.to(device)
      outputs = model(imgs)
      _, predicted = torch.max(outputs, 1)
      for t, p in zip(torch.as_tensor(labels, dtype=torch.int64).view(-1), 
                      torch.as_tensor(predicted, dtype=torch.int64).view(-1)):
        confusion_matrix[t, p] += 1
  return confusion_matrix


def evaluate_accuracy(model, dataloader, classes, verbose=True):
  # prepare to count predictions for each class
  correct_pred = {classname: 0 for classname in classes}
  total_pred = {classname: 0 for classname in classes}

  confusion_matrix = make_confusion_matrix(model, dataloader, len(classes))
  if verbose:
    total_correct = 0.0
    total_prediction = 0.0
    for i, classname in enumerate(classes):
      correct_count = confusion_matrix[i][i].item()
      class_pred = torch.sum(confusion_matrix[i]).item()

      total_correct += correct_count
      total_prediction += class_pred

      accuracy = 100 * float(correct_count) / class_pred
      print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                    accuracy))
  print("Global acccuracy is {:.1f}".format(100 * total_correct/total_prediction))
  return confusion_matrix


def test_2(model, dataloader, classes):
  # prepare to count predictions for each class
  correct_pred = {classname: 0 for classname in classes}
  total_pred = {classname: 0 for classname in classes}

  # again no gradients needed
  with torch.no_grad():
      for images, labels in dataloader:
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)
          _, predictions = torch.max(outputs, 1)
          # collect the correct predictions for each class
          for label, prediction in zip(labels, predictions):
              if label == prediction:
                  correct_pred[classes[label]] += 1
              total_pred[classes[label]] += 1

  # print accuracy for each class
  total_correct = 0.0
  total_prediction = 0.0
  for classname, correct_count in correct_pred.items():
      total_correct += correct_count
      total_prediction += total_pred[classname]
      accuracy = 100 * float(correct_count) / total_pred[classname]
      print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                    accuracy))
  print("Global acccuracy is {:.1f}".format(100 * total_correct/total_prediction))

