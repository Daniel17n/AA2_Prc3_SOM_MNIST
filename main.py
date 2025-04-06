import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from som import *  # Usaremos la implementación de SOM del notebook

def load_mnist_data():
    """Carga el dataset MNIST con PyTorch y lo normaliza."""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST(root="./data", train=True, download=True, transform=transform)
    
    x_train = torch.stack([data[0] for data in dataset])
    y_train = torch.tensor([data[1] for data in dataset])
    
    x_train = x_train.view(x_train.size(0), -1).numpy()
    x_train /= 255.0  # Normalización
    
    return x_train, y_train.numpy()

def train_som(x_train, grid_size=20, epochs=500):
    """Entrena el SOM con los datos de MNIST."""
    som = SOM(grid_size, grid_size)
    som.train(x_train, epochs, trace=1)
    return som

def map_labels_to_som(som, x_train, y_train):
    """Asigna etiquetas a las neuronas del SOM basado en las muestras entrenadas."""
    label_map = {}
    for img, label in zip(x_train, y_train):
        winner = som.predict(img)
        if winner in label_map:
            label_map[winner].append(label)
        else:
            label_map[winner] = [label]
    
    for key in label_map:
        label_map[key] = max(set(label_map[key]), key=label_map[key].count)
    
    return label_map

def plot_som_labels(label_map, grid_size=20):
    """Visualiza el mapa de etiquetas del SOM."""
    som_grid = np.full((grid_size, grid_size), -1)
    for (x, y), label in label_map.items():
        som_grid[x, y] = label
    
    plt.figure(figsize=(10, 10))
    for x in range(grid_size):
        for y in range(grid_size):
            plt.text(y, x, str(som_grid[x, y]) if som_grid[x, y] != -1 else '', ha='center', va='center', fontsize=8)
    plt.imshow(som_grid, cmap='coolwarm', alpha=0.6)
    plt.colorbar()
    plt.title("Mapa de etiquetas del SOM")
    plt.savefig("model_loss.png")

# Ejecutar el pipeline
x_train, y_train = load_mnist_data()
som = train_som(x_train, grid_size=2, epochs=5)
som.draw_map(x_train)
som.draw_mesh(x_train)
print(som.distribution(x_train, y_train))
label_map = map_labels_to_som(som, x_train, y_train)
plot_som_labels(label_map, grid_size=2)
