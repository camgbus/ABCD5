"""Tests training a classifier for Fashion MNIST.
"""

import os
import torch
torch.manual_seed(0)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from abcd.models.classification.FullyConnected import FullyConnected3
from abcd.training.ClassifierTrainer import ClassifierTrainer
from abcd.local.paths import test_data_path

def test_fashionmnist_training():
    
    output_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'test_objects', 'test_fashionmnist_training'))
    
    # Download data from the open datasets
    training_data = datasets.FashionMNIST(
        root=test_data_path,
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.FashionMNIST(
        root=test_data_path,
        train=False,
        download=True,
        transform=ToTensor(),
    )
    labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    
    batch_size = 64
    learning_rate = 1e-3
    loss_f = nn.CrossEntropyLoss()

    # Create dataloaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    eval_dataloaders = {'Train': train_dataloader, 'Test': test_dataloader}
    
    # Determine device for training
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using {} device".format(device))
    
    # Define model
    models_path = os.path.join(output_path, 'models')
    model = FullyConnected3(save_path=models_path, labels=labels)
    model = model.to(device)
    print(model)
    
    # Define optimizer and trainer
    trainer_path = os.path.join(output_path, 'results')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    trainer = ClassifierTrainer(trainer_path, device, optimizer, loss_f, labels=labels)
    
    # Train model
    trainer.train(model, train_dataloader, eval_dataloaders, 
    nr_epochs=20, starting_from_epoch=0,
    print_loss_every=5, eval_every=5, export_every=10, verbose=False)
    
test_fashionmnist_training()
