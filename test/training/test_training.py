"""Tests training a classifier for Fashion MNIST.
In this repository, two model states are stored for a model trained in one run for 15 epochs.
"""

import os
SEED = 0
import random
import numpy as np
np.random.seed(0)
random.seed(SEED)
import torch
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from abcd.models.classification.FullyConnected import FullyConnected3
from abcd.training.ClassifierTrainer import ClassifierTrainer
from abcd.local.paths import output_path, test_data_path
from abcd.utils.pytorch import assert_same_state_dicts
from abcd.utils.io import load_df
from abcd.data.pytorch.get_dataloaders import get_train_dl, get_eval_dls

def test_fashionmnist_training():
    '''Test that training for 10 epochs was successfully executed and all relevant files were created'''
    local_output_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'local', 'test_objects', 'test_fashionmnist_training'))
    train_dataloader, eval_dataloaders = get_data()
    model, trainer = get_model_trainer(local_output_path)
    
    # Train model
    trainer.train(model, train_dataloader, eval_dataloaders, 
        nr_epochs=10, starting_from_epoch=0,
        print_loss_every=5, eval_every=5, export_every=5, verbose=False)

    # Check that all the files have been created
    assert os.path.isfile(os.path.join(local_output_path, 'models', 'FullyConnected3_epoch10.pth'))
    trainer_path = os.path.join(os.path.join(local_output_path, 'trainer'))
    assert os.path.isfile(os.path.join(trainer_path, 'Progress F1.svg'))
    assert os.path.isfile(os.path.join(trainer_path, 'Trajectory CrossEntropyLoss.svg'))
    assert os.path.isfile(os.path.join(trainer_path, 'confusion_matrices', 'CM_10_train.svg'))
    df = load_df(trainer_path, file_name='progress')
    df = df.loc[df['F1'] > 0.65]
    len(df) == 2 # Train and test results for epoch 10

def test_reproducibility():
    '''Test that the stored model after 10 epochs is the same as the one just trained'''
    stored_obj_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'test_objects', 'test_fashionmnist_training'))
    local_output_path = os.path.join(output_path, 'test_objects', 'test_fashionmnist_training')
    stored_model, _ = get_model_trainer(stored_obj_path)
    local_model, _ = get_model_trainer(local_output_path)
    for epoch in [0, 5, 10]:
        stored_model.restore(state_name='epoch{}'.format(epoch))
        local_model.restore(state_name='epoch{}'.format(epoch))
        assert_same_state_dicts(stored_model, local_model)
    
def test_continue_training():
    '''Continue training the model from the previous test for 5 more epochs. Compare it to the 
    stored model and files trained from scratch for 15 epochs stored in the repository'''
    starting_from_epoch = 5
    local_output_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'local', 'test_objects', 'test_fashionmnist_training'))
    train_dataloader, eval_dataloaders = get_data(at_epoch=starting_from_epoch)
    model, trainer = get_model_trainer(local_output_path)
    
    # Train model
    trainer.train(model, train_dataloader, eval_dataloaders, 
    nr_epochs=10, starting_from_epoch=starting_from_epoch,
    print_loss_every=5, eval_every=5, export_every=5, verbose=True)
    
def test_reproducibility_after_continuing():
    '''Test that the stored model after 10 epochs is the same as the one just trained'''
    stored_obj_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'test_objects', 'test_fashionmnist_training'))
    local_output_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'local', 'test_objects', 'test_fashionmnist_training'))
    stored_model, _ = get_model_trainer(stored_obj_path)
    local_model, _ = get_model_trainer(local_output_path)
    for epoch in [10, 15]:
        stored_model.restore(state_name='epoch{}'.format(epoch))
        local_model.restore(state_name='epoch{}'.format(epoch))
        assert_same_state_dicts(stored_model, local_model)

def get_data(at_epoch=0):
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

    batch_size = 64
    train_dataloader = get_train_dl(training_data, batch_size, seed=SEED, at_epoch=0, device="cpu")
    eval_datasets = {'train': training_data, 'test': test_data}
    eval_dataloaders = get_eval_dls(eval_datasets, batch_size)
    
    return train_dataloader, eval_dataloaders

def get_model_trainer(output_path):
    learning_rate = 1e-3
    loss_f = nn.CrossEntropyLoss()
    device = "cpu"
    labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    
    # Define model
    models_path = os.path.join(output_path, 'models')
    model = FullyConnected3(save_path=models_path, labels=labels, input_size=28*28)
    model = model.to(device)
    
    # Define optimizer and trainer
    trainer_path = os.path.join(output_path, 'trainer')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    trainer = ClassifierTrainer(trainer_path, device, optimizer, loss_f, labels=labels, seed=SEED)
    
    return model, trainer