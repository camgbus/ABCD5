"""A general trainer with logic to perform backpropagation while storing and communicating progress.
"""
import os
import torch
from tqdm import tqdm
import pandas as pd
from abcd.utils.io import dump_df, load_df
from abcd.plotting.pygal.training_progress import plot_progress
from abcd.plotting.pygal.rendering import save

class Trainer():
    def __init__(self, trainer_path, device, optimizer, loss_f, metrics=None):
        if not os.path.exists(trainer_path):
            os.makedirs(trainer_path)
        self.trainer_path = trainer_path
        self.device = device
        self.optimizer = optimizer
        self.loss_f = loss_f
        self.loss_tracking = []    
        # Store entries as [epoch, dataset name, metric 1, metric 2, ..., metric n]
        self.metrics = metrics
        self.progress = []
        self.losses = [loss_f.__class__.__name__]
        self.loss_trajectory = []
        
    def train(self, model, train_dataloader, eval_dataloaders, 
              nr_epochs, starting_from_epoch=0,
              print_loss_every=5, eval_every=7, export_every=20, verbose=True):
        '''Trains a model for a given number of epochs. States and results are always recorded for
        the state before performing that epoch. E.g. state at epoch 0 is before epoch 0'''
        model.train()
        # If we are not continuing a previous training, initialize the progress elements
        if starting_from_epoch == 0:
            self.loss_trajectory, self.progress = [], []
        else:
            model.restore(state_name="epoch{}".format(starting_from_epoch))
            #load_df(self.trainer_path, file_name='progress')
            #load_df(self.trainer_path, file_name='loss_trajectory')
        # Run training, toring intermediate progress
        for t in tqdm(range(starting_from_epoch, starting_from_epoch+nr_epochs)):
            # Run evaluation before executing the epoch
            if t % eval_every == 0:
                self.eval(model, eval_dataloaders, epoch_ix=t, verbose=verbose)
            if t % export_every == 0:
                self.export(model, state_name="epoch{}".format(t), verbose=verbose)            
            loss_value = self.train_epoch(model, train_dataloader)
            if t % print_loss_every == 0 and verbose:
                print("Starting epoch {}, loss {}".format(t+1, loss_value))
        # Save final progress
        if verbose:
            print('\nFinished training')
        self.eval(model, eval_dataloaders, epoch_ix=t+1, verbose=verbose)
        self.export(model, state_name="epoch{}".format(t+1), verbose=verbose)
        
    def train_epoch(self, model, dataloader, records_per_epoch=0):
        '''Trains a model for 1 epoch'''
        nr_batches = len(dataloader)
        total_loss = 0
        for batch, (X, y) in enumerate(dataloader):
            # Backpropagation step
            pred = model(X)
            loss = self.loss_f(pred, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += loss.item()
        return total_loss / nr_batches
                
    def eval(self, model, eval_dataloaders, epoch_ix, verbose=False):
        '''Evaluates a model w.r.t. given metrics. Prints and saves this progress.'''
        pass
    
    def print_progress(self, epoch_ix, progress_summary):
        print('Epoch {}'.format(epoch_ix))
        for dataset_name, summary in progress_summary.items():
            msg = dataset_name
            for score, val in summary.items():
                msg += " {0}: {1:.3f}".format(score, val)
            print(msg)
    
    def plot_progress(self, progress_df, loss_trajectory_df):
        ''''Plot one line plot for the loss trajetories, one for the metrics'''
        for metric in self.metrics:
            plot = plot_progress(progress_df, metric)
            save(plot, self.trainer_path, 'Progress {}'.format(metric))
        for loss in self.losses:
            plot = plot_progress(loss_trajectory_df, loss)
            save(plot, self.trainer_path, 'Trajectory {}'.format(loss))
    
    def export(self, model, state_name, verbose=False):
        '''Saves the model state, exports results until this point (as a csv) and updates plots'''
        model.save(state_name=state_name, verbose=verbose)
        progress_df = pd.DataFrame(self.progress, columns = ['Epoch', 'Dataset'] + self.metrics)
        dump_df(progress_df, self.trainer_path, file_name='progress')
        loss_trajectory_df = pd.DataFrame(self.loss_trajectory, columns = ['Epoch', 'Dataset'] + self.losses)
        dump_df(loss_trajectory_df, self.trainer_path, file_name='loss_trajectory')
        self.plot_progress(progress_df, loss_trajectory_df)
        if verbose:
            print('Progress stored in {}'.format(self.trainer_path))