"""A general trainer with logic to perform backpropagation while storing and communicating progress.
"""
import os
import random
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from abcd.utils.io import dump_df, load_df, dump_pkl, load_pkl
from abcd.plotting.pygal.training_progress import plot_progress
from abcd.plotting.pygal.rendering import save

class Trainer():
    def __init__(self, trainer_path, device, optimizer, loss_f, metrics=None, seed=None):
        self.trainer_path = trainer_path
        self.states_path = os.path.join(self.trainer_path, 'states')
        if not os.path.exists(self.states_path):
            os.makedirs(self.states_path)
        self.name = self.__class__.__name__
        self.device = device
        self.optimizer = optimizer
        self.loss_f = loss_f
        self.loss_tracking = []    
        # Store entries as [epoch, dataset name, metric 1, metric 2, ..., metric n]
        self.metrics = metrics
        self.progress = []
        self.losses = [loss_f.__class__.__name__]
        self.loss_trajectory = []
        self.seed = seed
        
    def train(self, model, train_dataloader, eval_dataloaders, 
              nr_epochs, starting_from_epoch=0,
              print_loss_every=5, eval_every=7, export_every=20, verbose=True):
        '''Trains a model for a given number of epochs. States and results are always recorded for
        the state before performing that epoch. E.g. state at epoch 0 is before epoch 0'''
        # If we are not continuing a previous training, initialize the progress elements
        if starting_from_epoch == 0:
            self.loss_trajectory, self.progress = [], []
        else:
            model.restore(state_name="epoch{}".format(starting_from_epoch))
            self.restore(state_name="epoch{}".format(starting_from_epoch))
            self.loss_trajectory = list(filter(lambda x: x[0] < starting_from_epoch, self.loss_trajectory))
            self.progress = list(filter(lambda x: x[0] < starting_from_epoch, self.progress))
        # Run training, storing intermediate progress
        for t in tqdm(range(starting_from_epoch, starting_from_epoch+nr_epochs)):
            # Run evaluation before executing the epoch
            if t % eval_every == 0:
                self.eval(model, eval_dataloaders, epoch_ix=t, verbose=verbose)
            if t % export_every == 0:
                self.export(model, state_name="epoch{}".format(t), verbose=verbose)
            if self.seed is not None:
                train_dataloader.generator.manual_seed(self.seed+t)            
            loss_value = self.train_epoch(model, train_dataloader)
            if t % print_loss_every == 0 and verbose:
                print("Ending epoch {}, loss {}".format(t+1, loss_value))
        # Save final progress
        if verbose:
            print('Finished training')
        self.eval(model, eval_dataloaders, epoch_ix=t+1, verbose=verbose)
        self.export(model, state_name="epoch{}".format(t+1), verbose=verbose)
        
    def train_epoch(self, model, dataloader, records_per_epoch=0):
        '''Trains a model for 1 epoch'''
        model.train()
        nr_batches = len(dataloader)
        total_loss = 0
        self.optimizer.zero_grad()
        for _, (X, y) in enumerate(dataloader):
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
    
    def export(self, model, state_name, only_results=False, verbose=False):
        '''Saves the model state, exports results until this point (as a csv) and updates plots'''
        if not only_results:
            model.save(state_name=state_name, verbose=verbose)
            self.save(state_name=state_name, verbose=verbose)
        progress_df = pd.DataFrame(self.progress, columns = ['Epoch', 'Dataset'] + self.metrics)
        dump_df(progress_df, self.trainer_path, file_name='progress')
        loss_trajectory_df = pd.DataFrame(self.loss_trajectory, columns = ['Epoch', 'Dataset'] + self.losses)
        dump_df(loss_trajectory_df, self.trainer_path, file_name='loss_trajectory')
        if not only_results:
            self.plot_progress(progress_df, loss_trajectory_df)
        if verbose:
            print('Progress stored in {}'.format(self.trainer_path))
            
    def save(self, state_name='last', verbose=False):
        '''Saves an optimizer state'''
        optimizer_state_name = self.name+'_optimizer_'+state_name+'.pth'
        torch.save(self.optimizer.state_dict(), os.path.join(self.states_path, optimizer_state_name))
        '''        
        # Save state of random number generators
        random_gens_name = self.name+'_random_gen_states_'+state_name
        random_gen_states = {'random': random.getstate(), 'numpy': np.random.get_state(), 
                  'torch': torch.get_rng_state()}
        try:
            random_gen_states['cuda'] = torch.cuda.get_rng_state()
        except:
            pass
        dump_pkl(random_gen_states, path=self.states_path, file_name=random_gens_name)
        '''
        if verbose:
            print("Saved trainer state {} in {}".format(optimizer_state_name, self.states_path))
            
    def restore(self, state_name):
        '''Restores an optimizer state'''
        # Restore states of random generators
        '''
        random_gens_name = self.name+'_random_gen_states_'+state_name
        random_gen_states = load_pkl(path=self.states_path, file_name=random_gens_name)
        random.setstate(random_gen_states['random'])
        np.random.set_state(random_gen_states['numpy'])
        torch.set_rng_state(random_gen_states['torch'])
        if 'cuda' in random_gen_states:
            torch.cuda.set_rng_state_all(random_gen_states['cuda'])
        '''
        # Restore optimizer state
        optimizer_state_name = self.name+'_optimizer_'+state_name+'.pth'
        #random_gens_name = self.name+'_random_gen_states_'+state_name
        self.optimizer.load_state_dict(torch.load(os.path.join(self.states_path, optimizer_state_name)))
        # Restore the progress and loss trajectory up to that part
        self.progress = load_df(self.trainer_path, file_name='progress').values.tolist()
        self.loss_trajectory = load_df(self.trainer_path, file_name='loss_trajectory').values.tolist()
        