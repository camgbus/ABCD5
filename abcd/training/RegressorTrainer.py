import os
import torch
import numpy as np
from abcd.training.Trainer import Trainer
from abcd.validation.metrics.regression import mean_absolute_error, mean_squared_error, r2_score

#TODO: Make this file
# from abcd.plotting.seaborn.regression_metrics import plot_regression_metrics

METRICS = {"MAE": mean_absolute_error, "MSE": mean_squared_error, "r2": r2_score}

class RegressorTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(RegressorTrainer, self).__init__(*args, **kwargs)
        if self.metrics is None:
            self.metrics = ["MAE", "MSE", "r2"]

    def eval(self, model, eval_dataloaders, epoch_ix, verbose=False):
        '''Evaluates a model w.r.t. given metrics. Prints and saves this progress.'''
        model.eval()
        progress_summary = dict()

        for dataloader_name, dataloader in eval_dataloaders.items():
            progress_summary[dataloader_name] = dict()
            nr_batches = len(dataloader)
            total_loss = 0
            targets = []
            predictions = []
            with torch.no_grad():
                for X, y in dataloader:
                    X, y = X.to(self.device), y.to(self.device)
                    pred = model(X)

                    total_loss += self.loss_f(pred, y).item()
                    targets += list(y.detach().cpu().numpy())
                    predictions += list(pred.detach().cpu().numpy())
            total_loss /= nr_batches
            # This trainer only stores the total loss
            self.loss_trajectory.append([epoch_ix, dataloader_name, float(total_loss)])
            self.progress.append([epoch_ix, dataloader_name] + [
                METRICS[metric_name](targets, predictions) for metric_name in self.metrics])
            
            #TODO: save plots here

            # Summarize values into progress for printing
            if verbose:
                for loss_name in self.losses:
                    progress_summary[dataloader_name][loss_name] = float(total_loss)
                for metric_name in self.metrics:
                    score = METRICS[metric_name](targets, predictions)
                    progress_summary[dataloader_name][metric_name] = score
        if verbose:
            self.print_progress(epoch_ix, progress_summary)


    # TODO: scatter plot of predictions vs targets