import os
import torch
import numpy as np
from abcd.training.Trainer import Trainer
from abcd.validation.metrics.regression import mean_absolute_error, mean_squared_error, r2_score, accuracy, discretize
from abcd.validation.metrics.classification import confusion_matrix
from abcd.plotting.seaborn.scatter import plot_regression_scatter
from abcd.plotting.seaborn.confusion_matrix import plot_confusion_matrix
from abcd.plotting.seaborn.rendering import save

METRICS = {"MAE": mean_absolute_error, "MSE": mean_squared_error, "r2": r2_score,  "accuracy": accuracy}

class RegressorTrainer(Trainer):
    def __init__(self, trainer_path, device, optimizer, loss_f, 
                    thresholds=None, class_names=None, metrics=None, seed=None):
        """
        class_names and thresholds are used for discretizing predictions into classes. If these parameters 
        are all not None, then accuracy will be added to metrics (if no metrics provided) and confusion matrices 
        will be saved for the best model (model with lowest validation MSE).
        """
        self.class_names = class_names #List[str] e.g. ['male', 'female']
        self.thresholds = thresholds #List[float] e.g. [0.0, 0.5, 1.0]
        self.discretizing = bool(class_names and thresholds)

        super(RegressorTrainer, self).__init__(trainer_path, device, optimizer, loss_f, metrics, seed)

        # set default metrics
        if self.metrics is None:
            self.metrics = ["MSE", "MAE"]
            if self.discretizing:
                self.metrics.append("accuracy")
        elif "MSE" not in self.metrics:
            self.metrics.append("MSE") #need for saving best model
        
        # for saving best model
        self.val_metric = "MSE"
        self.best_val_metric = None
        self.best_model_details = None


    def eval(self, model, eval_dataloaders, epoch_ix, verbose=False):
        '''Evaluates a model w.r.t. given metrics. Prints and saves this progress.'''
        model.eval()
        progress_summary = dict()
        cms = {}

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
            metrics_values = [METRICS[metric_name](targets, predictions) if metric_name != "accuracy"
                              else accuracy(targets, predictions, self.class_names, self.thresholds) for metric_name in self.metrics]
            self.progress.append([epoch_ix, dataloader_name] + metrics_values)

            # Confusion matrices
            if self.discretizing:
                discretized_predictions = discretize(predictions, self.class_names, self.thresholds)
                discretized_targets = discretize(targets, self.class_names, self.thresholds)
                cm = confusion_matrix(discretized_targets, discretized_predictions)
                cms[dataloader_name] = cm

            # Summarize values into progress for printing progress and updating best model
            for loss_name in self.losses:
                progress_summary[dataloader_name][loss_name] = float(total_loss)
            for metric_name in self.metrics:
                if metric_name == "accuracy":
                    score = METRICS[metric_name](targets, predictions, self.class_names, self.thresholds)
                else:
                    score = METRICS[metric_name](targets, predictions)
                progress_summary[dataloader_name][metric_name] = score

        if verbose:
            self.print_progress(epoch_ix, progress_summary)

        # Save best model info
        if "val" in eval_dataloaders:  # assumes validation dataloader named 'val'
            curr_val_metric = progress_summary["val"][self.val_metric]
            if self.best_val_metric is None or curr_val_metric < self.best_val_metric:
                self.best_val_metric = curr_val_metric
                self.best_model_details = {
                    'state_dict': model.state_dict(),
                    'metrics': progress_summary,
                    'epoch_ix': epoch_ix,
                    'architecture': str(model),
                    'cms': cms,
                }
                if verbose:
                    print("\nNew best model.\n")


    def export_best_model(self, config=None):
        best_model_dir = os.path.join(self.trainer_path, 'best_model')
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)

        # save model state
        torch.save(self.best_model_details['state_dict'], os.path.join(best_model_dir, 'best_model.pth'))

        # save confusion matrices
        if self.discretizing:
            for key in self.best_model_details['cms']:
                cm = self.best_model_details['cms'][key]
                self.plot_confusion_matrix(cm, file_name="CM_best_model_{}".format(key), path=os.path.join(best_model_dir))

        # save model details to .txt file
        with open(os.path.join(best_model_dir, "best_model_details.txt"), 'w') as f:
            f.write(self.best_model_details['architecture'] + "\n\n")
            
            if config:
                for key,val in config.items():
                    f.write("{}: {}\n".format(key, val))
            
            f.write("stopped at epoch: {}\n".format(self.best_model_details['epoch_ix']))

            for key, value in self.best_model_details['metrics'].items():
                f.write(f"\n{key} metrics:\n")
                for metric, score in value.items():
                    if isinstance(score, float):
                        score = round(score, 3) #3 decimal places
                    f.write(f"{metric}: {score}\n")
            
        return self.best_model_details
        

    def plot_confusion_matrix(self, cm, file_name, path=None):
        if not path: path = self.trainer_path
        plot = plot_confusion_matrix(cm, labels=self.class_names, figure_size=(12,10))
        path = os.path.join(path, 'confusion_matrices')
        if not os.path.exists(path):
            os.makedirs(path)
        save(plot, path, file_name=file_name)
        plot.close()


    def plot_scatter(self, targets, predictions, file_name):
        '''Plot the scatter plot for the given epoch.'''
        plot = plot_regression_scatter(predictions, targets)
        path = os.path.join(self.trainer_path, 'scatter_plots')
        if not os.path.exists(path):
            os.makedirs(path)
        save(plot, path, file_name=file_name)
        plot.close()