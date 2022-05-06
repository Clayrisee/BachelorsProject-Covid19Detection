import torch
import numpy as np
import os

class CustomCallback:
    """
    This custom callbacks based on earlystop callback that will stop the training process if the desired metrics doesn't improve.
    """

    def __init__(self,
        checkpoint_path:str,
        metric:str,
        mode='min',
        patience=10, 
        verbose=False, delta=0,trace_func=print):
        """
        Args:
        checkpoint_path (str) : Path for the check point to be saved to.
        metric (str) : Metric that will be monitored during training process.
        mode (str) : metri monitoring mode ['min' or 'max']
        patience (int) : How long to wait after last metric improvement.
        verbose (bool) : If true, will prints a message for each monitored metri improvement.
        delta (float) : min change in the monitored quantity to qualife as an improvement.
        trace_func (function): trace print function.
        """
        self.checkpoint_path = checkpoint_path
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.trace_func = trace_func
        self.early_stop = False
        self.best_score = None
        self.counter = 0
        self.metric_min = np.Inf if mode == 'max' else -(np.Inf)

    def __call__(self, model, optimizer, metrics):
        """
        Args:
        model : Trained model in pytorch format
        metrics (dict) : Result dictionary performance metrics
        """
        score = metrics[self.metric]

        if self.best_score is None :
            self.best_score = score
            self.save_checkpoint(score, model, optimizer)
        
        else:
            if self.mode == 'min':
                if score > self.best_score + self.delta:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True
                else:
                    self.best_score = score
                    self.save_checkpoint(score, model, optimizer)
                    self.counter = 0

            elif self.mode == 'max':
                if score < self.best_score + self.delta:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True
                else:
                    self.best_score = score
                    self.save_checkpoint(score, model, optimizer)
                    self.counter = 0
            else: 
                NotImplementedError('Unknown Mode Please Choose Between min/max')
        
    
    def save_checkpoint(self, score, model, optimizer):
        """
        Save model when desired metric got best score.
        """
        # print(self.checkpoint_path)

        if self.verbose:
            self.trace_func(f'Val {self.metric} got best scored ({self.metric_min:.6f} --> {score:.6f}). Saving Model..')
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        torch.save(model.state_dict(), os.path.join(self.checkpoint_path, 'best_model.pth'))
        torch.save(optimizer.state_dict(), os.path.join(self.checkpoint_path, 'best_optimizer.pth'))
        self.metric_min = score
