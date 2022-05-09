import numpy as np
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
import torch

class ClassificationMetrics:
    """
    ClassificationMetrics Class is used to calculate accuracy, precision, recall, and f1-score.
    """
    def __init__(self):
        """
        This init process will init empty dict to save result metrics.
        """
        self.result_metrics = dict()
    
    def __convert_to_numpy_array(self, y_true:torch.Tensor , y_pred:torch.Tensor):
        """
        This method is to used to convert torch.Tensor into Numpy Array.
        Args:
            y_true (torch.Tensor): batch of ground truth label.
            y_pred (torch.Tensor): batch of prediction label.
        Return:
            (tuple): return tuple of y_true and y_pred that converted into numpy array.
        """
        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()

        return (y_true, y_pred)

    def __calculate_metrics(self, y_true:np.array, y_pred:np.array, averages='macro')->dict:
        """
        This method will calculate classification metrics.
        Args:
            y_true (np.Array): batch of ground truth label.
            y_pred (np.Array): batch of prediction label.
            averages (str): averages type to calculate metrics.
        Return:
            result_dict (dict): Dictionary that contains result metrics.
        """
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        precision = precision_score(y_true= y_true, y_pred=y_pred, average=averages)
        recall = recall_score(y_true= y_true, y_pred=y_pred, average=averages)
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average=averages)
        result_dict = {
            'precision':precision,
            'accuracy': acc,
            'recall': recall,
            'f1_score': f1,
        }
        return result_dict

    def __call__(self, y_true:torch.Tensor, y_pred:torch.Tensor, averages='macro') -> dict:
        """
        This method will triggered when this object instances called and will calculate metrics.
        Args:
            y_true (torch.Tensor): batch of ground truth label.
            y_pred (torch.Tensor): batch of prediction label.
            averages (str): averages type to calculate metrics.
        Return:
            result_dict (dict): Dictionary that contains result metrics.
        """
        y_true, y_pred = self.__convert_to_numpy_array(y_true, y_pred)
        self.result_metrics = self.__calculate_metrics(y_true=y_true, y_pred=y_pred, averages=averages)
        return self.result_metrics
        