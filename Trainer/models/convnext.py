import timm
import torch.nn as nn
import torch.nn.functional as F

class ConvNext(nn.Module):
    def __init__(self, pretrained=True, output_class=4):
        """
        Init Model Layer using timm package.
        Args:
            pretrained (True): Choose if you want to use pretrained weight from ImageNet Dataset or Not.
            output_class (int): Output class of the model.
        """
        super(ConvNext, self).__init__()
        self.network = timm.create_model('convnext_base', pretrained=pretrained, num_classes=output_class)
    
    def forward(self, x):
        """
        Method to pass forward the batch input into each layer in dataset. (feature extract and classifier)
        Args:
            x (torch.Tensor) : Batch of Input Tensor.
        """
        x = self.network(x)
        return F.softmax(x, dim=1)
    
    def freeze(self):
        """
        Method to freeze weight at feature extractor.
        """
        for param in self.network.parameters():
            param.requires_grad = False
        for param in self.network.head.parameters():
            param.requires_grad = True
    
    def unfreeze(self):
        """
        Method to unfreeze weight at feature extractor.
        """
        for param in self.network.parameters():
            param.requires_grad = True