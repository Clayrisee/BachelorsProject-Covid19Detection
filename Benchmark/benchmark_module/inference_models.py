import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class InferenceModel(nn.Module):
    def __init__(self, input_size, backbone, output_class, device):
        super(InferenceModel, self).__init__()
        self.input_size = input_size
        self.network = timm.create_model(self._get_backbone_names(backbone=backbone), 
                            pretrained=False, num_classes=output_class)
        
        self.device = device
        self.network.to(device)
    
    def _get_backbone_names(self, backbone:str):
        backbone_dict = {
            'vit': 'vit_small_patch32_224',
            'efficientnet_b0':'efficientnet_b0',
            'convnext': 'convnext_base'
        }
        return backbone_dict[backbone]
    
    def forward(self, x):
        """
        Method to pass forward the batch input into each layer in dataset. (feature extract and classifier)
        Args:
            x (torch.Tensor) : Batch of Input Tensor.
        """
        x = self.network(x)
        return F.softmax(x, dim=1)
    
    def preprocessing_img(self, img):
        preprocess = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return preprocess(img)
    
    @torch.no_grad()
    def predict(self, input_img):
        preprocess_img = torch.unsqueeze(self.preprocessing_img(input_img), dim=0)
        # print(preprocess_img.shape)
        preprocess_img = preprocess_img.to(self.device)
        return self(preprocess_img)
