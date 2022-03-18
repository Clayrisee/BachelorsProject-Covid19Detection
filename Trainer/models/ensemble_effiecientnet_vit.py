import timm
import torch.nn as nn
import torch


class EnsembleEfficientNetViT(nn.Module):
    """
    Ensemble 2 SOTA Model which are EfficientNet and Vision Transformer that has best accuracy at ImageNet Dataset.
    """
    def __init__(self, pretrained=True, output_class=4):
        """
        Init Model Layer using timm package.
        Args:
            pretrained (True): Choose if you want to use pretrained weight from ImageNet Dataset or Not.
            output_class (int): Output class of the model.
        """
        super(EnsembleEfficientNetViT, self).__init__()
        vit_small = timm.create_model('vit_small_patch32_224', pretrained=pretrained)
        efficientnet_b1 = timm.create_model('efficientnet_b1', pretrained=True)
        self.feature_extract_1 = nn.Sequential(vit_small.patch_embed, vit_small.pos_drop, vit_small.blocks,
                                    vit_small.norm, vit_small.pre_logits
                                    # ,nn.Flatten()
                                    )
        self.feature_extract_2 = nn.Sequential(
            efficientnet_b1.conv_stem, efficientnet_b1.bn1, efficientnet_b1.act1, efficientnet_b1.blocks,
            efficientnet_b1.conv_head, efficientnet_b1.bn2, efficientnet_b1.act2, efficientnet_b1.global_pool
        )
        self.linear= nn.Linear(20096, output_class)
    
    def forward(self, x):
        """
        Method to pass forward the batch input into each layer in dataset. (feature extract and classifier)
        Args:
            x (torch.Tensor) : Batch of Input Tensor.
        """
        output_1 = self.feature_extract_1(x) # pass forward vit model -> output shape: (1, 49, 384)
        output_2 = self.feature_extract_2(x) # pass forward convnext model -> output shape: (1, 1280)
        output_1 = output_1.reshape(-1, 1) # flatten -> flatten to (18816, 1)
        output_2 = output_2.reshape(-1, 1) # reshape to (1280, 1)
        merge_output = torch.concat((output_1, output_2)) # concat between to output -> output shape: (20096, 1)
        merge_output = merge_output.reshape(1, -1) # reshape to pass forward into linear layer -> output shape: (1, 20096)
        final_output = self.linear(merge_output) # pass forward into classifier layer
        return final_output

    def freeze(self):
        """
        Method to freeze weight at feature extractor.
        """
        for param in self.feature_extract_1.parameters():
            param.requires_grad = False
        
        for param_2 in self.feature_extract_2.paramaters():
            param_2.requires_grad = False
    
    def unfreeze(self):
        """
        Method to unfreeze weight at feature extractor.
        """
        for param in self.feature_extract_1.parameters():
            param.requires_grad = True
        for param_2 in self.feature_extract_2.paramaters():
            param_2.requires_grad = True