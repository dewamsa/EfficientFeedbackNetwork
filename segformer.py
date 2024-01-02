import torch
from torch import nn
from torchinfo import summary

from transformers import SegformerForSemanticSegmentation, SegformerConfig


class SegformerHF_pretrained(nn.Module):
    def __init__(
        self,
        channels = 3,
        num_classes=2
    ):
        super().__init__()
        pretrained_model_name = "nvidia/mit-b0" 
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name,
            id2label=[0,1],
            label2id=[0,1]
        )
        self.up = nn.Upsample(scale_factor=4,mode='bilinear')
    
    def forward(self,x):
        all_output = self.model(x)
        logits = all_output.logits
        out = self.up(logits)
        return out

class SegformerHF_scratch(nn.Module):
    def __init__(
        self,
        channels = 3,
        num_classes=2
    ):
        super().__init__()
        configuration = SegformerConfig() 
        self.model = SegformerForSemanticSegmentation(configuration)
        self.up = nn.Upsample(scale_factor=4,mode='bilinear')
    
    def forward(self,x):
        all_output = self.model(x)
        logits = all_output.logits
        out = self.up(logits)
        return out

if __name__=='__main__':
    model = SegformerHF_pretrained(num_classes=2)
    summary(model,(1,3,512,512))