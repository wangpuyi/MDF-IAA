import torch.nn as nn
import torch.nn as nn
import torch

from segment_anything import sam_model_registry


class Sam_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.setup()
    
    def setup(self):
        model_type = 'vit_h'
        checkpoint = 'checkpoints/sam_vit_h_4b8939.pth'
        self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    
    def forward(self, x):
        x = self.sam_model.preprocess(x) #torch.Size([1, 3, 1024, 1024])
        # print('module: ', x.size()) #torch.Size([16, 3, 1024, 1024]) 注意维度unsqueeze(0)
        feature_sam = torch.stack([self.sam_model.image_encoder(x[i].unsqueeze(0)) for i in range(x.size(0))]) #torch.Size([16, 1, 256, 64, 64])
        feature_sam = feature_sam.squeeze()#torch.Size([16, 256, 64, 64])
        m = nn.AdaptiveAvgPool2d((1, 1))
        feature_sam = m(feature_sam).squeeze()#torch.Size([16, 256])

        return feature_sam #torch.Size([16, 256])
