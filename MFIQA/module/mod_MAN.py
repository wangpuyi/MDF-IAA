import torch
import torch.nn as nn
import torch.nn.functional as F
from .maniqa import MANIQA
import torch.nn.init as init


class mod_MAN(nn.Module):
    def __init__(self, num_features=1024):
        super(mod_MAN, self).__init__()
        self.man_iqa = MANIQA() # 修改后的模型 forward直接返回fc_score之前的特征

        # Freeze the parameters of the man_iqa model
        for param in self.man_iqa.parameters():
            param.requires_grad = False
            
        self.in_feature_size = self.man_iqa.fc_score[0].in_features
        self.fc1 = nn.Linear(self.in_feature_size, self.in_feature_size)
        self.fc2 = nn.Linear(self.in_feature_size, num_features)
        self.norm = nn.LayerNorm(num_features)
        self.activation = nn.GELU()
        # Initialize the weights using Kaiming initialization
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        # 加载预训练的权重，如果有的话
        self.load_pretrained()

    def load_pretrained(self):
        device = torch.device("cuda")
        # 这里需要定义如何加载预训练的权重，可能会涉及到状态字典的更新
        checkpoint = torch.load('/home/wangpuyi/MFIQA/checkpoints/ckpt_koniq10k.pt', map_location= device)
        self.man_iqa.load_state_dict(checkpoint, strict=False)
        
    def forward(self, x):
        # 提取 fc_score 之前的特征
        features = self.man_iqa.forward(x)
        
        # # 将这些特征通过新的全连接层
        # features = self.fc1(features)
        # features = self.fc2(features)
        
        # # 应用 Layer Normalization 和 GELU 激活
        # features = self.norm(features)
        # features = self.activation(features)
        
        return features
