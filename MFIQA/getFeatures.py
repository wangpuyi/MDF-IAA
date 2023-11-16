from Features import FeatureDataset
# 现在可以在DataLoader中使用这个combined_dataset来创建一个可迭代的数据加载器
from torch.utils.data import DataLoader
from FeatureModule import FeatureModule
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
from tqdm import tqdm

dataset_ = 'ImageReward'
type_ = 'validation'

if dataset_ == 'ImageReward':
    features_dir = '/data/wangpuyi_data/MFIQA/ImageReward/'
elif dataset_ == 'LAION':
    features_dir = '/data/wangpuyi_data/MFIQA/LAION/'
elif dataset_ == 'AGIQA-3k':
    features_dir = '/data/wangpuyi_data/MFIQA/AGIQA-3k/'
elif dataset_ == 'test':
    features_dir = '/data/wangpuyi_data/MFIQA/test/'

if not os.path.exists(features_dir):
    os.makedirs(features_dir)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try_dataset = FeatureDataset(dataset_, type_)
try_module = FeatureModule().to(device)
# try_dataset = DataLoader(try_dataset, batch_size=1, shuffle=False)

# 在训练循环中使用dataloader
with torch.no_grad():
    for index, data in tqdm(enumerate(try_dataset)):
        combined_data, label = data 
        data1, data2, data3 = combined_data
        data1 = data1.unsqueeze(0).to(device)# data1:  torch.Size([3, 224, 224])
        data2 = data2.unsqueeze(0).to(device)# data2:  torch.Size([3, 224, 224])
        data3 = data3.unsqueeze(0).to(device)# data3:  torch.Size([3, 1024, 1024])
        
        # print('data1: ', data1.shape, 'data2: ', data2.shape, 'data3: ', data3.shape)
        features = try_module((data1, data2, data3))
        fea1, fea2, fea3 = features
        fea1_cpu = fea1.to('cpu')
        fea2_cpu = fea2.to('cpu')
        fea3_cpu = fea3.to('cpu')
        # print("feature1: ", fea1.shape, "feature2: ", fea2.shape, "feature3: ", fea3.shape)
        # feature1:  torch.Size([1, 1024]) feature2:  torch.Size([1, 1568]) feature3:  torch.Size([256])
        saved = {'IAA': fea1, 'MAN': fea2, 'SAM': fea3}

        if dataset_ == 'AGIQA-3k':
            torch.save(saved, features_dir + str(index) + '_SAM_MAN_IAA_' + '.pth')
        else:
            torch.save(saved, features_dir + str(index) + '_SAM_MAN_IAA_' + type_ + '.pth') 
