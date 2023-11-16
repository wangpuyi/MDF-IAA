import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
from torchvision import transforms

class MANDataset(Dataset):
    def __init__(self, dataset='ImageReward', type='train'):
        """
        初始化数据集。
        Args:
            data (list, ndarray, DataFrame): 包含特征的数据。
            labels (list, ndarray): 包含标签的数据。
        """
        self.dataset = dataset
        if self.dataset == 'ImageReward':
            self.root = '/data/wangpuyi_data/ImageRewardDB'
            if type == 'train':
                df = pd.read_csv('data/ImageReward/train.csv')
            elif type == 'validation':
                df = pd.read_csv('data/ImageReward/validation.csv')
            elif type == 'test':
                df = pd.read_csv('data/ImageReward/test.csv')
            self.paths = df['path'].tolist()
            self.labels = df['fidelity'].values.tolist()

        if self.dataset == 'LAION':
            self.root = '/data/wangpuyi_data/home/jdp/simulacra-aesthetic-captions'
            if type == 'train':
                df = pd.read_csv('data/LAION/mytrain.csv')
            elif type == 'validation':
                df = pd.read_csv('data/LAION/myvalidation.csv')
            elif type == 'test':
                df = pd.read_csv('data/LAION/mytest.csv')
            self.paths = df['path'].tolist()
            self.labels = df['rating'].values.tolist()

        if self.dataset == 'AGIQA-3k':
            self.root = '/data/wangpuyi_data/AGIQA-3K'
            if type == 'train':
                df = pd.read_csv('data/AGIQA-3k/train.csv')
            elif type == 'validation':
                df = pd.read_csv('data/AGIQA-3k/validation.csv')
            elif type == 'test':
                df = pd.read_csv('data/AGIQA-3k/test.csv') 
            self.paths = df['name'].tolist()
            self.labels = df['mos_quality'].values.tolist()   

        if self.dataset == 'test': #取100张
            self.root = '/data/wangpuyi_data/AGIQA-3K'
            if type == 'train':
                df = pd.read_csv('data/AGIQA-3k/train.csv')
            elif type == 'validation':
                df = pd.read_csv('data/AGIQA-3k/validation.csv')
            elif type == 'test':
                df = pd.read_csv('data/AGIQA-3k/test.csv')
            self.paths = df['name'].tolist()[:100]
            self.labels = df['mos_quality'].values.tolist()[:100]    

        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    def __len__(self):
        """
        返回数据集中的样本数。
        """
        return len(self.labels)
    
    def __getitem__(self, index):
        """
        根据指定的索引返回一个样本及其标签。
        Args:
            index (int): 数据样本的索引值。
        """
        image_path = os.path.join(self.root, self.paths[index])
        label = self.labels[index]

        img = Image.open(image_path).convert('RGB')
        img = self.transforms(img)


        return img

# # 使用示例
# # 假设我们有一些数据和标签
# data = ... # 这里填充你的数据，例如Numpy数组或其他
# labels = ... # 这里填充你的标签数据，例如Numpy数组或其他

# # 创建数据集实例
# dataset = CustomDataset(data=data, labels=labels)

# # 现在可以在DataLoader中使用这个dataset来创建一个可迭代的数据加载器
# from torch.utils.data import DataLoader

# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# # 然后可以在训练循环中使用dataloader
# for batch in dataloader:
#     batch_data, batch_labels = batch
#     # 执行你的训练步骤...
