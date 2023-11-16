import os
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)

class IAADataset(Dataset):
    #如果要把三个模型统一起来，需要一个取图片，取分数的类； 在类中调用三个模型， 返回三个模型需要的输入
    def __init__(self, dataset='ImageReward', type='train'):
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
        
        self.transform = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            normalize])

    def __len__(self):
        return len(self.labels)

    #可以修改为输入图片路径，返回图片的tensor
    def __getitem__(self, item):
        img_path = os.path.join(self.root, self.paths[item])
        image = default_loader(img_path)
        image = image.resize((224, 224))
        x = self.transform(image)

        return x