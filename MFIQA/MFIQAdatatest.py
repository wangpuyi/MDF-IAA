from torch.utils.data import Dataset
import torch
import pandas as pd

class MFIQADataset(Dataset):
    def __init__(self, dataset='ImageReward', type='train'):
        self.dataset = dataset
        self.type = type
        if self.dataset == 'ImageReward':
            if type == 'train':
                df = pd.read_csv('data/ImageReward/train.csv')
            elif type == 'validation':
                df = pd.read_csv('data/ImageReward/validation.csv')
            elif type == 'test':
                df = pd.read_csv('data/ImageReward/test.csv')
            self.paths = df['path'].tolist()
            self.labels = df['fidelity'].values.tolist()

        if self.dataset == 'LAION':
            if type == 'train':
                df = pd.read_csv('data/LAION/mytrain.csv')
            elif type == 'validation':
                df = pd.read_csv('data/LAION/myvalidation.csv')
            elif type == 'test':
                df = pd.read_csv('data/LAION/mytest.csv')
            self.labels = df['rating'].values.tolist()

        if self.dataset == 'AGIQA-3k':
            if type == 'train':
                df = pd.read_csv('data/AGIQA-3k/train.csv')
            elif type == 'validation':
                df = pd.read_csv('data/AGIQA-3k/validation.csv')
            elif type == 'test':
                df = pd.read_csv('data/AGIQA-3k/test.csv') 
            self.labels = df['mos_quality'].values.tolist()      
        
        if self.dataset == 'test': #取100张
            if type == 'train':
                df = pd.read_csv('data/AGIQA-3k/train.csv')
            elif type == 'validation':
                df = pd.read_csv('data/AGIQA-3k/validation.csv')
            elif type == 'test':
                df = pd.read_csv('data/AGIQA-3k/test.csv')
            self.labels = df['mos_quality'].values.tolist()[:100]

    def __len__(self):
        """
        返回数据集中的样本数。
        """
        return len(self.labels)
    
    def __getitem__(self, item):
        label = self.labels[item]
        if self.dataset == 'ImageReward':
            features = torch.load('/data/wangpuyi_data/MFIQA/ImageReward/'+str(item)+'_SAM_MAN_IAA_' + self.type + '.pth')
        elif self.dataset == 'LAION':
            features = torch.load('/data/wangpuyi_data/MFIQA/LAION/'+str(item)+'_SAM_MAN_IAA_' + self.type + '.pth')
        elif self.dataset == 'AGIQA-3k':
            features = torch.load('/data/wangpuyi_data/MFIQA/AGIQA-3k/'+str(item)+'_SAM_MAN_IAA_' + '.pth')
        elif self.dataset == 'test':
            features = torch.load('/data/wangpuyi_data/MFIQA/test/'+str(item)+'_SAM_MAN_IAA_' + 'train' + '.pth')
        
        #使用三个特征直接相连
        #得到1*1024, 1*1568, 256的特征
        fea1 = features['IAA']
        fea2 = features['MAN']
        fea3 = features['SAM'].unsqueeze(0)
        # print("feature1: ", fea1.shape, "feature2: ", fea2.shape, "feature3: ", fea3.shape)
        tensor = torch.cat((fea1,fea2,fea3), 1)

        return tensor, label
        