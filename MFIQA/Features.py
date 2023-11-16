from torch.utils.data import Dataset
from dataset.IAA_dataset import IAADataset
from dataset.MAN_dataset import MANDataset
from dataset.SamDataset import SamDataset_mod
import pandas as pd



class FeatureDataset(Dataset):
    def __init__(self, dataset='ImageReward', type='train'):
        """
        初始化数据集。
        Args:
            dataset1, dataset2, dataset3 (Dataset): 三个数据集实例。
        """
        self.dataset = dataset
        if self.dataset == 'ImageReward':
            self.dataset1 = IAADataset('ImageReward', type)
            self.dataset2 = MANDataset('ImageReward', type)
            self.dataset3 = SamDataset_mod('ImageReward', type)
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
            self.dataset1 = IAADataset('LAION', type)
            self.dataset2 = MANDataset('LAION', type)
            self.dataset3 = SamDataset_mod('LAION', type)
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
            self.dataset1 = IAADataset('AGIQA-3k', type)
            self.dataset2 = MANDataset('AGIQA-3k', type)
            self.dataset3 = SamDataset_mod('AGIQA-3k', type)
            self.root = '/data/wangpuyi_data/AGIQA-3K'
            if type == 'train':
                df = pd.read_csv('data/AGIQA-3k/train.csv')
            elif type == 'validation':
                df = pd.read_csv('data/AGIQA-3k/validation.csv')
            elif type == 'test':
                df = pd.read_csv('data/AGIQA-3k/test.csv') 
            elif type == 'data':
                df = pd.read_csv('data/AGIQA-3k/data.csv')
            self.paths = df['name'].tolist()
            self.labels = df['mos_quality'].values.tolist()      
        
        if self.dataset == 'test': #取100张
            self.dataset1 = IAADataset('AGIQA-3k', type)
            self.dataset2 = MANDataset('AGIQA-3k', type)
            self.dataset3 = SamDataset_mod('AGIQA-3k', type)
            self.root = '/data/wangpuyi_data/AGIQA-3K'
            if type == 'train':
                df = pd.read_csv('data/AGIQA-3k/train.csv')
            elif type == 'validation':
                df = pd.read_csv('data/AGIQA-3k/validation.csv')
            elif type == 'test':
                df = pd.read_csv('data/AGIQA-3k/test.csv')
            self.paths = df['name'].tolist()[:100]
            self.labels = df['mos_quality'].values.tolist()[:100]
        
        # 确保三个数据集的大小相同
        assert len(self.dataset1) == len(self.dataset2) == len(self.dataset3), "Datasets do not have the same size"

    def __len__(self):
        """
        返回数据集中的样本数。
        """
        return len(self.labels)
    
    def __getitem__(self, index):
        """
        根据指定的索引返回来自三个数据集的样本及其标签的组合。
        Args:
            index (int): 数据样本的索引值。
        """
        data1 = self.dataset1[index]
        data2 = self.dataset2[index]
        data3 = self.dataset3[index]
        
        # 每个dataset返回处理过 适应自身对应module的数据
        combined_data = data1, data2, data3 # 每个数据集返回的是data
        label = self.labels[index]
        
        return combined_data, label

# # 示例使用：
# # 假设我们已经有了三个数据集实例
# dataset1 = ...
# dataset2 = ...
# dataset3 = ...

# # 创建组合数据集实例
# combined_dataset = MFIQADataset(dataset1, dataset2, dataset3)

# # 现在可以在DataLoader中使用这个combined_dataset来创建一个可迭代的数据加载器
# from torch.utils.data import DataLoader

# dataloader = DataLoader(combined_dataset, batch_size=4, shuffle=True)

# # 在训练循环中使用dataloader
# for batch in dataloader:
#     batch_data, batch_labels = batch
#     data1, data2, data3 = batch_data
#     label1, label2, label3 = batch_labels
#     # 执行你的训练步骤...
