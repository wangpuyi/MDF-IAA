from torchvision import transforms
from torch.utils.data import Dataset
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

from copy import deepcopy
from typing import Tuple


IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)

class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length #1024

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

import pandas as pd
import os

class SamDataset_mod(Dataset):
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

    def __len__(self):
        # return self.df.shape[0]
        return len(self.labels)
    
    #可以修改为输入图片路径，返回图片的tensor
    def __getitem__(self, item):
        img_path = os.path.join(self.root, self.paths[item])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        transform = ResizeLongestSide(1024)
        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image)
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        return transformed_image.squeeze() #[16, 3, 1024, 1024]