# 替换后的 dataset/dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import random

class MyDataset(Dataset):
    def __init__(self, data_path, mode='train'):
        """
        mode: 'train', 'val' 或 'test'
        """
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        
        # 严格读取官方划分列表
        list_file = os.path.join(data_path, 'list', f'{mode}.txt')
        with open(list_file, 'r') as f:
            self.name_list = [line.strip() for line in f.readlines()]
            
        # 铁律1：必须是 ImageNet 均值方差归一化
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
        # 铁律2：尺寸必须强制 256
        self.to_tensor = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        name = self.name_list[index]
        # 根据 TinyCD 的要求，读取 A, B 和 label 文件夹
        img_a = Image.open(os.path.join(self.data_path, 'A', name)).convert('RGB')
        img_b = Image.open(os.path.join(self.data_path, 'B', name)).convert('RGB')
        mask = Image.open(os.path.join(self.data_path, 'label', name)).convert('L') # 灰度图

        # 铁律3：自己手写严格的数据增强 (仅水平、垂直、旋转90度)
        if self.mode == 'train':
            if random.random() > 0.5:
                img_a = img_a.transpose(Image.FLIP_LEFT_RIGHT)
                img_b = img_b.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                img_a = img_a.transpose(Image.FLIP_TOP_BOTTOM)
                img_b = img_b.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() > 0.5:
                img_a = img_a.transpose(Image.ROTATE_90)
                img_b = img_b.transpose(Image.ROTATE_90)
                mask = mask.transpose(Image.ROTATE_90)

        # 转 Tensor & 归一化
        img_a = self.normalize(self.to_tensor(img_a))
        img_b = self.normalize(self.to_tensor(img_b))
        
        # Mask 处理：保证尺寸 256，且像素值为 0 或 1
        mask = T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST)(mask)
        mask = T.ToTensor()(mask)
        mask = (mask > 0).float() # 二值化保护

        # TinyCD 返回通常是一个字典或元组，这里保持元组格式以兼容
        return img_a, img_b, mask