import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# 自定义数据集类
class PatientDataset(Dataset):
    def __init__(self, patient_ids, labels, data_root, mask_root, transform=None):
        """
        Args:
            patient_ids: 病人ID列表
            labels: 对应的标签列表 (0或1)
            data_root: 影像数据根目录
            mask_root: mask数据根目录
            transform: 数据增强
        """
        self.patient_ids = patient_ids
        self.labels = labels
        self.data_root = data_root
        self.mask_root = mask_root
        self.transform = transform

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        label = self.labels[idx]

        # 加载影像数据
        data_path = os.path.join(self.data_root, f"{patient_id}.npz")
        cropped_data = np.load(data_path)
        image_data = cropped_data["data"]  # shape: (32, 32, slices)

        # 加载mask数据
        mask_patient_root = os.path.join(self.mask_root, patient_id)
        mask_paths = os.listdir(mask_patient_root)
        mask_paths.sort()

        masks = []
        for mask_path in mask_paths:
            mask_slice = Image.open(os.path.join(mask_patient_root, mask_path))
            mask_slice = np.array(mask_slice)
            masks.append(mask_slice)

        masks = np.array(masks).transpose(1, 2, 0)  # shape: (32, 32, slices)

        # 数据预处理
        # 归一化影像数据 (0-65535 -> 0-1)
        image_data = image_data.astype(np.float32) / 65535.0

        # 二值化mask (0-255 -> 0-1)
        masks = (masks > 128).astype(np.float32)

        # 转换为PyTorch tensor格式 (C, H, W, D)
        image_data = torch.from_numpy(image_data).permute(2, 0, 1).unsqueeze(0)  # (1, slices, 32, 32)
        masks = torch.from_numpy(masks).permute(2, 0, 1).unsqueeze(0)  # (1, slices, 32, 32)

        # 应用数据增强
        if self.transform:
            # 这里可以添加3D数据增强
            pass

        return {
            'image': image_data,  # (1, slices, 32, 32)
            'mask': masks,        # (1, slices, 32, 32)
            'label': torch.tensor(label, dtype=torch.long),
            'patient_id': patient_id
        }

class DynamicSliceDataset(Dataset):
    def __init__(self, patient_ids, labels, data_root, mask_root, transform=None, target_slices=None):
        self.patient_ids = patient_ids
        self.labels = labels
        self.data_root = data_root
        self.mask_root = mask_root
        self.transform = transform
        self.target_slices = target_slices  # 目标切片数，None表示保持原样

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        label = self.labels[idx]

        # 加载数据
        data_path = os.path.join(self.data_root, f"{patient_id}.npz")
        cropped_data = np.load(data_path)
        image_data = cropped_data["data"]  # (H, W, slices)

        # 加载mask
        mask_patient_root = os.path.join(self.mask_root, patient_id)
        mask_paths = os.listdir(mask_patient_root)
        mask_paths.sort()

        masks = []
        for mask_path in mask_paths:
            mask_slice = Image.open(os.path.join(mask_patient_root, mask_path))
            mask_slice = np.array(mask_slice)
            masks.append(mask_slice)

        masks = np.array(masks).transpose(1, 2, 0)  # (H, W, slices)

        # 处理可变切片数量
        if self.target_slices is not None:
            image_data, masks = self._adjust_slices(image_data, masks, self.target_slices)

        # 数据预处理
        image_data = image_data.astype(np.float32) / 65535.0
        masks = (masks > 128).astype(np.float32)

        # 转换为PyTorch tensor
        image_data = torch.from_numpy(image_data).permute(2, 0, 1).unsqueeze(0)  # (1, slices, H, W)
        masks = torch.from_numpy(masks).permute(2, 0, 1).unsqueeze(0)  # (1, slices, H, W)

        return {
            'image': image_data,
            'mask': masks,
            'label': torch.tensor(label, dtype=torch.long),
            'patient_id': patient_id,
            'num_slices': image_data.shape[1]  # 返回实际切片数
        }

    def _adjust_slices(self, image_data, masks, target_slices):
        """调整切片数量到目标值"""
        current_slices = image_data.shape[2]

        if current_slices == target_slices:
            return image_data, masks

        if current_slices < target_slices:
            # 切片不足，进行填充
            pad_depth = target_slices - current_slices
            # 在深度维度进行对称填充
            pad_before = pad_depth // 2
            pad_after = pad_depth - pad_before

            image_data = np.pad(image_data, ((0,0), (0,0), (pad_before, pad_after)),
                              mode='constant', constant_values=0)
            masks = np.pad(masks, ((0,0), (0,0), (pad_before, pad_after)),
                         mode='constant', constant_values=0)
        else:
            # 切片过多，进行中心裁剪
            start = (current_slices - target_slices) // 2
            end = start + target_slices
            image_data = image_data[:, :, start:end]
            masks = masks[:, :, start:end]

        return image_data, masks

# 统计所有病人的切片数量，找到合适的target_slices
def analyze_slice_distribution(patient_ids, data_root):
    slice_counts = []
    for patient_id in patient_ids:
        data_path = os.path.join(data_root, f"{patient_id}.npz")
        if os.path.exists(data_path):
            cropped_data = np.load(data_path)
            slice_counts.append(cropped_data["data"].shape[2])

    print(f"切片数量统计:")
    print(f"最小值: {min(slice_counts)}")
    print(f"最大值: {max(slice_counts)}")
    print(f"平均值: {np.mean(slice_counts):.2f}")
    print(f"中位数: {np.median(slice_counts)}")

    # 可以选择中位数或众数作为目标切片数
    target_slices = int(np.median(slice_counts))
    print(f"推荐的目标切片数: {target_slices}")

    return target_slices, slice_counts
