import os
import numpy as np
import cv2
from tqdm import tqdm
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import nibabel as nib
import pydicom
import cv2
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score


import nibabel as nib
import pydicom
import cv2
import os
import numpy as np

import cv2
import os
import numpy as np
from tqdm import tqdm

def crop_tumor_region(ct_img, tumor_mask, output_shape=(48, 48)):
    """
    裁切肿瘤区域并将非肿瘤区域置黑，同时返回裁切索引信息
    :param ct_img: CT图像，形状为 (H, W, Z) = (512, 512, N)
    :param tumor_mask: 肿瘤掩码，形状同上，1表示肿瘤，0表示非肿瘤
    :param output_shape: 输出图像的(H, W) = (48, 48)
    :return: 
        cropped_img: 处理后的图像，形状为 (output_H, output_W, M)，M≤N
        tumor_crop_mask: 对应的掩码
        crop_info: 包含裁切索引信息的字典
    """
    print("ct_img shape:", ct_img.shape)
    print("tumor_mask shape:", tumor_mask.shape)
    # 确认输入维度顺序正确 (H, W, Z)
    if len(ct_img.shape) != 3:
        raise ValueError(f"输入图像必须是3维 (H, W, Z)，但得到 {ct_img.shape}")
    
    # 找到肿瘤区域的坐标（Y=高度, X=宽度, Z=深度）
    y_coords, x_coords, z_coords = np.where(tumor_mask == 1)
    # x_coords, y_coords, z_coords = np.where(tumor_mask == 1)

    
    if len(y_coords) == 0:
        # 无肿瘤区域，返回全黑图像
        empty_img = np.zeros((output_shape[0], output_shape[1], ct_img.shape[2]), dtype=np.float32)
        empty_mask = np.zeros((output_shape[0], output_shape[1], ct_img.shape[2]), dtype=np.int32)
        crop_info = {
            'has_tumor': False,
            'original_shape': ct_img.shape,
            'output_shape': (output_shape[0], output_shape[1], ct_img.shape[2]),
            'z_slices': list(range(ct_img.shape[2])),
            'bounding_box': None,
            'padding': {'y': 0, 'x': 0}
        }
        return empty_img, empty_mask, crop_info
    
    # 确定肿瘤区域在各维度的边界
    y_min, y_max = np.min(y_coords), np.max(y_coords)  # 高度方向
    x_min, x_max = np.min(x_coords), np.max(x_coords)  # 宽度方向
    z_min, z_max = np.min(z_coords), np.max(z_coords)  # 深度方向（Z轴切片）
    
    # 计算肿瘤中心坐标
    y_center = (y_min + y_max) // 2
    x_center = (x_min + x_max) // 2
    
    # 计算裁切半尺寸（输出尺寸的一半）
    y_half = output_shape[0] // 2  # 高度方向半长
    x_half = output_shape[1] // 2  # 宽度方向半长
    
    # 确定裁切范围（确保在图像边界内）
    # 高度方向 (Y)
    y_start = max(0, y_center - y_half)
    y_end = min(ct_img.shape[0], y_start + output_shape[0])
    
    # 宽度方向 (X)
    x_start = max(0, x_center - x_half)
    x_end = min(ct_img.shape[1], x_start + output_shape[1])
    
    # 只保留包含肿瘤的Z轴切片（减少无效切片）
    z_start = z_min
    z_end = z_max + 1  # 切片索引是左闭右开
    z_slices = list(range(z_start, z_end))  # 保留的Z轴切片索引
    
    # 裁切肿瘤区域 (H, W, Z)
    cropped_img = ct_img[y_start:y_end, x_start:x_end, z_start:z_end]
    
    # 填充不足尺寸为目标大小（用黑色填充）
    pad_y = max(0, output_shape[0] - cropped_img.shape[0])
    pad_x = max(0, output_shape[1] - cropped_img.shape[1])
    
    cropped_img = np.pad(
        cropped_img,
        (
            (pad_y // 2, pad_y - pad_y // 2),  # Y轴填充
            (pad_x // 2, pad_x - pad_x // 2),  # X轴填充
            (0, 0)  # Z轴不填充（只保留有肿瘤的切片）
        ),
        mode='constant',
        constant_values=0.0
    )
    
    # 裁切对应的掩码并填充
    tumor_crop_mask = tumor_mask[y_start:y_end, x_start:x_end, z_start:z_end]
    tumor_crop_mask = np.pad(
        tumor_crop_mask,
        (
            (pad_y // 2, pad_y - pad_y // 2),
            (pad_x // 2, pad_x - pad_x // 2),
            (0, 0)
        ),
        mode='constant',
        constant_values=0
    )
    
    # 将非肿瘤区域置为黑色
    # cropped_img[np.where(tumor_crop_mask == 0)] = 0.0
    
    # 构建裁切信息字典
    crop_info = {
        'has_tumor': True,
        'original_shape': tuple(int(d) for d in ct_img.shape),
        'output_shape': tuple(int(d) for d in cropped_img.shape),
        'bounding_box': {
            'y_range': (int(y_start), int(y_end)),
            'x_range': (int(x_start), int(x_end)),
            'z_range': (int(z_start), int(z_end)),
            'tumor_center': (int(y_center), int(x_center)),
            'tumor_bbox': (int(y_min), int(y_max), int(x_min), int(x_max), int(z_min), int(z_max))
        },
        'z_slices': [int(slice_idx) for slice_idx in z_slices],  # 保留的Z轴切片索引列表
        'padding': {
            'y': int(pad_y),
            'x': int(pad_x)
        },
        'tumor_volume': int(len(y_coords))  # 肿瘤体素数量
    }


    return cropped_img, tumor_crop_mask, crop_info



# 读取 DICOM 图像序列
def read_dcm_sequence(folder_path):
    dcm_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".dcm")])
    print(f"Found {len(dcm_files)} DICOM files.")
    images = []
    for file in dcm_files:
        dcm_data = pydicom.dcmread(file)
        try:
            pixel_data = dcm_data.pixel_array.astype('>u2')
            # pixel_data = np.zeros((512, 512)).astype('>u2')
            # print(pixel_data.min(), pixel_data.max(), pixel_data.dtype, pixel_data.shape)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            pixel_data = np.zeros((512, 512)).astype('>u2')
                    
        images.append(pixel_data)
    return images

class MedicalDataLoader:
    def __init__(self, patient_folder, mask_folder, max_patients=20):
        """
        Initialize data loader
        :param patient_folder: Path to folder containing patient DICOM images
        :param mask_folder: Path to folder containing patient mask files
        :param max_patients: Limit on number of patients to load
        """
        self.patient_folder = patient_folder
        self.mask_folder = mask_folder
        self.max_patients = max_patients

    def load_patient_data(self, patient_name):
        """
        Load DICOM image sequence and mask for a single patient
        :param patient_name: Patient name (folder name)
        :return: Image sequence (list of numpy arrays), mask (numpy array)
        """
        # Load DICOM image sequence
        patient_path = os.path.join(self.patient_folder, patient_name)

        images = read_dcm_sequence(patient_path)

        # Load mask file
        mask_file = self.find_mask_file(patient_name)
        if mask_file is None:
            raise FileNotFoundError(f"No mask file found for patient: {patient_name}")

        # 读取 .nii 文件作为掩码
        nii_img = nib.load(mask_file)
        mask_data = nii_img.get_fdata().astype(np.uint8)  # 确保掩码为整数类型

        ### TODO: 预处理掩码，不确定是否需要进行以下步骤
        # mask_data = np.flipud(mask_data) ### TODO: 上下翻转
        # mask_data = np.fliplr(mask_data) ### TODO: 左右翻转
        # mask_data = mask_data[:,:,::-1] ### TODO: 前后翻转

        return images, mask_data

    def find_mask_file(self, patient_name):
        """
        Find mask file matching patient name
        :param patient_name: Patient name
        :return: Full path to mask file (str) or None
        """
        for file in os.listdir(self.mask_folder):
            if patient_name in file and file.endswith(".nii"):
                return os.path.join(self.mask_folder, file)
        return None

    def load_all_data(self):
        """
        Load data for all patients
        :return: Dictionary with patient names as keys and (image sequence, mask) as values
        """
        data = {}
        patient_names = [name for name in os.listdir(self.patient_folder) if os.path.isdir(os.path.join(self.patient_folder, name))]
        patient_names = patient_names[:self.max_patients]  # Limit number of patients loaded
        patient_names.sort()  # Ensure consistent order
        
        print(patient_names)
        for patient_id, patient_name in enumerate(patient_names):
            if patient_name in ["heweifu", "lvrongxi", "wangshuihua", "zanghuaixiang", "zhuqingyu"]:
                continue
            patient_name = patient_name.lower()
            print(f"{patient_id} Loading data for patient: {patient_name}")
            images, mask = self.load_patient_data(patient_name)
            data[patient_name] = (images, mask)
        return data


# Inherit and extend the original MedicalDataLoader
class EnhancedMedicalDataLoader(MedicalDataLoader):
    def __init__(self, patient_folder, mask_folder, max_patients=None):
        super().__init__(patient_folder, mask_folder, max_patients)
        self.data_cache = None

    def load_patient_data(self, patient_name):
        """Override loading method to add preprocessing steps"""
        images, mask = super().load_patient_data(patient_name)
        return images, mask

    def get_patient_list(self):
        """Get list of patients"""
        if max_patients is None:
            return [name for name in os.listdir(self.patient_folder)
                if os.path.isdir(os.path.join(self.patient_folder, name))]
        return [name for name in os.listdir(self.patient_folder)
                if os.path.isdir(os.path.join(self.patient_folder, name))][:self.max_patients]

    def load_and_cache_data(self):
        """Load and cache all data to avoid repeated loading"""
        if self.data_cache is None:
            self.data_cache = self.load_all_data()
        return self.data_cache



# 将掩码应用到图像上
def save_full_images(images, mask, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # 创建输出文件夹

    for i, img in enumerate(images):
        # 将 DICOM 图像转换为 RGB 格式
        # print("img")
        # print(img.min(), img.max(), img.dtype)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 获取当前切片的掩码
        mask_slice = mask[:, :, i]
        mask_num = np.sum(mask_slice == 1)
        if mask_num > 0:
            # print(f"Processing slice {i}, mask pixel count: {mask_num}")
            # 将掩码部分标记为红色
            # img_rgb[mask_slice == 1] = [0, 0, 255]  # 红色 (BGR 格式)

            # 保存标记后的图像
            output_path = os.path.join(output_folder, f"image_{i:03d}.png")
            # print("rgb")
            # print(img_rgb.min(), img_rgb.max(), img_rgb.dtype)
            cv2.imwrite(output_path, img_rgb)  # 确保数据类型为 uint8
            # print(f"Saved: {output_path}")

def apply_mask_on_images(images, mask, images_folder, result_folder):
    os.makedirs(result_folder, exist_ok=True)  # 创建输出文件夹

    for i, img in enumerate(images):
        # 获取当前切片的掩码
        mask_slice = mask[:, :, i]
        mask_num = np.sum(mask_slice == 1)
        if mask_num > 0:
            img_path = os.path.join(images_folder, f"image_{i:03d}.png")
            full_img = plt.imread(img_path)
            original_img = full_img.copy()
            full_img[mask_slice == 1] = [1,0,0]
            concatenated_image = np.hstack((original_img, full_img))
            new_img_path = os.path.join(result_folder, f"image_{i:03d}.png")
            plt.imsave(new_img_path, concatenated_image)


            

def save_mask_slices_cv2(mask, save_dir, z_slices):
    """
    使用OpenCV保存mask切片（文件更小，速度更快）
    :param mask: 3D mask数组
    :param save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    num_slices = mask.shape[2]
    
    for slice_idx in range(num_slices):
        z_slice = z_slices[slice_idx]
        # 获取当前切片
        slice_data = mask[:, :, slice_idx]
        
        # 转换为8位图像 (0 和 255)
        slice_uint8 = (slice_data * 255).astype(np.uint8)
        
        # 保存图片
        save_path = os.path.join(save_dir, f'mask_slice_{z_slice:03d}.png')
        cv2.imwrite(save_path, slice_uint8)
        
        print(f'已保存: {save_path}')
    
    print(f'总共保存了 {num_slices} 个切片')

def save_cropped_img_plt(result, save_dir, z_slices):
    """
    使用OpenCV保存mask切片（文件更小，速度更快）
    :param mask: 3D mask数组
    :param save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    assert len(z_slices) == result.shape[2], "z_slices长度必须与mask的第三维度匹配"
    num_slices = result.shape[2]
    
    for slice_idx in range(num_slices):
        z_slice = z_slices[slice_idx]
        # 获取当前切片
        slice_data = result[:, :, slice_idx]
        
        # 保存图片
        save_path = os.path.join(save_dir, f'img_slice_{z_slice:03d}.png')
        plt.imsave(save_path, slice_data, cmap='gray')
        
        print(f'已保存: {save_path}')
    
    print(f'总共保存了 {num_slices} 个切片')

def concatenate_and_save_images(img1, img2, z_slices, save_path, convert_img2_to_rgba=False):
    """
    水平拼接两张尺寸一样的图并保存
    :param img1: 第一张图像
    :param img2: 第二张图像
    :param save_path: 保存路径
    """
    # # 确保两张图片的尺寸相同
    # print(f"img1 shape: {img1.shape}, img2 shape: {img2.shape}")
    # 将灰度图像转换为RGBA图像
    if convert_img2_to_rgba:
        img2_rgba = np.stack((img2,) * 3, axis=-1)  # 复制灰度值到RGB通道
        img2_rgba = np.pad(img2_rgba, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=1)  # 添加alpha通道，值为255
        img2 = img2_rgba
        
    # 水平拼接图片
    concatenated_image = np.hstack((img1, img2))
    
    # 保存拼接后的图片
    plt.imsave(save_path, concatenated_image, cmap='gray')
    
    print(f'已保存拼接后的图片: {save_path}')

def concatenate_img_and_mask(img_folder, mask_folder, save_path, z_slices):
    """
    水平拼接图像和掩码并保存
    :param img: 图像
    :param mask: 掩码
    :param save_path: 保存路径
    """
    # 确保图像和掩码的尺寸相同
    os.makedirs(save_path, exist_ok=True)

    
    for slice_idx in range(len(z_slices)):
        z_slice = z_slices[slice_idx]
        img_save_path = os.path.join(img_folder, f'img_slice_{z_slice:03d}.png')
        mask_save_path = os.path.join(mask_folder, f'mask_slice_{z_slice:03d}.png')
        img = plt.imread(img_save_path)
        mask = plt.imread(mask_save_path)
        concatenated_save_path = os.path.join(save_path, f'concat_slice_{z_slice:03d}.png')
        concatenate_and_save_images(img, mask, z_slices, concatenated_save_path, convert_img2_to_rgba=True)


if __name__ == '__main__':
    # patient_folder = "/Users/yangzidong/Desktop/yuqing/medical/202510/MTC_3D/MTC patient"
    # mask_folder = "/Users/yangzidong/Desktop/yuqing/medical/202510/MTC_3D/tumor3d"

    patient_folder = "/Users/yangzidong/Desktop/yuqing/medical/202510/patient3"
    mask_folder = "/Users/yangzidong/Desktop/yuqing/medical/202510/patient3"
    # max_patients = 10
    dataloader = EnhancedMedicalDataLoader(patient_folder, mask_folder, max_patients=3)
    data_cache = dataloader.load_and_cache_data()


    patient_names = list(data_cache.keys())
    for patient_name in tqdm(patient_names, 
                            desc="处理患者数据", 
                            total=len(patient_names),
                            ncols=100):  # 进度条宽度
        # 保存原始大小的图像
        # try:
        if True:
            save_full_images(images = data_cache[patient_name][0], 
                                mask = data_cache[patient_name][1], 
                                output_folder = f"images_vis/{patient_name}")

            apply_mask_on_images(images = data_cache[patient_name][0], 
                                mask = data_cache[patient_name][1], 
                                images_folder= f"images_vis/{patient_name}", 
                                result_folder= f"images_mask_vis/{patient_name}")
            # 裁切肿瘤区域并将非肿瘤区域置黑
            result, tumor_crop_mask, cropped_info = crop_tumor_region(ct_img = np.array(data_cache[patient_name][0]).transpose((1,2,0)), 
                            tumor_mask = data_cache[patient_name][1], 
                            output_shape=(48, 48))

            # 保存 cropped mask （黑白）
            save_mask_slices_cv2(tumor_crop_mask, f"cropped_mask/{patient_name}", cropped_info['z_slices'])
            save_cropped_img_plt(result, f"cropped_img/{patient_name}", cropped_info['z_slices'])
            concatenate_img_and_mask(f"cropped_img/{patient_name}", f"cropped_mask/{patient_name}", f"concat_img_mask/{patient_name}", cropped_info['z_slices'])
            # 保存 cropped 图像至 npz
            os.makedirs('cropped_data', exist_ok=True)
            np.savez(f'cropped_data/{patient_name}.npz', data=result)
            print("cropped_info", cropped_info)
            import json
            os.makedirs('cropped_info', exist_ok=True)
            with open(f'cropped_info/{patient_name}_info.json', 'w') as f:
                json.dump(cropped_info, f, indent=4)
        # except Exception as e:
        #     print(f"Error load {patient_name}: {e}")


