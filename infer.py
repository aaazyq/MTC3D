import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import shap
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
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
from dataset import DynamicSliceDataset
from model import GlobalPool3DCNN
from train_utils import visualize_cam, visualize_shap

df = pd.read_csv('data/cleaned_data.csv')
df = df.rename(columns={'name': 'patient_id', "level": "label"})
# 分割训练集和验证集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# 数据路径
data_root = "data/cropped_data"  # 修改为您的数据路径
mask_root = "data/cropped_mask"  # 修改为您的mask路径

train_dataset = DynamicSliceDataset(
    patient_ids=train_df['patient_id'].tolist(),
        labels=train_df['label'].tolist(),
        data_root=data_root,
        mask_root=mask_root,
        transform=None,
        target_slices=10
)
val_dataset = DynamicSliceDataset(
    patient_ids=val_df['patient_id'].tolist(),
        labels=val_df['label'].tolist(),
        data_root=data_root,
        mask_root=mask_root,
        transform=None,
        target_slices=10
)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GlobalPool3DCNN(num_classes=2)
model = model.to(device)


# 加载最佳模型
best_model_path = 'output/best_model.pth'
if os.path.exists(best_model_path):
    checkpoint = torch.load(best_model_path, weights_only=False)
    print(checkpoint.keys())
    model.load_state_dict(checkpoint['model_state_dict'])
    print("已加载最佳模型进行可解释性分析")

# 生成CAM可视化
visualize_cam(model, val_loader, device, save_dir='cam_results')

# 生成SHAP可视化
visualize_shap(model, val_loader, device, save_dir='shap_results')