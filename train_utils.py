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

# 基础解释器类
class BaseInterpreter:
    @staticmethod
    def process_target_class(target_class, device):
        """处理目标类别，确保其为正确形状的tensor"""
        if isinstance(target_class, (int, float)):
            return torch.tensor([[target_class]], device=device)
        elif isinstance(target_class, torch.Tensor):
            if target_class.dim() == 0:
                return target_class.view(1, 1).to(device)
            elif target_class.dim() == 1:
                return target_class.unsqueeze(1).to(device)
            else:
                return target_class.to(device)
        else:
            raise ValueError(f"Unsupported target_class type: {type(target_class)}")

# 修复的SHAP解释器实现 - 简化版本
class SimpleSHAP(BaseInterpreter):
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def _process_target_class(self, target_class, device):
        """处理目标类别，确保其为正确形状的tensor"""
        if isinstance(target_class, (int, float)):
            return torch.tensor([[target_class]], device=device)
        elif isinstance(target_class, torch.Tensor):
            if target_class.dim() == 0:
                return target_class.view(1, 1).to(device)
            elif target_class.dim() == 1:
                return target_class.unsqueeze(1).to(device)
            else:
                return target_class.to(device)
        else:
            raise ValueError(f"Unsupported target_class type: {type(target_class)}")

    def generate_shap_values(self, input_tensor, target_class=None):
        """
        使用梯度方法近似SHAP值
        """
        self.model.eval()

        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1)

        # 启用梯度计算
        input_tensor = input_tensor.clone().requires_grad_(True)

        # 前向传播
        output = self.model(input_tensor)

        # 确保target_class是tensor并且形状正确
        if isinstance(target_class, (int, float)):
            target_class = torch.tensor([[target_class]], device=input_tensor.device)
        elif isinstance(target_class, torch.Tensor):
            if target_class.dim() == 0:
                target_class = target_class.view(1, 1)
            elif target_class.dim() == 1:
                target_class = target_class.unsqueeze(1)

        # Create a mask for the target class outputs
        mask = torch.zeros_like(output)
        mask.scatter_(1, target_class, 1.0)

        # Calculate gradients of the target class output with respect to the input
        self.model.zero_grad()
        output.backward(gradient=mask, retain_graph=True)        # Use absolute gradients as feature importance
        shap_values = input_tensor.grad.abs().cpu().detach().numpy()

        return shap_values

# 修复的CAM实现
class CAM(BaseInterpreter):
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = None
        self.backward_handle = None
        
    def _process_target_class(self, target_class, device):
        """处理目标类别，确保其为正确形状的tensor"""
        if isinstance(target_class, (int, float)):
            return torch.tensor([[target_class]], device=device)
        elif isinstance(target_class, torch.Tensor):
            if target_class.dim() == 0:
                return target_class.view(1, 1).to(device)
            elif target_class.dim() == 1:
                return target_class.unsqueeze(1).to(device)
            else:
                return target_class.to(device)
        else:
            raise ValueError(f"Unsupported target_class type: {type(target_class)}")

    def _register_hooks(self):
        """注册前向和后向钩子"""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # 移除之前的钩子（如果存在）
        self._remove_hooks()

        # 注册新钩子
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_backward_hook(backward_hook)

    def _remove_hooks(self):
        """移除钩子"""
        if self.forward_handle is not None:
            self.forward_handle.remove()
            self.forward_handle = None
        if self.backward_handle is not None:
            self.backward_handle.remove()
            self.backward_handle = None

    def generate_cam(self, input_tensor, target_class=None):
        """生成类激活图"""
        # 注册钩子
        self._register_hooks()

        try:
            self.model.eval()

            # 确保输入需要梯度
            input_tensor = input_tensor.clone().requires_grad_(True)

            # 前向传播
            output = self.model(input_tensor)

            if target_class is None:
                target_class = output.argmax(dim=1)

            # 确保target_class是tensor并且形状正确
            if isinstance(target_class, (int, float)):
                target_class = torch.tensor([[target_class]], device=input_tensor.device)
            elif isinstance(target_class, torch.Tensor):
                if target_class.dim() == 0:
                    target_class = target_class.view(1, 1)
                elif target_class.dim() == 1:
                    target_class = target_class.unsqueeze(1)

            # 创建one-hot编码
            one_hot = torch.zeros_like(output)
            one_hot.scatter_(1, target_class, 1.0)

            # 反向传播
            self.model.zero_grad()
            output.backward(gradient=one_hot, retain_graph=False)

            if self.gradients is None or self.activations is None:
                return None

            # 计算权重
            weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)

            # 计算CAM
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            cam = F.relu(cam)

            # 归一化
            cam = cam.squeeze().cpu().numpy()
            if cam.max() - cam.min() > 1e-8:
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            else:
                cam = np.zeros_like(cam)

            return cam

        finally:
            # 确保钩子被移除
            self._remove_hooks()

# 修复的Grad-CAM实现
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = None
        self.backward_handle = None
        
    def _process_target_class(self, target_class, device):
        """处理目标类别，确保其为正确形状的tensor"""
        if isinstance(target_class, (int, float)):
            return torch.tensor([[target_class]], device=device)
        elif isinstance(target_class, torch.Tensor):
            if target_class.dim() == 0:
                return target_class.view(1, 1).to(device)
            elif target_class.dim() == 1:
                return target_class.unsqueeze(1).to(device)
            else:
                return target_class.to(device)
        else:
            raise ValueError(f"Unsupported target_class type: {type(target_class)}")

    def _register_hooks(self):
        """注册前向和后向钩子"""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # 移除之前的钩子（如果存在）
        self._remove_hooks()

        # 注册新钩子
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_backward_hook(backward_hook)

    def _remove_hooks(self):
        """移除钩子"""
        if self.forward_handle is not None:
            self.forward_handle.remove()
            self.forward_handle = None
        if self.backward_handle is not None:
            self.backward_handle.remove()
            self.backward_handle = None

    def generate_heatmap(self, input_tensor, target_class=None):
        """生成热图"""
        # 注册钩子
        self._register_hooks()

        try:
            self.model.eval()
            input_tensor = input_tensor.clone().requires_grad_(True)
            output = self.model(input_tensor)

            if target_class is None:
                target_class = output.argmax(dim=1)

            target_class = output.argmax(dim=1)
            
            # 确保target_class是tensor并且形状正确
            if isinstance(target_class, (int, float)):
                target_class = torch.tensor([[target_class]], device=input_tensor.device)
            elif isinstance(target_class, torch.Tensor):
                if target_class.dim() == 0:
                    target_class = target_class.view(1, 1)
                elif target_class.dim() == 1:
                    target_class = target_class.unsqueeze(1)

            # 创建one-hot编码
            one_hot = torch.zeros_like(output)
            one_hot.scatter_(1, target_class, 1.0)

            self.model.zero_grad()
            output.backward(gradient=one_hot, retain_graph=False)

            if self.gradients is None or self.activations is None:
                return None

            weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)
            heatmap = torch.sum(weights * self.activations, dim=1, keepdim=True)
            heatmap = F.relu(heatmap)

            heatmap = heatmap.squeeze().cpu().detach().numpy()
            if heatmap.max() - heatmap.min() > 1e-8:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            else:
                heatmap = np.zeros_like(heatmap)

            return heatmap

        finally:
            # 确保钩子被移除
            self._remove_hooks()

# 修复的CAM和SHAP可视化函数
def visualize_cam(model, dataloader, device, save_dir='cam_results', num_samples=4):
    """
    生成CAM可视化结果
    """
    os.makedirs(save_dir, exist_ok=True)

    # 选择目标层 (通常是最后一个卷积层)
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d):
            target_layer = module
            print(f"使用目标层: {name}")
            break  # 只取第一个卷积层

    if target_layer is None:
        print("警告: 未找到合适的卷积层")
        return

    # 初始化CAM
    cam_generator = CAM(model, target_layer)

    # 处理样本
    model.eval()
    sample_count = 0

    print("生成CAM可视化...")
    for batch in tqdm(dataloader, desc="Processing samples"):
        if sample_count >= num_samples:
            break

        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        labels = batch['label'].to(device)
        patient_ids = batch['patient_id']

        # 模型预测
        with torch.no_grad():
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            probabilities = F.softmax(outputs, dim=1)

        for i in range(images.size(0)):
            if sample_count >= num_samples:
                break

            patient_id = patient_ids[i]
            true_label = labels[i].item()
            pred_label = predictions[i].item()
            prob_high_risk = probabilities[i, 1].item()

            # 处理单个图像
            single_image = images[i:i+1].clone()
            single_mask = masks[i:i+1].clone()

            # 生成CAM
            # try:
            if True:
                with torch.enable_grad():
                    cam = cam_generator.generate_cam(single_image, pred_label)
                    if cam is not None:
                        _plot_cam(
                            image=single_image.squeeze(0),  # 移除批次维度
                            mask=single_mask.squeeze(0),  # 使用原图作为mask
                            cam=cam,
                            patient_id=patient_id,
                            true_label=true_label,
                            pred_label=pred_label,
                            prob_high_risk=prob_high_risk,
                            save_dir=save_dir,
                            sample_idx=sample_count
                        )
            # except Exception as e:
            #     print(f"生成CAM时出错: {e}")

            sample_count += 1

def visualize_shap(model, dataloader, device, save_dir='shap_results', num_samples=4):
    """
    生成SHAP可视化结果
    """
    os.makedirs(save_dir, exist_ok=True)

    # 初始化SHAP
    shap_explainer = SimpleSHAP(model, device)

    # 处理样本
    model.eval()
    sample_count = 0

    print("生成SHAP可视化...")
    for batch in tqdm(dataloader, desc="Processing samples"):
        if sample_count >= num_samples:
            break

        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        labels = batch['label'].to(device)
        patient_ids = batch['patient_id']

        # 模型预测
        with torch.no_grad():
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            probabilities = F.softmax(outputs, dim=1)

        for i in range(images.size(0)):
            if sample_count >= num_samples:
                break

            patient_id = patient_ids[i]
            true_label = labels[i].item()
            pred_label = predictions[i].item()
            prob_high_risk = probabilities[i, 1].item()

            # 处理单个图像
            single_image = images[i:i+1].clone()
            single_mask = masks[i:i+1].clone()
            print(single_image.shape, single_mask.shape)

            # 生成SHAP值
            # try:
            if True:
                shap_values = shap_explainer.generate_shap_values(single_image, pred_label)
                if shap_values is not None:
                    _plot_shap(
                        image=single_image.squeeze(0),  # 移除批次维度
                        mask=single_mask.squeeze(0),  # 使用原图作为mask
                        shap_values=shap_values[0] if len(shap_values.shape) > 1 else shap_values,
                        patient_id=patient_id,
                        true_label=true_label,
                        pred_label=pred_label,
                        prob_high_risk=prob_high_risk,
                        save_dir=save_dir,
                        sample_idx=sample_count
                    )
            # except Exception as e:
            #     print(f"生成SHAP值时出错: {e}")

            sample_count += 1

# 修复的训练函数（包含AUC）
def train_model_with_metrics(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir='results'):
    """
    包含AUC指标和热图保存的训练函数 - 修复版本
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'heatmaps'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'auc_curves'), exist_ok=True)

    # 初始化Grad-CAM
    target_layer = None
    for module in model.modules():
        if isinstance(module, nn.Conv3d):
            target_layer = module
            break  # 只取第一个卷积层

    if target_layer is None:
        print("警告: 未找到合适的卷积层用于Grad-CAM")
        grad_cam = None
    else:
        print(f"使用层进行Grad-CAM")
        grad_cam = GradCAM(model, target_layer)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_aucs = []
    val_aucs = []
    all_predictions = []

    best_val_auc = 0.0
    best_model_weights = None

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 50)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        train_outputs = []
        train_targets = []

        train_pbar = tqdm(train_loader, desc='Training')
        for batch in train_pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # 计算准确率
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # 收集预测概率用于AUC计算
            probabilities = F.softmax(outputs, dim=1)
            train_outputs.extend(probabilities[:, 1].detach().cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{torch.sum(preds == labels.data).item() / len(labels):.4f}'
            })

        # 计算训练AUC
        try:
            train_auc = roc_auc_score(train_targets, train_outputs)
        except ValueError:
            train_auc = 0.5  # 如果只有一类，设为0.5

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.cpu().numpy())
        train_aucs.append(train_auc)

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_outputs = []
        val_targets = []
        all_val_predictions = []
        all_val_labels = []

        # 保存一些验证样本的热图
        heatmap_saved = False

        val_pbar = tqdm(val_loader, desc='Validation')
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                patient_ids = batch['patient_id']

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * images.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

                # 收集预测概率用于AUC计算
                probabilities = F.softmax(outputs, dim=1)
                val_outputs.extend(probabilities[:, 1].detach().cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                all_val_predictions.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

                # 为第一个batch生成和保存热图
                if not heatmap_saved and batch_idx == 0 and grad_cam is not None:
                    try:
                        # 确保模型在评估模式
                        model.eval()
                        with torch.enable_grad():
                            save_heatmaps(grad_cam, images, labels, patient_ids, epoch, save_dir)
                        heatmap_saved = True
                    except Exception as e:
                        print(f"生成热图时出错: {e}")
                        heatmap_saved = True  # 标记为已保存，避免重复尝试

                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{torch.sum(preds == labels.data).item() / len(labels):.4f}'
                })

        # 计算验证AUC
        try:
            val_auc = roc_auc_score(val_targets, val_outputs)
        except ValueError:
            val_auc = 0.5

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)

        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc.cpu().numpy())
        val_aucs.append(val_auc)

        # 保存每个epoch的预测结果
        epoch_predictions = {
            'epoch': epoch,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'val_predictions': all_val_predictions,
            'val_labels': all_val_labels,
            'val_probabilities': val_outputs
        }
        all_predictions.append(epoch_predictions)

        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Train AUC: {train_auc:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}, Val AUC: {val_auc:.4f}')

        # 保存AUC曲线
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            _plot_auc_curves(val_outputs, val_targets, epoch, save_dir)
            _plot_confusion_matrix(all_val_predictions, all_val_labels, epoch, save_dir)

        # 保存最佳模型（基于AUC）
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_weights = model.state_dict().copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_weights,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'val_acc': val_epoch_acc,
            }, os.path.join(save_dir, 'best_model.pth'))

            print(f"新的最佳模型已保存! Val AUC: {val_auc:.4f}")

    # 加载最佳模型权重
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # 绘制最终的AUC曲线和性能报告
    _plot_final_metrics(all_predictions, save_dir)
    _generate_performance_report(all_predictions, save_dir)

    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_aucs': train_aucs,
        'val_aucs': val_aucs,
        'best_val_auc': best_val_auc,
        'all_predictions': all_predictions
    }

# 保留之前定义的辅助函数
def _plot_auc_curves(probabilities, labels, epoch, save_dir):
    """绘制AUC-ROC曲线"""
    from sklearn.metrics import roc_curve

    try:
        fpr, tpr, thresholds = roc_curve(labels, probabilities)
        auc_score = roc_auc_score(labels, probabilities)

        plt.figure(figsize=(10, 8))

        plt.subplot(2, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Epoch {epoch+1}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'auc_curves', f'auc_epoch_{epoch+1}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"绘制AUC曲线时出错: {e}")

def _plot_confusion_matrix(predictions, labels, epoch, save_dir):
    """绘制混淆矩阵"""
    try:
        cm = confusion_matrix(labels, predictions)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        plt.title(f'Confusion Matrix - Epoch {epoch+1}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'auc_curves', f'confusion_matrix_epoch_{epoch+1}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"绘制混淆矩阵时出错: {e}")

def _plot_final_metrics(all_predictions, save_dir):
    """绘制最终的训练指标曲线"""
    try:
        epochs = range(1, len(all_predictions) + 1)

        train_aucs = [pred['train_auc'] for pred in all_predictions]
        val_aucs = [pred['val_auc'] for pred in all_predictions]

        best_epoch = np.argmax(val_aucs)
        best_auc = val_aucs[best_epoch]

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_aucs, label='Train AUC', marker='o')
        plt.plot(epochs, val_aucs, label='Val AUC', marker='s')
        plt.axvline(x=best_epoch+1, color='red', linestyle='--',
                    label=f'Best Epoch: {best_epoch+1}\nBest AUC: {best_auc:.4f}')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('Training and Validation AUC')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'final_metrics.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"绘制最终指标时出错: {e}")

def _generate_performance_report(all_predictions, save_dir):
    """生成性能报告"""
    try:
        val_aucs = [pred['val_auc'] for pred in all_predictions]
        best_epoch = np.argmax(val_aucs)
        best_predictions = all_predictions[best_epoch]

        predictions = best_predictions['val_predictions']
        labels = best_predictions['val_labels']
        probabilities = best_predictions['val_probabilities']

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        auc = roc_auc_score(labels, probabilities)

        class_report = classification_report(labels, predictions, target_names=['Class 0', 'Class 1'])

        report_path = os.path.join(save_dir, 'performance_report.txt')
        with open(report_path, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("MODEL PERFORMANCE REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Best Epoch: {best_epoch + 1}\n")
            f.write(f"Validation AUC: {auc:.4f}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1-Score: {f1:.4f}\n\n")

            f.write("Classification Report:\n")
            f.write(class_report)

        print(f"性能报告已保存至: {report_path}")
    except Exception as e:
        print(f"生成性能报告时出错: {e}")

def save_heatmaps(grad_cam, images, labels, patient_ids, epoch, save_dir):
    """保存热图可视化"""
    batch_size = images.shape[0]

    for i in range(min(4, batch_size)):
        try:
            # 生成热图
            with torch.enable_grad():
                heatmap = grad_cam.generate_heatmap(images[i:i+1], labels[i:i+1])

            if heatmap is not None:
                slice_idx = heatmap.shape[0] // 2
                heatmap_slice = heatmap[slice_idx]
                image_slice = images[i, 0, slice_idx].cpu().detach().numpy()

                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

                ax1.imshow(image_slice, cmap='gray')
                ax1.set_title(f'Original Image\nPatient: {patient_ids[i]}')
                ax1.axis('off')

                im = ax2.imshow(heatmap_slice, cmap='jet', alpha=0.8)
                ax2.set_title('Grad-CAM Heatmap')
                ax2.axis('off')
                plt.colorbar(im, ax=ax2)

                ax3.imshow(image_slice, cmap='gray')
                ax3.imshow(heatmap_slice, cmap='jet', alpha=0.5)
                ax3.set_title('Overlay')
                ax3.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'heatmaps', f'epoch_{epoch+1}_patient_{patient_ids[i]}_heatmap.png'),
                           dpi=300, bbox_inches='tight')
                plt.close()

        except Exception as e:
            print(f"生成热图时出错 (患者 {patient_ids[i]}): {e}")
            continue

def _plot_cam(image, mask, cam, patient_id, true_label, pred_label, prob_high_risk, save_dir, sample_idx):
    """绘制CAM可视化图"""
    # try:
    if True:
        # 选择中间切片
        image = image[0]
        slice_idx = image.shape[0] // 2
        image_slice = image[slice_idx].cpu().numpy()

        mask = mask[0]
        mask_slice = np.int64(mask[slice_idx].cpu().numpy())
        # print(mask_slice)
        print("cam",(image_slice == 0).sum(), (mask_slice == 0).sum())

        # 处理CAM切片
        cam_slice = cam[slice_idx] if cam is not None else np.zeros_like(image_slice)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # 原始图像
        ax1.imshow(image_slice, cmap='gray')
        ax1.set_title(f'Original CT\nPatient: {patient_id}\nTrue: {true_label}, Pred: {pred_label}')
        ax1.axis('off')

        # CAM热图
        if cam is not None:
            im2 = ax2.imshow(cam_slice, cmap='jet', alpha=0.8)
            ax2.set_title('Class Activation Map\nActivation Intensity')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2)

        # CAM叠加
        image_slice[mask_slice == 0] = -1.0

        ax3.imshow(image_slice, cmap='gray')
        if cam is not None:
            ax3.imshow(cam_slice, cmap='jet', alpha=0.5)
        ax3.set_title('CAM Overlay\nHigh Activation = Model Focus')
        ax3.axis('off')

        risk_status = "High Risk" if pred_label == 1 else "Low Risk"
        prob_text = f"High Risk Probability: {prob_high_risk:.3f}"
        fig.suptitle(f'CAM Analysis - {risk_status}\n{prob_text}',
                     fontsize=16, y=0.95)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'cam_sample_{sample_idx}_{patient_id}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    # except Exception as e:
    #     print(f"绘制CAM可视化时出错: {e}")

def _plot_shap(image, mask, shap_values, patient_id, true_label, pred_label, prob_high_risk, save_dir, sample_idx):
    """绘制SHAP可视化图"""
    # try:
    if True:
        # 选择中间切片
        image = image[0]
        slice_idx = image.shape[0] // 2
        image_slice = image[slice_idx].cpu().numpy()

        mask = mask[0]
        mask_slice = np.int64(mask[slice_idx].cpu().numpy())
        # print(mask_slice)
        print((image_slice == 0).sum(), (mask_slice == 0).sum())

        # 处理SHAP值
        if isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 4:  # (batch, depth, height, width)
                shap_slice = shap_values[0, slice_idx]  # 取第一个批次和中间切片
            elif shap_values.ndim == 3:  # (depth, height, width)
                shap_slice = shap_values[slice_idx]
            elif shap_values.ndim == 2:  # (height, width) - 已经是切片
                shap_slice = shap_values
            else:
                print(f"警告: 意外的SHAP值形状 {shap_values.shape}")
                shap_slice = np.zeros_like(image_slice)
        else:
            print(f"警告: 意外的SHAP值类型 {type(shap_values)}")
            shap_slice = np.zeros_like(image_slice)

        # 调整SHAP切片大小以匹配图像
        if shap_slice.shape != image_slice.shape:
            import cv2
            shap_slice = cv2.resize(shap_slice, image_slice.shape[::-1])

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # 原始图像
        ax1.imshow(image_slice, cmap='gray')
        ax1.set_title(f'Original CT\nPatient: {patient_id}\nTrue: {true_label}, Pred: {pred_label}')
        ax1.axis('off')

        # SHAP方向热图
        if shap_slice.max() - shap_slice.min() > 1e-8:
            shap_directional = 2 * (shap_slice - shap_slice.min()) / (shap_slice.max() - shap_slice.min()) - 1
        else:
            shap_directional = np.zeros_like(shap_slice)

        im2 = ax2.imshow(shap_directional, cmap='bwr', alpha=0.8, vmin=-1, vmax=1)
        ax2.set_title('SHAP Values\nRed=High Risk, Blue=Low Risk')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2)

        # SHAP叠加
        shap_directional[mask_slice == 0] = -1.0
        ax3.imshow(image_slice, cmap='gray')
        ax3.imshow(shap_directional, cmap='bwr', alpha=0.5, vmin=-1, vmax=1)
        ax3.set_title('SHAP Overlay\nRisk Attribution')
        ax3.axis('off')

        risk_status = "High Risk" if pred_label == 1 else "Low Risk"
        prob_text = f"High Risk Probability: {prob_high_risk:.3f}"
        fig.suptitle(f'SHAP Analysis - {risk_status}\n{prob_text}',
                     fontsize=16, y=0.95)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'shap_sample_{sample_idx}_{patient_id}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    # except Exception as e:
    #     print(f"绘制SHAP可视化时出错: {e}")