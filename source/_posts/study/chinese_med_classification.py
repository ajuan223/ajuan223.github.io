import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50, densenet121

from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import time
import copy
from tqdm import tqdm

# 设置随机种子确保结果可重复
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# 1. 数据增强和预处理
def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# 2. 自定义中药数据集类
class ChineseMedicineDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 3. 定义不同的CNN模型

# 自定义CNN模型
class CustomCNN(nn.Module):
    def __init__(self, num_classes=20):
        super(CustomCNN, self).__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        # 第二个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        
        # 第三个卷积块
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)
        
        # 全连接层
        self.flatten = nn.Flatten()
        # 224/8 = 28
        self.fc1 = nn.Linear(256 * 28 * 28, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # 第一个卷积块
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        
        # 第二个卷积块
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout2(x)
        
        # 第三个卷积块
        x = self.pool3(F.relu(self.bn5(self.conv5(x))))
        x = self.dropout3(x)
        
        # 全连接层
        x = self.flatten(x)
        x = F.relu(self.bn6(self.fc1(x)))
        x = self.dropout4(x)
        x = self.fc2(x)
        
        return x

# 创建基于ResNet的模型
def create_resnet_model(num_classes=20, freeze_backbone=True):
    model = resnet18(pretrained=True)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # 修改最后的全连接层
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

# 创建基于DenseNet的模型
def create_densenet_model(num_classes=20, freeze_backbone=True):
    model = densenet121(pretrained=True)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # 修改最后的全连接层
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    
    return model

# 4. 数据集分割和加载
def load_and_split_data(data_dir, batch_size=32, val_split=0.2):
    train_transform, val_transform = get_transforms()
    
    # 加载完整数据集
    full_dataset = ChineseMedicineDataset(data_dir, transform=None)
    
    # 计算分割大小
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    # 随机分割数据集
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 应用不同的转换
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, full_dataset.classes

# 5. 训练和评估函数
def train_model(model, train_loader, val_loader, criterion, optimizer, 
               scheduler=None, num_epochs=25, device='cuda'):
    since = time.time()
    model = model.to(device)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        if scheduler:
            scheduler.step()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # 验证阶段
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        
        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc.item())
        
        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # 保存最佳模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f'best_model_{model.__class__.__name__}.pth')
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, history

# 6. 模型评估函数
def evaluate_model(model, data_loader, device='cuda', num_classes=20):
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # 获取每个类别的概率
            probs = F.softmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 计算评估指标
    accuracy = np.mean(all_preds == all_labels)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    # 计算并绘制ROC曲线 (使用one-vs-rest策略)
    plt.figure(figsize=(10, 8))
    
    # 存储每个类别的AUC值
    auc_values = []
    
    for i in range(num_classes):
        # 二分类标签
        binary_labels = (all_labels == i).astype(int)
        class_probs = all_probs[:, i]
        
        # 计算ROC
        fpr, tpr, _ = roc_curve(binary_labels, class_probs)
        roc_auc = auc(fpr, tpr)
        auc_values.append(roc_auc)
        
        plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_curve_{model.__class__.__name__}.png')
    plt.close()
    
    # 绘制混淆矩阵
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(all_labels, all_preds):
        cm[true_label, pred_label] += 1
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'confusion_matrix_{model.__class__.__name__}.png')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_values': auc_values,
        'confusion_matrix': cm
    }

# 7. 主函数 - 运行三种不同模型配置
def main():
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 数据集路径
    data_dir = 'path/to/chinese_medicine_dataset'  # 请替换为实际数据集路径
    
    # 加载和分割数据
    train_loader, val_loader, class_names = load_and_split_data(data_dir, batch_size=32)
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    
    # 创建结果记录
    results = {}
    
    # 模型配置列表
    model_configs = [
        {
            'name': 'CustomCNN',
            'model': CustomCNN(num_classes=num_classes),
            'optimizer': optim.Adam,
            'lr': 0.001,
            'weight_decay': 1e-4,
            'epochs': 30
        },
        {
            'name': 'ResNet18',
            'model': create_resnet_model(num_classes=num_classes, freeze_backbone=True),
            'optimizer': optim.SGD,
            'lr': 0.01,
            'weight_decay': 5e-4,
            'epochs': 20
        },
        {
            'name': 'DenseNet121',
            'model': create_densenet_model(num_classes=num_classes, freeze_backbone=True),
            'optimizer': optim.Adam,
            'lr': 0.0005,
            'weight_decay': 1e-4,
            'epochs': 15
        }
    ]
    
    # 训练和评估每个模型配置
    for config in model_configs:
        print(f"\n\n{'='*50}")
        print(f"Training model: {config['name']}")
        print(f"{'='*50}\n")
        
        model = config['model']
        criterion = nn.CrossEntropyLoss()
        optimizer = config['optimizer'](
            model.parameters(), 
            lr=config['lr'], 
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # 训练模型
        trained_model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            scheduler=scheduler, num_epochs=config['epochs'], device=device
        )
        
        # 评估模型
        print(f"\nEvaluating model: {config['name']}")
        eval_results = evaluate_model(
            trained_model, val_loader, device=device, num_classes=num_classes
        )
        
        # 保存训练历史和评估结果
        results[config['name']] = {
            'history': history,
            'evaluation': eval_results
        }
        
        # 绘制学习曲线
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'{config["name"]} - Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title(f'{config["name"]} - Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'learning_curve_{config["name"]}.png')
        plt.close()
    
    # 比较不同模型的性能
    compare_models(results)

# 8. 模型比较函数
def compare_models(results):
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # 创建比较图表
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(models))
    width = 0.2
    offsets = [-width*1.5, -width*0.5, width*0.5, width*1.5]
    
    for i, metric in enumerate(metrics):
        values = [results[model]['evaluation'][metric] for model in models]
        plt.bar(x + offsets[i], values, width, label=metric.capitalize())
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Performance Comparison of Different Models')
    plt.xticks(x, models)
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig('model_comparison.png')
    plt.close()
    
    # 打印综合评估结果
    print("\n===== 模型性能比较 =====")
    for model in models:
        print(f"\nModel: {model}")
        for metric in metrics:
            print(f"{metric.capitalize()}: {results[model]['evaluation'][metric]:.4f}")
        print(f"平均AUC: {np.mean(results[model]['evaluation']['auc_values']):.4f}")

if __name__ == "__main__":
    main()