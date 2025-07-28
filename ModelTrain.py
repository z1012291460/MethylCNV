# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch import cuda
import sys, os
import random
import numpy as np
from sklearn import metrics
import gc
import time
import logging
from collections import OrderedDict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 命令行参数
parser = argparse.ArgumentParser(description='CNV-LSTM')
parser.add_argument('--lr', type=float, default=0.0001, help='初始学习率')
parser.add_argument('--model_name', type=str, default='CNV_BiLSTM', help='模型名称') 
parser.add_argument('--clip', type=float, default=1, help='梯度裁剪阈值')
parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
parser.add_argument('--batch_size', type=int, default=192, help='批量大小')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout比例')
parser.add_argument('--save_root', type=str, default='./model_output', help='保存路径')
parser.add_argument('--data_path', type=str, default='train_data_human_bin100_new_chrom.npz', help='特征数据路径')
parser.add_argument('--gpuid', type=int, default=0, help='CUDA GPU ID')
parser.add_argument('--n_features', type=int, default=15, help='特征数量')
parser.add_argument('--seq_length', type=int, default=50, help='序列长度')
parser.add_argument('--stride', type=int, default=10, help='滑动窗口步长')
parser.add_argument('--bin_rnn_size', type=int, default=128, help='Bin RNN隐藏层大小')
parser.add_argument('--num_layers', type=int, default=1, help='LSTM层数')
parser.add_argument('--bidirectional', type=bool, default=True, help='是否使用双向LSTM')
parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
parser.add_argument('--val_size', type=float, default=0.1, help='验证集比例')
parser.add_argument('--num_workers', type=int, default=4, help='数据加载的工作进程数')
args = parser.parse_args()

# 设置随机种子
torch.manual_seed(2023)
np.random.seed(2023)
random.seed(2023)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(2023)

# 设置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CNV-LSTM")

if not os.path.exists(args.save_root):
    os.makedirs(args.save_root)

# 数据集类
class CNVSequenceDataset(Dataset):
    """CNV序列数据集"""
    def __init__(self, features, labels, seq_length=50):
        self.features = features
        self.labels = labels
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature_seq = self.features[idx]  # 已经是序列形式
        label = self.labels[idx]  # 取序列中心点的标签
        
        sample = {
            'features': torch.FloatTensor(feature_seq),
            'label': torch.LongTensor([label])
        }
        return sample

# 创建序列数据
def create_sequences(X, y, chrom_col='chrom', bin_start_col='bin_start', 
                     seq_length=50, stride=10):
    """
    将特征数据X和标签y按照染色体和位置排序，
    创建固定长度的序列用于LSTM训练，并对DEL和DUP类型样本进行数据增强
    """
    logger.info("创建序列数据...")
    sequences_X = []
    sequences_y = []
    
    # 按染色体分组处理
    for chrom in X[chrom_col].unique():
        # 筛选当前染色体的数据
        chrom_mask = X[chrom_col] == chrom
        
        # 获取当前染色体对应的原始索引
        orig_indices = np.where(chrom_mask)[0]
        
        # 创建临时DataFrame并添加原始索引列
        temp_df = X[chrom_mask].copy()
        temp_df['orig_idx'] = orig_indices
        
        # 按bin_start排序
        temp_df = temp_df.sort_values(by=bin_start_col).reset_index(drop=True)
        
        # 获取排序后的原始索引用于提取标签
        sorted_indices = temp_df['orig_idx'].values
        y_chrom = y[sorted_indices]
        
        # 去除辅助列
        temp_df = temp_df.drop('orig_idx', axis=1)
        
        # 丢弃非特征列
        feature_cols = [col for col in temp_df.columns 
                         if col not in [chrom_col, bin_start_col, 'sample_id']]
        X_features = temp_df[feature_cols].values
        
        # 创建序列
        for i in range(0, len(temp_df) - seq_length + 1, stride):
            seq_x = X_features[i:i+seq_length]
            # 取序列中心位置的标签
            center_idx = i + seq_length // 2
            seq_y = y_chrom[center_idx]
            
            # 添加原始序列
            sequences_X.append(seq_x)
            sequences_y.append(seq_y)
            
            # 对DEL类(1)和DUP类(2)样本进行数据增强
            if seq_y == 1:  # DEL类
                # 对DEL类添加3个噪声变种 (重复次数更多)
                for noise_scale in [0.03, 0.05, 0.07]:
                    noise = np.random.normal(0, noise_scale, seq_x.shape)
                    sequences_X.append(seq_x + noise)
                    sequences_y.append(seq_y)
                    
            elif seq_y == 2:  # DUP类
                # 对DUP类添加2个噪声变种而非1个
                for noise_scale in [0.04]:
                    noise = np.random.normal(0, noise_scale, seq_x.shape)
                    sequences_X.append(seq_x + noise)
                    sequences_y.append(seq_y)
    
    logger.info(f"原始序列数: {len(sequences_X)}, 其中包含增强样本")
    return np.array(sequences_X), np.array(sequences_y)

# 加载和准备数据
def load_and_prepare_data(data_path, seq_length=50, stride=10):
    """加载特征数据，标准化，创建序列，划分训练/验证/测试集"""
    logger.info(f"加载数据文件: {data_path}")
    data = np.load(data_path, allow_pickle=True)
    
    # 获取特征和标签
    X = pd.DataFrame(data['features'])
    feature_names = data['feature_names']
    X.columns = feature_names
    y = data['labels']
    
    logger.info(f"特征形状: {X.shape}, 标签形状: {y.shape}")
    logger.info(f"标签分布: 0 (正常): {np.sum(y==0)}, 1 (DEL): {np.sum(y==1)}, 2 (DUP): {np.sum(y==2)}")
    
    # 标准化特征
    numerical_cols = [col for col in X.columns if col not in ['chrom', 'bin_start', 'sample_id']]
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # 创建序列数据
    X_seq, y_seq = create_sequences(X, y, seq_length=seq_length, stride=stride)
    logger.info(f"创建了 {len(X_seq)} 个序列")
    
    # 分割数据集
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_seq, y_seq, test_size=args.test_size, random_state=42, stratify=y_seq)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=args.val_size/(1-args.test_size), 
        random_state=42, stratify=y_train_val)
    
    logger.info(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    
    # 创建数据集和数据加载器
    train_dataset = CNVSequenceDataset(X_train, y_train, seq_length)
    val_dataset = CNVSequenceDataset(X_val, y_val, seq_length)
    test_dataset = CNVSequenceDataset(X_test, y_test, seq_length)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,  # 使用4个工作进程并行加载数据
        pin_memory=True,  # 加速CPU到GPU的数据传输
        prefetch_factor=2  # 预加载数据批次
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    # 获取实际特征数量
    actual_features = len(numerical_cols)
    logger.info(f"实际特征数量: {actual_features}")
    
    return train_loader, val_loader, test_loader, scaler, actual_features

# 注意力机制
class Attention(nn.Module):
    def __init__(self, hidden_size, level_name="bin"):
        super(Attention, self).__init__()
        self.level_name = level_name
        self.context_vector = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.softmax = nn.Softmax(dim=1)
        # 初始化
        nn.init.uniform_(self.context_vector, -0.1, 0.1)
        
    def forward(self, x):
        # x: [batch, seq_len, hidden]
        # 计算注意力得分
        attn_scores = torch.matmul(x, self.context_vector)  # [batch, seq_len, 1]
        attn_weights = self.softmax(attn_scores.squeeze(2))  # [batch, seq_len]
        
        # 应用注意力权重
        weighted_output = torch.bmm(attn_weights.unsqueeze(1), x)  # [batch, 1, hidden]
        return weighted_output.squeeze(1), attn_weights

# 特征级编码器
class FeatureEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True, dropout=0.5):
        super(FeatureEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = Attention(hidden_size * self.num_directions, "bin")
        
    def forward(self, x):
        # x: [batch, seq_len, input_size]
        outputs, _ = self.lstm(x)
        context, attention = self.attention(outputs)
        
        # 添加全局平均池化作为残差连接
        global_avg = torch.mean(outputs, dim=1)
        context = context + global_avg
        
        return context, attention

# 添加焦点损失类
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# CNV检测模型
class CNVDetector(nn.Module):
    def __init__(self, args):
        super(CNVDetector, self).__init__()
        self.n_features = args.n_features
        self.seq_length = args.seq_length
        self.hidden_size = args.bin_rnn_size
        self.num_directions = 2 if args.bidirectional else 1
        
        # 特征编码器
        self.feature_encoders = nn.ModuleList()
        for i in range(self.n_features):
            self.feature_encoders.append(
                FeatureEncoder(
                    input_size=1,
                    hidden_size=self.hidden_size,
                    num_layers=args.num_layers,
                    bidirectional=args.bidirectional,
                    dropout=args.dropout
                )
            )
        
        # 特征融合编码器
        feature_hidden_size = self.hidden_size * self.num_directions
        self.feature_fusion = nn.LSTM(
            input_size=feature_hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=args.bidirectional
        )
        
        self.feature_attention = Attention(self.hidden_size * self.num_directions, "feature")
        
        # 分类器
        classifier_input_size = self.hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 128),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(args.dropout/2),
            nn.Linear(64, 3)  # 3类: 正常, DEL, DUP
        )
        
    def forward(self, x):
        # x: [batch, seq_len, n_features]
        batch_size = x.size(0)
        actual_features = x.size(2)  # 获取实际特征数量
        
        # 处理每种特征
        feature_outputs = []
        bin_attentions = []
        
        # 只处理可用的特征数量
        for i in range(min(self.n_features, actual_features)):
            # 提取单个特征
            feature = x[:, :, i:i+1]  # [batch, seq_len, 1]
            
            # 通过特征编码器
            feature_output, bin_attn = self.feature_encoders[i](feature)
            feature_outputs.append(feature_output.unsqueeze(1))  # [batch, 1, hidden]
            bin_attentions.append(bin_attn)
        
        # 合并特征表示
        feature_repr = torch.cat(feature_outputs, dim=1)  # [batch, n_features, hidden]
        
        # 添加特征交互层
        feature_interact = torch.bmm(
            feature_repr.transpose(1, 2),  # [batch, hidden, n_features]
            feature_repr  # [batch, n_features, hidden]
        )  # [batch, hidden, hidden]
        feature_interact = F.relu(feature_interact)
        feature_interact = torch.bmm(feature_repr, feature_interact)  # [batch, n_features, hidden]
        feature_repr = feature_repr + 0.15 * feature_interact  # 残差连接
        
        # 特征融合
        fused_output, _ = self.feature_fusion(feature_repr)
        context_vector, feature_attn = self.feature_attention(fused_output)
        
        # 分类
        logits = self.classifier(context_vector)
        
        bin_attentions_tensor = torch.stack(bin_attentions, dim=1)  # [batch, n_features, seq_len]
        
        return logits, feature_attn, bin_attentions_tensor

# 训练函数
def train(model, train_loader, optimizer, criterion, device, clip=1.0):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, batch in enumerate(train_loader):
        features = batch['features'].to(device)
        labels = batch['label'].squeeze(1).to(device)
        
        optimizer.zero_grad()
        outputs, feature_attn, bin_attn = model(features)
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        total_loss += loss.item()
        
        # 收集预测结果
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        if batch_idx % 100 == 0:
            logger.info(f'Train Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
    
    # 计算总损失和指标
    avg_loss = total_loss / len(train_loader)
    accuracy = metrics.accuracy_score(all_labels, all_preds)
    f1 = metrics.f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1, all_preds, all_labels

# 验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            features = batch['features'].to(device)
            labels = batch['label'].squeeze(1).to(device)
            
            outputs, feature_attn, bin_attn = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # 收集预测结果
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # 计算总损失和指标
    avg_loss = total_loss / len(val_loader)
    accuracy = metrics.accuracy_score(all_labels, all_preds)
    f1 = metrics.f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = metrics.confusion_matrix(all_labels, all_preds)
    
    return avg_loss, accuracy, f1, conf_matrix, all_preds, all_labels

# 测试函数
def test(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_feature_attn = []
    all_bin_attn = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            labels = batch['label'].squeeze(1).to(device)
            
            outputs, feature_attn, bin_attn = model(features)
            
            # 收集预测结果和注意力
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_feature_attn.append(feature_attn.cpu().numpy())
            all_bin_attn.append(bin_attn.cpu().numpy())
    
    # 计算指标
    accuracy = metrics.accuracy_score(all_labels, all_preds)
    precision = metrics.precision_score(all_labels, all_preds, average='weighted')
    recall = metrics.recall_score(all_labels, all_preds, average='weighted')
    f1 = metrics.f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = metrics.confusion_matrix(all_labels, all_preds)
    
    # 计算每个类别的准确率和召回率
    per_class_precision = metrics.precision_score(all_labels, all_preds, average=None)
    per_class_recall = metrics.recall_score(all_labels, all_preds, average=None)
    
    # 计算每个类别的F1分数
    per_class_f1 = metrics.f1_score(all_labels, all_preds, average=None)
    
    # 合并注意力权重
    all_feature_attn = np.concatenate(all_feature_attn, axis=0)
    all_bin_attn = np.concatenate(all_bin_attn, axis=0)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'predictions': all_preds,
        'labels': all_labels,
        'feature_attention': all_feature_attn,
        'bin_attention': all_bin_attn
    }
    
    return results

# 主函数
def main():
    # 检查GPU是否可用
    device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 输出模型参数
    logger.info("======= 模型参数 =======")
    logger.info(f"学习率: {args.lr}")
    logger.info(f"批量大小: {args.batch_size}")
    logger.info(f"序列长度: {args.seq_length}")
    logger.info(f"LSTM隐藏层大小: {args.bin_rnn_size}")
    logger.info(f"LSTM层数: {args.num_layers}")
    logger.info(f"是否使用双向LSTM: {args.bidirectional}")
    logger.info(f"丢弃率: {args.dropout}")
    logger.info(f"数据加载器进程数: {args.num_workers}")
    logger.info("========================")
    
    # 加载数据并获取实际特征数量
    train_loader, val_loader, test_loader, scaler, actual_features = load_and_prepare_data(
        args.data_path, args.seq_length, args.stride)
    
    # 更新模型的特征数量
    args.n_features = actual_features
    logger.info(f"使用动态获取的特征数量: {args.n_features}")
    
    # 统计各类别样本数量
    train_labels = []
    for batch in train_loader:
        labels = batch['label'].squeeze(1).numpy()
        train_labels.extend(labels)
    train_labels = np.array(train_labels)

    class_counts = [np.sum(train_labels==i) for i in range(3)]
    logger.info("======= 训练数据分布 =======")
    logger.info(f"类别 0 (Normal): {class_counts[0]} 样本, 占比: {class_counts[0]/len(train_labels):.2%}")
    logger.info(f"类别 1 (DEL): {class_counts[1]} 样本, 占比: {class_counts[1]/len(train_labels):.2%}")
    logger.info(f"类别 2 (DUP): {class_counts[2]} 样本, 占比: {class_counts[2]/len(train_labels):.2%}")
    logger.info(f"总样本数: {len(train_labels)}")
    logger.info("===========================")
    
    # 创建模型
    model = CNVDetector(args).to(device)
    logger.info(f"模型结构:\n{model}")
    
    # 设置类别权重 (保持原有权重设置)
    weights = torch.FloatTensor([1.0, 3.0, 1.5]).to(device)

    # 使用带权重的焦点损失
    criterion = FocalLoss(gamma=2.0, weight=weights)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 使用学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # 训练模型
    best_val_f1 = 0
    best_epoch = 0
    patience = 10  # 早停
    patience_counter = 0
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # 训练
        train_loss, train_acc, train_f1, _, _ = train(
            model, train_loader, optimizer, criterion, device, args.clip)
        
        # 验证
        val_loss, val_acc, val_f1, val_conf_matrix, _, _ = validate(
            model, val_loader, criterion, device)
        
        logger.info(f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        logger.info(f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        logger.info(f"Val Confusion Matrix:\n{val_conf_matrix}")
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'scaler': scaler
            }, os.path.join(args.save_root, f"{args.model_name}_best.pth"))
            
            logger.info(f"保存最佳模型, F1: {val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"早停! {patience}个epoch没有改善。")
                break
        
        # 在验证后添加：
        scheduler.step(val_f1)
    
    logger.info(f"完成训练! 最佳验证F1: {best_val_f1:.4f} (Epoch {best_epoch+1})")
    
    # 加载最佳模型进行测试
    checkpoint = torch.load(os.path.join(args.save_root, f"{args.model_name}_best.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 在测试集上测试
    test_results = test(model, test_loader, device)
    
    logger.info("测试集结果:")
    logger.info(f"总体准确率: {test_results['accuracy']:.4f}")
    logger.info(f"总体精确率: {test_results['precision']:.4f}")
    logger.info(f"总体召回率: {test_results['recall']:.4f}")
    logger.info(f"总体F1分数: {test_results['f1']:.4f}")
    
    logger.info("\n各类别性能指标:")
    class_names = ['Normal', 'DEL', 'DUP']
    for i, class_name in enumerate(class_names):
        logger.info(f"{class_name}类 - 精确率: {test_results['per_class_precision'][i]:.4f}, " +
                   f"召回率: {test_results['per_class_recall'][i]:.4f}, " +
                   f"F1分数: {test_results['per_class_f1'][i]:.4f}")
    
    logger.info(f"\n混淆矩阵:\n{test_results['confusion_matrix']}")
    
    # 根据混淆矩阵计算并输出更详细的分析
    conf_mat = test_results['confusion_matrix']
    logger.info("\n基于混淆矩阵的详细分析:")
    for i, class_name in enumerate(class_names):
        true_positive = conf_mat[i, i]
        false_negative = np.sum(conf_mat[i, :]) - true_positive
        false_positive = np.sum(conf_mat[:, i]) - true_positive
        
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        
        logger.info(f"{class_name}类 - 真阳性: {true_positive}, 假阴性: {false_negative}, 假阳性: {false_positive}")
        logger.info(f"{class_name}类 - 精确率: {precision:.4f}, 召回率: {recall:.4f}")
        
        # 输出被误分为其他类的情况
        for j, other_class in enumerate(class_names):
            if i != j:
                logger.info(f"  {class_name}误分为{other_class}: {conf_mat[i, j]} ({conf_mat[i, j]/np.sum(conf_mat[i, :]):.2%})")
    
    # 保存测试结果
    np.savez(os.path.join(args.save_root, f"{args.model_name}_test_results.npz"),
             predictions=test_results['predictions'],
             labels=test_results['labels'],
             feature_attention=test_results['feature_attention'],
             bin_attention=test_results['bin_attention'],
             confusion_matrix=test_results['confusion_matrix'])
    
    # 保存混淆矩阵图
    plt.figure(figsize=(10, 8))
    conf_mat = test_results['confusion_matrix']
    classes = ['Normal', 'DEL', 'DUP']
    
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = 'd'
    thresh = conf_mat.max() / 2.
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(j, i, format(conf_mat[i, j], fmt),
                     ha="center", va="center",
                     color="white" if conf_mat[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_root, f"{args.model_name}_confusion_matrix.png"))
    
    # 保存特征重要性
    feature_importance = np.mean(test_results['feature_attention'], axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Feature Index')
    plt.ylabel('Attention Weight')
    plt.savefig(os.path.join(args.save_root, f"{args.model_name}_feature_importance.png"))
    
    logger.info(f"所有结果已保存到 {args.save_root}")

if __name__ == "__main__":
    main()