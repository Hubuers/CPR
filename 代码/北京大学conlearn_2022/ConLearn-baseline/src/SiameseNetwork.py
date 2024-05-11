import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import random
# 固定随机种子
def set_seed(seed_value=42):
    random.seed(seed_value)  # Python random module.
    np.random.seed(seed_value)  # Numpy module.
    torch.manual_seed(seed_value)  # PyTorch.
    if torch.cuda.is_available():  # CUDA.
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)  # 可以在这里设置任何您喜欢的数字

# 从CSV文件中加载概念嵌入
def load_embeddings(embeddings_file):
    df_embeddings = pd.read_csv(embeddings_file)
    embeddings = {
        row['node_name']: torch.tensor(
            [float(x) for x in row['embeddings'].strip('[]').split(',')], dtype=torch.float
        )
        for _, row in df_embeddings.iterrows()
    }

    return embeddings

embeddings = load_embeddings('path_to_your_output.csv')


# 定义数据集
class PrerequisiteDataset(Dataset):
    def __init__(self, csv_file, embeddings):
        self.dataframe = pd.read_csv(csv_file)
        self.embeddings = embeddings

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        concept_a_emb = self.embeddings[row['A']]
        concept_b_emb = self.embeddings[row['B']]
        label = row['result']
        return concept_a_emb, concept_b_emb, torch.tensor(label, dtype=torch.float)

# 创建数据加载器
def create_dataloader(csv_file, embeddings, batch_size=64):
    dataset = PrerequisiteDataset(csv_file, embeddings)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = create_dataloader('../data/MOOC/train_concept_pair.csv', embeddings)
val_loader = create_dataloader('../data/MOOC/val_concept_pair.csv', embeddings)
test_loader = create_dataloader('../data/MOOC/test_concept_pair.csv', embeddings)

# 定义孪生网络
class SiameseNetwork(nn.Module):
    def __init__(self, representation_size):
        super(SiameseNetwork, self).__init__()
        # 前馈网络部分
        self.ffn = nn.Linear(representation_size, representation_size)
        # 分类部分
        self.classifier = nn.Linear(representation_size * 4, 1)

    def forward(self, e_i, e_j):
        # 两个前馈网络的处理，共享权重
        e_i = F.relu(self.ffn(e_i))
        e_j = F.relu(self.ffn(e_j))

        # 拼接两个向量，并且包括元素级相减和相乘的结果
        combined = torch.cat([e_i, e_j, e_i - e_j, e_i * e_j], dim=1)

        # 分类
        p = torch.sigmoid(self.classifier(combined))

        return p

# 评估函数
def evaluate(model, data_loader):
    model.eval()
    predictions, actuals = [], []
    scores = []
    with torch.no_grad():
        for concept_a_emb, concept_b_emb, label in data_loader:
            outputs = model(concept_a_emb, concept_b_emb).squeeze()
            predicted = (outputs > 0.5).float()
            predictions.extend(predicted.numpy())
            actuals.extend(label.numpy())
            scores.extend(outputs.numpy())

    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions, average='binary')
    recall = recall_score(actuals, predictions, average='binary')
    f1 = f1_score(actuals, predictions, average='binary')
    auc = roc_auc_score(actuals, scores)

    return accuracy, precision, recall, f1, auc


# 初始化孪生网络和优化器
representation_size = next(iter(embeddings.values())).size(0)
model = SiameseNetwork(representation_size)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 训练循环中跟踪最佳模型
best_val_F1 = 0
best_model_state = None

# 训练循环
num_epochs = 100  # 这个值可以根据需要调整
for epoch in range(num_epochs):
    model.train()
    losses = []  # 用于收集每个batch的损失
    for concept_a_emb, concept_b_emb, label in train_loader:
        optimizer.zero_grad()
        outputs = model(concept_a_emb, concept_b_emb)
        loss = criterion(outputs.squeeze(), label)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # 计算平均损失
    avg_loss = np.mean(losses)

    # 在验证集上评估
    val_metrics = evaluate(model, val_loader)
    print(
        f'Epoch {epoch + 1}, Loss: {avg_loss:.6f}, '
        f'Validation Metrics: ACC={val_metrics[0]:.4f}, Pre={val_metrics[1]:.4f}, Recall={val_metrics[2]:.4f}, F1={val_metrics[3]:.4f}, AUC={val_metrics[4]:.4f}')

    # 如果在验证集上的F1更好，则保存模型状态
    if val_metrics[3] > best_val_F1:
        best_val_F1 = val_metrics[3]
        best_model_state = model.state_dict()
        torch.save(best_model_state, '../Dataset/data_mining/best_model.pth')
        print(f'Saved new best model with F1: {best_val_F1:.4f}')

# 测试最佳模型
model.load_state_dict(torch.load('../Dataset/data_mining/best_model.pth'))
test_metrics = evaluate(model, test_loader)
print(f'Test Metrics with Best Model: ACC={test_metrics[0]:.4f}, Pre={test_metrics[1]:.4f}, Recall={test_metrics[2]:.4f}, F1={test_metrics[3]:.4f}, AUC={test_metrics[4]:.4f}')
