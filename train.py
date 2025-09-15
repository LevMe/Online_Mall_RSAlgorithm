import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import RecommendationModel
from tqdm import tqdm

# --- 1. 配置参数 ---
DATA = r'./Datasets/douban_music/douban_music.tsv'
SEQ_LEN = 15  # 序列最大长度 (为了在小数据集上生效，调小该值)
BATCH_SIZE = 256
EPOCHS = 20  # 增加epoch以更好地学习
LEARNING_RATE = 0.001
EMBEDDING_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_ITEMS = 500
print(f"Using device: {DEVICE}")


# --- 2. 数据预处理 (V2) ---
def preprocess_data(filepath='ratings.csv'):
    """
    数据预处理函数 (V2):
    - 加载数据。
    - 创建ID映射，其中索引0保留给padding。
    - 生成用户行为序列。
    """
    print("Starting data preprocessing ...")
    df = pd.read_csv(filepath, sep='\t', encoding='utf-8')
    df = df[['UserId', 'ItemId', 'Timestamp']]
    df.rename(columns={'UserId': 'user_id', 'ItemId': 'item_id', 'Timestamp': 'timestamp'}, inplace=True)

    print(f"原始数据规模: {df.shape[0]} 条记录, {df['user_id'].nunique()} 个用户, {df['item_id'].nunique()} 个商品")

    # 2. 数据精简：筛选出交互最频繁的商品
    print(f"\n步骤 1: 筛选数据，仅保留Top {K_ITEMS} 热门商品相关的交互...")
    item_counts = df['item_id'].value_counts()
    num_items_to_keep = min(K_ITEMS, len(item_counts))
    top_items = item_counts.nlargest(num_items_to_keep).index
    print(f"已识别出 {len(top_items)} 个最热门的商品。")

    # 3. 基于热门商品，过滤交互数据
    df_filtered = df[df['item_id'].isin(top_items)]
    print(
        f"数据精简后规模: {df_filtered.shape[0]} 条记录, {df_filtered['user_id'].nunique()} 个用户, {df_filtered['item_id'].nunique()} 个商品")

    # --- 从这里开始，所有操作都基于 df_filtered ---
    df_filtered.sort_values(by='timestamp', inplace=True)

    # ID映射: 索引从1开始，0留给padding
    unique_users = df_filtered['user_id'].unique()
    unique_items = df_filtered['item_id'].unique()

    user_map = {int(id): int(i + 1) for i, id in enumerate(unique_users)}
    item_map = {int(id): int(i + 1) for i, id in enumerate(unique_items)}

    num_users = len(user_map)
    num_items = len(item_map)

    with open('maps/user_map.json', 'w') as f:
        json.dump(user_map, f)
    with open('maps/item_map.json', 'w') as f:
        json.dump(item_map, f)
    print("ID maps saved.")

    df_filtered['user_idx'] = df_filtered['user_id'].map(user_map)
    df_filtered['item_idx'] = df_filtered['item_id'].map(item_map)

    sequences = df_filtered.groupby('user_idx')['item_idx'].apply(list)

    print("Data preprocessing finished.")
    return sequences.to_dict(), num_users, num_items


# --- 3. 构建PyTorch数据集 (V2) ---
class SequenceDataset(Dataset):
    """
    PyTorch Dataset (V2):
    - 过滤掉行为序列过短的用户。
    - 使用滑动窗口从长序列中生成固定长度的训练样本。
    """

    def __init__(self, sequences_dict, seq_len):
        self.seq_len = seq_len
        self.sequences_dict = sequences_dict
        self.samples = self._create_samples()

    def _create_samples(self):
        samples = []
        for user_idx, item_list in self.sequences_dict.items():
            # 过滤掉行为序列长度不足以生成一个训练样本的用户
            if len(item_list) <= self.seq_len:
                continue

            # 使用滑动窗口生成样本
            # 窗口大小为 seq_len + 1 (输入 + 标签)
            for i in range(len(item_list) - self.seq_len):
                input_seq = item_list[i: i + self.seq_len]
                label = item_list[i + self.seq_len]
                samples.append((user_idx, input_seq, label))
        print(f"Created {len(samples)} samples from data.")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user_idx, input_seq, label = self.samples[idx]
        # 所有样本都已是固定长度，无需再padding
        return torch.tensor(user_idx), torch.tensor(input_seq, dtype=torch.long), torch.tensor(label)


# --- 4. 训练流程 (V2) ---
def train():
    """
    主训练函数。
    """
    sequences_dict, num_users, num_items = preprocess_data(DATA)

    dataset = SequenceDataset(sequences_dict, SEQ_LEN)
    if len(dataset) == 0:
        print(
            "No training samples were generated. This might be because all user sequences are shorter than SEQ_LEN+1.")
        print("Try reducing SEQ_LEN. Exiting.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型时，数量要+1，因为索引0被用作padding
    model = RecommendationModel(
        num_users=num_users + 1,
        num_items=num_items + 1,
        embedding_dim=EMBEDDING_DIM,
        seq_len=SEQ_LEN
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for user_ids, item_seqs, labels in progress_bar:
            user_ids, item_seqs, labels = user_ids.to(DEVICE), item_seqs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(user_ids, item_seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")

    print("Training finished.")

    torch.save(model.state_dict(), 'checkpoint/model.pth')
    print("Model state dict saved to model.pth")

    # 提取商品Embedding时，要忽略索引0 (padding)
    item_embeddings = model.item_embedding.weight.detach().cpu().numpy()[1:]
    np.save('checkpoint/item_embeddings.npy', item_embeddings)
    print("Item embeddings saved to item_embeddings.npy")
    # 注意：保存的item_embeddings.npy的行数将是num_items，其索引与item_map中的索引(1 to N)相对应。


if __name__ == '__main__':
    train()
