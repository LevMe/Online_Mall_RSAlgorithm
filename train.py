import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import RecommendationModel
from tqdm import tqdm
import os

# --- 1. 配置参数 ---
DATA_PATH = r'./Datasets/douban_music/douban_music.tsv'
SEQ_LEN = 15  # 序列最大长度 (为了在小数据集上生效，调小该值)
BATCH_SIZE = 256
EPOCHS = 20  # 增加epoch以更好地学习
LEARNING_RATE = 0.001
EMBEDDING_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_ITEMS = 300
CHECKPOINT_DIR = 'checkpoint'
MAPS_DIR = 'maps'

print(f"Using device: {DEVICE}")


# --- 辅助函数：创建目录 ---
def create_dirs():
    """确保所有必要的目录都存在"""
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    if not os.path.exists(MAPS_DIR):
        os.makedirs(MAPS_DIR)


# --- 2. 数据预处理 (从文件) ---
def preprocess_data_from_file(filepath=DATA_PATH):
    """
    数据预处理函数 (V2 - 从文件加载):
    - 加载TSV数据。
    - 创建ID映射，其中索引0保留给padding。
    - 生成用户行为序列。
    """
    print("Starting data preprocessing from file...")
    df = pd.read_csv(filepath, sep='\t', encoding='utf-8')
    df = df[['UserId', 'ItemId', 'Timestamp']]
    df.rename(columns={'UserId': 'user_id', 'ItemId': 'item_id', 'Timestamp': 'timestamp'}, inplace=True)
    return _process_dataframe(df)


# --- 新增：数据预处理 (从后端数据) ---
def preprocess_data_from_backend(user_behaviors):
    """
    处理从后端传递过来的用户行为数据。
    - user_behaviors: 一个字典列表, e.g., [{'user_id': 1, 'product_id': 101, 'timestamp': 1625078400}, ...]
    """
    print("Starting data preprocessing from backend data...")
    if not user_behaviors:
        print("Error: Received empty user behavior data.")
        return None, 0, 0

    df = pd.DataFrame(user_behaviors)
    # 将 'product_id' 重命名为 'item_id' 以匹配后续流程
    df.rename(columns={'product_id': 'item_id'}, inplace=True)

    # 确保数据类型正确
    df['user_id'] = pd.to_numeric(df['user_id'])
    df['item_id'] = pd.to_numeric(df['item_id'])
    df['timestamp'] = pd.to_numeric(df['timestamp'])

    return _process_dataframe(df)


def _process_dataframe(df):
    """
    DataFrame处理的核心逻辑，被两个预处理函数共享。
    """
    print(f"原始数据规模: {df.shape[0]} 条记录, {df['user_id'].nunique()} 个用户, {df['item_id'].nunique()} 个商品")

    print(f"\n步骤 1: 筛选数据，仅保留Top {K_ITEMS} 热门商品相关的交互...")
    item_counts = df['item_id'].value_counts()
    num_items_to_keep = min(K_ITEMS, len(item_counts))
    top_items = item_counts.nlargest(num_items_to_keep).index
    print(f"已识别出 {len(top_items)} 个最热门的商品。")

    df_filtered = df[df['item_id'].isin(top_items)]
    print(
        f"数据精简后规模: {df_filtered.shape[0]} 条记录, {df_filtered['user_id'].nunique()} 个用户, {df_filtered['item_id'].nunique()} 个商品")

    df_filtered.sort_values(by='timestamp', inplace=True)

    unique_users = df_filtered['user_id'].unique()
    unique_items = df_filtered['item_id'].unique()

    user_map = {int(id): int(i + 1) for i, id in enumerate(unique_users)}
    item_map = {int(id): int(i + 1) for i, id in enumerate(unique_items)}

    num_users = len(user_map)
    num_items = len(item_map)

    create_dirs()  # 确保目录存在
    with open(os.path.join(MAPS_DIR, 'user_map.json'), 'w') as f:
        json.dump(user_map, f)
    with open(os.path.join(MAPS_DIR, 'item_map.json'), 'w') as f:
        json.dump(item_map, f)
    print("ID maps saved.")

    df_filtered['user_idx'] = df_filtered['user_id'].map(user_map)
    df_filtered['item_idx'] = df_filtered['item_id'].map(item_map)

    sequences = df_filtered.groupby('user_idx')['item_idx'].apply(list)

    print("Data preprocessing finished.")
    return sequences.to_dict(), num_users, num_items


# --- 3. 构建PyTorch数据集 (V2) ---
class SequenceDataset(Dataset):
    def __init__(self, sequences_dict, seq_len):
        self.seq_len = seq_len
        self.sequences_dict = sequences_dict
        self.samples = self._create_samples()

    def _create_samples(self):
        samples = []
        for user_idx, item_list in self.sequences_dict.items():
            if len(item_list) <= self.seq_len:
                continue
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
        return torch.tensor(user_idx, dtype=torch.long), torch.tensor(input_seq, dtype=torch.long), torch.tensor(label,
                                                                                                                 dtype=torch.long)


# --- 4. 训练流程 (V2) ---
def start_training_process(sequences_dict, num_users, num_items):
    """
    核心训练逻辑，被离线训练和文件训练共享。
    """
    dataset = SequenceDataset(sequences_dict, SEQ_LEN)
    if len(dataset) == 0:
        print(
            "No training samples were generated. This might be because all user sequences are shorter than SEQ_LEN+1.")
        print("Try reducing SEQ_LEN. Exiting.")
        return False

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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

    create_dirs()  # 确保目录存在
    model_path = os.path.join(CHECKPOINT_DIR, 'model.pth')
    embeddings_path = os.path.join(CHECKPOINT_DIR, 'item_embeddings.npy')

    torch.save(model.state_dict(), model_path)
    print(f"Model state dict saved to {model_path}")

    item_embeddings = model.item_embedding.weight.detach().cpu().numpy()[1:]
    np.save(embeddings_path, item_embeddings)
    print(f"Item embeddings saved to {embeddings_path}")
    return True


# --- 新增：离线训练专用函数 ---
def offline_train(user_behaviors):
    """
    专用于离线训练的函数，由后端触发。
    """
    print("--- Starting Offline Training Job ---")
    sequences_dict, num_users, num_items = preprocess_data_from_backend(user_behaviors)
    if sequences_dict is None:
        print("Offline training aborted due to data processing error.")
        return False

    success = start_training_process(sequences_dict, num_users, num_items)
    print("--- Offline Training Job Finished ---")
    return success


# --- 原有的主函数，用于直接运行文件进行训练 ---
def main_train_from_file():
    """
    主训练函数 (从文件)。
    """
    sequences_dict, num_users, num_items = preprocess_data_from_file(DATA_PATH)
    if sequences_dict is None:
        print("Training from file aborted due to data processing error.")
        return
    start_training_process(sequences_dict, num_users, num_items)


if __name__ == '__main__':
    main_train_from_file()
