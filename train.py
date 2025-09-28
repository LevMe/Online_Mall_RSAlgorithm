import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from clickhouse_driver import Client
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
# --- ClickHouse 配置 ---
CLICKHOUSE_HOST = 'localhost'  # 你的 ClickHouse 主机
CLICKHOUSE_PORT = 9000  # 你的 ClickHouse 端口
CLICKHOUSE_DB = 'online_mall'  # 你的数据库名
CLICKHOUSE_USER = 'default'  # <-- 新增
CLICKHOUSE_PASSWORD = '10086111'  # <-- 新增 (请确保与您的配置一致)

print(f"Using device: {DEVICE}")


# --- 辅助函数：创建目录 ---
def create_dirs():
    """确保所有必要的目录都存在"""
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    if not os.path.exists(MAPS_DIR):
        os.makedirs(MAPS_DIR)


def fetch_data_from_clickhouse():
    """
    从 ClickHouse 中获取所有的用户行为数据。
    """
    print("Connecting to ClickHouse to fetch user behaviors...")
    try:
        # 修改这里，加入 user 和 password
        client = Client(host=CLICKHOUSE_HOST, port=CLICKHOUSE_PORT, database=CLICKHOUSE_DB, user=CLICKHOUSE_USER,
                        password=CLICKHOUSE_PASSWORD)
        # 你可以增加 WHERE 条件来筛选特定的时间范围
        query = "SELECT user_id, product_id, toUnixTimestamp(timestamp) as timestamp FROM user_behaviors"

        data = client.execute(query, with_column_types=True)

        columns = [col[0] for col in data[1]]
        df = pd.DataFrame(data[0], columns=columns)

        print(f"Successfully fetched {len(df)} records from ClickHouse.")
        return df
    except Exception as e:
        print(f"Error fetching data from ClickHouse: {e}")
        return None


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


def _process_dataframe(df):
    """
    DataFrame处理的核心逻辑，被两个预处理函数共享。
    """
    # 确保列名是 item_id, user_id, timestamp
    if 'product_id' in df.columns:
        df.rename(columns={'product_id': 'item_id'}, inplace=True)

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
def offline_train():
    """
    专用于离线训练的函数，由后端触发。
    现在它会直接从ClickHouse拉取数据。
    """
    print("--- Starting Offline Training Job ---")

    # 1. 从 ClickHouse 获取数据
    df = fetch_data_from_clickhouse()
    if df is None or df.empty:
        print("Offline training aborted: no data fetched from ClickHouse.")
        return False

    # 2. 处理 DataFrame
    sequences_dict, num_users, num_items = _process_dataframe(df)
    if sequences_dict is None:
        print("Offline training aborted due to data processing error.")
        return False

    # 3. 开始训练
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
    offline_train()
