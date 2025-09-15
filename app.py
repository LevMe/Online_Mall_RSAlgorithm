import json
import torch
import numpy as np
from flask import Flask, request, jsonify
from model import RecommendationModel
import os

os.environ['KMP_DUplicate_LIB_OK']='True'

# --- 1. 配置与全局变量 ---
MODEL_PATH = 'checkpoint/model.pth'
ITEM_EMBEDDINGS_PATH = 'checkpoint/item_embeddings.npy'
USER_MAP_PATH = 'maps/user_map.json'
ITEM_MAP_PATH = 'maps/item_map.json'
SEQ_LEN = 15  # 必须与训练时一致
EMBEDDING_DIM = 64  # 必须与训练时一致

app = Flask(__name__)

model = None
item_embeddings = None
user_map = None
item_map = None
rev_item_map = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 2. 服务初始化 (V2) ---
def load_artifacts():
    global model, item_embeddings, user_map, item_map, rev_item_map

    print("Loading artifacts...")
    with open(USER_MAP_PATH, 'r') as f:
        user_map = json.load(f)
    with open(ITEM_MAP_PATH, 'r') as f:
        item_map = json.load(f)

    # 反向映射: item_idx -> item_id。注意索引从1开始
    rev_item_map = {v: k for k, v in item_map.items()}

    num_users = len(user_map)
    num_items = len(item_map)

    # 模型初始化数量要+1以包含padding token
    model = RecommendationModel(
        num_users=num_users + 1,
        num_items=num_items + 1,
        embedding_dim=EMBEDDING_DIM,
        seq_len=SEQ_LEN
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded.")

    # item_embeddings.npy 中不包含padding位，其索引0对应原始item_idx 1
    item_embeddings = np.load(ITEM_EMBEDDINGS_PATH)
    print("Item embeddings loaded.")
    print("Artifacts loaded successfully.")


load_artifacts()


# --- 3. API 接口定义 (V2) ---
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON input"}), 400

    user_id = data.get('user_id')
    recent_clicks = data.get('recent_clicks', [])
    top_k = data.get('top_k', 10)

    # 检查是否已加载必要的资源
    if user_map is None or item_map is None or model is None or item_embeddings is None:
        return jsonify({"error": "Service not properly initialized. Please check server logs."}), 500

    if not user_id or user_id not in user_map:
        return jsonify({"error": f"User '{user_id}' not found."}), 404

    user_idx = user_map[user_id]

    # 转换点击序列，使用.get()避免KeyError
    click_indices = [item_map.get(item_id) for item_id in recent_clicks]
    click_indices = [idx for idx in click_indices if idx is not None]

    # 准备模型输入: 填充序列，使用0作为padding value
    padded_seq = np.zeros(SEQ_LEN, dtype=np.int64)
    if len(click_indices) > 0:
        seq_start = max(0, len(click_indices) - SEQ_LEN)
        padded_seq[-len(click_indices[seq_start:]):] = click_indices[seq_start:]

    user_tensor = torch.tensor([[user_idx]], dtype=torch.long).to(device)
    seq_tensor = torch.tensor([padded_seq], dtype=torch.long).to(device)

    # 生成用户Embedding
    with torch.no_grad():
        user_embedding = model.get_user_embedding(user_tensor, seq_tensor).squeeze().cpu().numpy()

    if np.linalg.norm(user_embedding) == 0:
        return jsonify({"message": "Could not generate a recommendation for this user.", "recommended_item_ids": []})

    # 计算相似度
    user_emb_norm = user_embedding / np.linalg.norm(user_embedding)
    item_embeds_norm = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)

    scores = np.dot(item_embeds_norm, user_emb_norm)

    # 过滤掉已点击的item
    # 注意: item_embeddings的索引0对应item_idx 1
    for item_idx in click_indices:
        if (item_idx - 1) < len(scores):
            scores[item_idx - 1] = -np.inf

    # 获取Top-K
    # +1是因为item_map的索引从1开始
    top_indices = np.argsort(-scores)[:top_k]

    recommended_item_ids = [rev_item_map[idx + 1] for idx in top_indices]

    return jsonify({"recommended_item_ids": recommended_item_ids})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
