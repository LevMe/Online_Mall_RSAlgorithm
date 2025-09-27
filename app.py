# levme/online_mall_rsalgorithm/Online_Mall_RSAlgorithm-eafea5390b98ba9de36a038ae0c0e729bceaeac0/app.py
import json
import torch
import numpy as np
from flask import Flask, request, jsonify
from model import RecommendationModel
import os
import threading  # 引入线程模块
from train import offline_train  # 从train.py中引入离线训练函数

os.environ['KMP_DUplicate_LIB_OK'] = 'True'

# --- 1. 配置与全局变量 ---
CHECKPOINT_DIR = 'checkpoint'
MAPS_DIR = 'maps'
MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'model.pth')
ITEM_EMBEDDINGS_PATH = os.path.join(CHECKPOINT_DIR, 'item_embeddings.npy')
USER_MAP_PATH = os.path.join(MAPS_DIR, 'user_map.json')
ITEM_MAP_PATH = os.path.join(MAPS_DIR, 'item_map.json')
SEQ_LEN = 15
EMBEDDING_DIM = 64

app = Flask(__name__)

model = None
item_embeddings = None
user_map = None
item_map = None
rev_item_map = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 新增：用于防止重复训练的全局锁 ---
training_in_progress = False


# --- 2. 服务初始化 (V2) ---
def load_artifacts():
    global model, item_embeddings, user_map, item_map, rev_item_map

    print("Loading artifacts...")

    # 检查必要文件是否存在
    if not all(os.path.exists(p) for p in [MODEL_PATH, ITEM_EMBEDDINGS_PATH, USER_MAP_PATH, ITEM_MAP_PATH]):
        print(
            "Warning: One or more artifact files are missing. Service might not work correctly until a model is trained.")
        return

    with open(USER_MAP_PATH, 'r') as f:
        user_map = json.load(f)
    with open(ITEM_MAP_PATH, 'r') as f:
        item_map = json.load(f)

    rev_item_map = {v: k for k, v in item_map.items()}
    num_users = len(user_map)
    num_items = len(item_map)

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

    if user_map is None or item_map is None or model is None or item_embeddings is None:
        return jsonify({"error": "Service not properly initialized. A model needs to be trained first."}), 503

    # user_id 可能是数字或字符串，统一转换为字符串进行匹配
    user_id_str = str(user_id)
    if user_id_str not in user_map:
        return jsonify({"error": f"User '{user_id_str}' not found."}), 404

    user_idx = user_map[user_id_str]

    click_indices = [item_map.get(str(item_id)) for item_id in recent_clicks]
    click_indices = [idx for idx in click_indices if idx is not None]

    padded_seq = np.zeros(SEQ_LEN, dtype=np.int64)
    if len(click_indices) > 0:
        seq_start = max(0, len(click_indices) - SEQ_LEN)
        padded_seq[-len(click_indices[seq_start:]):] = click_indices[seq_start:]

    user_tensor = torch.tensor([user_idx], dtype=torch.long).to(device)  # 注意 user_idx 已经是单个值
    seq_tensor = torch.tensor([padded_seq], dtype=torch.long).to(device)

    with torch.no_grad():
        user_embedding = model.get_user_embedding(user_tensor, seq_tensor).squeeze().cpu().numpy()

    if np.linalg.norm(user_embedding) == 0:
        return jsonify({"message": "Could not generate a recommendation for this user.", "recommended_item_ids": []})

    user_emb_norm = user_embedding / np.linalg.norm(user_embedding)
    item_embeds_norm = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)
    scores = np.dot(item_embeds_norm, user_emb_norm)

    for item_idx in click_indices:
        if (item_idx - 1) < len(scores):
            scores[item_idx - 1] = -np.inf

    top_indices = np.argsort(-scores)[:top_k]
    recommended_item_ids = [rev_item_map.get(idx + 1) for idx in top_indices]
    recommended_item_ids = [item_id for item_id in recommended_item_ids if item_id is not None]

    return jsonify({"recommended_item_ids": recommended_item_ids})


# --- 离线训练专用接口 ---
@app.route('/trigger_training', methods=['POST'])
def trigger_training():
    global training_in_progress
    if training_in_progress:
        return jsonify({"status": "error", "message": "A training process is already running."}), 409

    # 定义一个在线程中运行的函数
    def training_job():
        global training_in_progress
        training_in_progress = True

        # 直接调用 offline_train，不再传递数据
        success = offline_train()

        if success:
            print("Training successful. Reloading artifacts...")
            load_artifacts()
        else:
            print("Training failed. Check logs for details.")

        training_in_progress = False

    thread = threading.Thread(target=training_job)
    thread.start()

    return jsonify({"status": "success", "message": "Training process started in the background."})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)