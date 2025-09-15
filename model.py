import torch
import torch.nn as nn
import math


class RecommendationModel(nn.Module):
    """
    基于Transformer的个性化推荐模型 (V2)。
    此版本明确处理了padding，并优化了模型结构。
    """

    def __init__(self, num_users, num_items, embedding_dim=64, seq_len=20, nhead=4, num_layers=2, dropout=0.1):
        """
        模型初始化。

        参数:
            num_users (int): 用户总数 (包括 padding token)。
            num_items (int): 商品总数 (包括 padding token)。
            embedding_dim (int): Embedding向量的维度。
            seq_len (int): 输入行为序列的固定长度。
            nhead (int): Transformer Encoder中的多头注意力头数。
            num_layers (int): Transformer Encoder的层数。
            dropout (float): Dropout的比率。
        """
        super(RecommendationModel, self).__init__()
        self.embedding_dim = embedding_dim

        # 1. Embedding层
        # 用户长期兴趣Embedding。索引0为padding，不参与训练。
        self.user_embedding = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        # 商品Embedding。索引0为padding，不参与训练。
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        # 位置编码
        self.pos_embedding = nn.Embedding(seq_len, embedding_dim)

        # 2. 短期兴趣捕获模块 (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 输出层
        self.fc = nn.Linear(embedding_dim, num_items)

        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        """初始化模型参数"""
        initrange = 0.1
        self.user_embedding.weight.data.uniform_(-initrange, initrange)
        self.item_embedding.weight.data.uniform_(-initrange, initrange)
        self.pos_embedding.weight.data.uniform_(-initrange, initrange)
        # 保证padding token的embedding为0向量
        self.user_embedding.weight.data[0].zero_()
        self.item_embedding.weight.data[0].zero_()

        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, user_id, item_seq):
        """
        模型的前向传播，用于训练阶段。

        参数:
            user_id (torch.Tensor): 用户ID的Tensor，形状为 (batch_size, 1)。
            item_seq (torch.Tensor): 用户行为商品序列的Tensor，形状为 (batch_size, seq_len)。

        返回:
            torch.Tensor: 对下一个商品的预测得分，形状为 (batch_size, num_items)。
        """
        # 获取商品序列的Embedding
        item_embeds = self.item_embedding(item_seq)  # (batch_size, seq_len, embedding_dim)

        # 创建位置ID并获取位置编码
        positions = torch.arange(0, item_seq.size(1), device=item_seq.device).unsqueeze(0)
        pos_embeds = self.pos_embedding(positions)  # (1, seq_len, embedding_dim)

        seq_embedding = self.dropout(item_embeds + pos_embeds)

        # Transformer编码
        # 在V2版本中，由于输入是固定长度的真实序列，不再需要mask
        transformer_output = self.transformer_encoder(seq_embedding)  # (batch_size, seq_len, embedding_dim)

        # 提取序列最后一个位置的输出用于预测
        last_item_embedding = transformer_output[:, -1, :]  # (batch_size, embedding_dim)

        logits = self.fc(last_item_embedding)

        return logits

    def get_user_embedding(self, user_id, item_seq):
        """
        获取用户的组合Embedding，用于推理阶段。

        参数:
            user_id (torch.Tensor): 用户ID的Tensor，形状为 (1, 1)。
            item_seq (torch.Tensor): 填充后的用户行为商品序列，形状为 (1, seq_len)。

        返回:
            torch.Tensor: 代表用户当前综合兴趣的Embedding，形状为 (1, embedding_dim)。
        """
        with torch.no_grad():
            # 1. 长期兴趣Embedding
            long_term_embedding = self.user_embedding(user_id).squeeze(0)

            # 2. 短期兴趣Embedding
            item_embeds = self.item_embedding(item_seq)
            positions = torch.arange(0, item_seq.size(1), device=item_seq.device).unsqueeze(0)
            pos_embeds = self.pos_embedding(positions)

            seq_embedding = item_embeds + pos_embeds

            # 生成attention mask，忽略padding部分
            padding_mask = (item_seq == 0)  # 形状: (1, seq_len)

            transformer_output = self.transformer_encoder(seq_embedding, src_key_padding_mask=padding_mask)

            # 使用序列输出的平均值作为短期兴趣的代表 (忽略padding)
            # 计算非padding部分的数量
            non_pad_count = (item_seq != 0).sum(dim=1, keepdim=True)
            sum_embeds = transformer_output.sum(dim=1)
            # 防止除以0
            short_term_embedding = sum_embeds / non_pad_count.clamp(min=1)

            # 3. 结合长期和短期兴趣
            combined_embedding = long_term_embedding + short_term_embedding

            return combined_embedding

