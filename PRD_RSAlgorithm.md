# **Prompt: 开发基于PyTorch的个性化推荐算法服务**

## **1\. 核心任务**

你需要为我开发一套完整的、基于深度学习的个性化推荐算法。这套算法将作为独立的服务，通过 API 与在线商城后端进行交互。

**项目技术栈**: Python, PyTorch, Flask, Numpy

**最终交付物应包含三个独立的Python脚本**:

1. model.py: 定义推荐模型的网络结构。  
2. train.py: 负责数据处理、模型训练和产出物保存。  
3. app.py: 基于 Flask 框架，提供在线推荐的 API 服务。

## **2\. 算法与模型设计 (model.py)**

你需要设计一个能够同时捕捉用户**长期和短期兴趣**的深度学习模型。

### **2.1 模型架构**

该模型输入用户的 user\_id 和他最近互动的 item\_sequence（商品ID序列），输出一个代表用户当前兴趣的 user\_embedding。

具体结构如下：

1. **Embedding层**:  
   * User Embedding Layer: 将 user\_id 映射为一个固定维度的向量，代表用户的**长期兴趣**。  
   * Item Embedding Layer: 将 item\_id 映射为一个固定维度的向量。该层权重将在模型训练后，作为商品特征库直接用于应用阶段。  
2. **短期兴趣捕获模块 (Transformer)**:  
   * 输入用户最近点击的商品序列（例如，最近的20个商品ID）。  
   * 将序列中的 Item ID 通过 Item Embedding Layer 转换为向量序列。  
   * 添加**位置编码 (Positional Encoding)**，让模型理解商品的先后顺序。  
   * 将处理后的向量序列输入一个 **Transformer Encoder** 层。Transformer 的自注意力机制可以有效捕捉序列中的关键商品，生成一个代表用户**短期兴趣**的上下文向量。  
3. **输出**:  
   * **训练阶段**: 模型的训练任务是\*\*“下一次点击预测”\*\*。即根据用户 \[item\_1, ..., item\_n-1\] 的序列，预测下一个 item\_n。你需要用序列的 Transformer 输出去和所有商品的 Item Embedding 计算一个相似度分数（通常是内积），然后通过 Softmax 得到预测概率。  
   * **推理阶段**: 模型需要提供一个方法 get\_user\_embedding，该方法将 User Embedding (长期) 和 Transformer输出的序列Embedding (短期) 进行结合（例如，通过相加或拼接），生成最终的、统一的 user\_embedding。

## **3\. 离线训练流程 (train.py)**

该脚本负责模型的离线训练和相关资产的保存。

### **3.1 数据源**

* 假设的输入数据是一个 ratings.csv 文件，包含三列: user\_id, item\_id, timestamp。

### **3.2 数据预处理**

1. **ID 映射**: 为 user\_id 和 item\_id 创建映射表，将它们转换为从 0 开始的连续整数索引。请将这些映射表保存为 JSON 文件（例如 user\_map.json, item\_map.json）。  
2. **序列生成**: 按 user\_id 分组，并按 timestamp 对每个用户的行为进行排序，生成每个用户的完整行为序列。  
3. **数据集构建**:  
   * 使用**滑动窗口**方法从每个用户的行为序列中生成训练样本。例如，对于一个长序列 \[i1, i2, i3, i4, i5\]，可以生成样本：  
     * input: \[i1\], label: i2  
     * input: \[i1, i2\], label: i3  
     * input: \[i1, i2, i3\], label: i4  
     * ...  
   * 使用 PyTorch 的 Dataset 和 DataLoader 来高效地加载数据。

### **3.3 模型训练**

1. **损失函数**: 使用 CrossEntropyLoss (交叉熵损失)。  
2. **优化器**: 使用 Adam 或 AdamW。  
3. **训练循环**: 实现标准 PyTorch 训练循环，包含模型的前向传播、损失计算、反向传播和参数更新。

### **3.4 产出物保存**

训练完成后，必须保存以下三个关键产出物：

1. **模型状态字典**: torch.save(model.state\_dict(), 'model.pth')  
2. **商品 Embedding 矩阵**: 从训练好的模型中提取 Item Embedding Layer 的权重，并保存为一个 Numpy 文件 item\_embeddings.npy。  
3. **ID 映射表**: 在预处理阶段生成的 user\_map.json 和 item\_map.json。

## **4\. 在线应用服务 (app.py)**

这是一个基于 Flask 的 Web 服务，用于提供实时的推荐结果。

### **4.1 服务初始化**

1. 在服务启动时，一次性加载 model.pth, item\_embeddings.npy, user\_map.json, item\_map.json 到内存中。  
2. 将模型设置为评估模式 model.eval()。

### **4.2 API 接口定义**

* **路径**: POST /recommend  
* **请求体 (JSON)**:  
  {  
    "user\_id": "some\_user\_id\_123",  
    "recent\_clicks": \["item\_id\_A", "item\_id\_B", "item\_id\_C"\],  
    "top\_k": 10  
  }

* **成功响应 (JSON)**:  
  {  
    "recommended\_item\_ids": \["item\_id\_X", "item\_id\_Y", "item\_id\_Z", ...\]  
  }

### **4.3 推荐逻辑**

1. 接收到请求后，使用 ID 映射表将请求中的 user\_id 和 recent\_clicks 列表转换为模型内部的索引。  
2. 调用模型的 get\_user\_embedding 方法，生成该用户的实时 Embedding。  
3. 使用该 user\_embedding 与内存中加载的**所有** item\_embeddings 计算余弦相似度。  
4. 根据相似度得分从高到低排序，选出 Top-K 个商品。  
5. 将这些商品索引转换回原始的 item\_id，并作为响应返回。  
6. **代码中请注明**：对于大规模商品（例如超过10万），可以使用 FAISS 库来加速相似度搜索，但在此次实现中，使用 numpy 或 torch 的矩阵运算即可。

## **5\. 全局要求**

* **代码注释**: 为所有函数、类和关键逻辑块编写清晰、详细的中文注释。  
* **模块化**: 确保代码结构清晰，遵循 Python 编程规范。  
* **环境依赖**: 在项目开头提供一个 requirements.txt 的示例，列出所有必要的库（pytorch, flask, numpy等）。

