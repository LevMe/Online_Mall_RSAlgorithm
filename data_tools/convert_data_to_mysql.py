# coding: utf-8 
"""
@Time    : 2025/9/21 20:32
@Author  : Y.H LEE
"""
import pandas as pd
from sqlalchemy import create_engine, text
import time

# --- 1. 配置参数 ---
TSV_FILE_PATH = '../Datasets/douban_music/douban_music.tsv'  # <-- 输入的TSV文件路径
USER_EVENTS_TABLE = 'user_behaviors'  # <-- 用户事件表的表名
PRODUCTS_TABLE = 'products'           # <-- 【新增】产品表的表名
TOP_K = 300  # <-- 设置K值

# --- 数据库连接配置 ---
# SQLite
# DB_URI = 'sqlite:///history_data.db'


# MySQL (示例)
DB_URI = 'mysql+mysqlconnector://root:10086111@localhost/online_mall'


def update_product_ids(engine, top_k_item_ids, products_table_name):
    """
    【增加诊断日志】使用大数偏移法，安全地更新product.id。
    """
    print("\n--- 步骤 2: 更新产品ID ---")

    top_k_ids_list = sorted(top_k_item_ids, reverse=True)

    # 【调试】打印排序后的TSV ItemID，确认数据源
    print("=" * 40)
    print(f"【调试】从TSV获取的Top-K ID (降序前5): {top_k_ids_list[:5]}")
    print(f"【调试】从TSV获取的Top-K ID (降序后5): {top_k_ids_list[-5:]}")
    print("=" * 40)

    try:
        with engine.begin() as conn:  # 自动事务管理
            # 1. 从数据库读取ID
            query = text(f"SELECT id FROM {products_table_name} ORDER BY id DESC")
            db_ids_df = pd.read_sql(query, conn)

            if db_ids_df.empty:
                print(f"错误：数据库中的 '{products_table_name}' 表为空。")
                return False

            # 【调试】打印排序后的数据库ID，确认数据源
            db_ids_list = db_ids_df['id'].tolist()
            print(f"【调试】从数据库获取的Product ID (降序前5): {db_ids_list[:5]}")
            print(f"【调试】从数据库获取的Product ID (降序后5): {db_ids_list[-5:]}")
            print("=" * 40)

            # 2. 检查数量匹配
            if len(top_k_ids_list) != len(db_ids_df):
                print("错误：Top-K商品数量与数据库中的产品数量不匹配！")
                # ... (此部分代码不变)
                return False

            # 计算偏移量 (此部分代码不变)
            max_db_id = db_ids_df['id'].max()
            max_new_id = max(top_k_ids_list)
            offset = int(max(max_db_id, max_new_id) + 1)

            # 3. 创建映射DataFrame
            mapping_df = pd.DataFrame({
                'old_id': db_ids_df['id'],
                'new_id': top_k_ids_list
            })

            # 【！！！关键调试步骤！！！】 打印最终生成的映射关系
            print("\n【调试】生成的最终ID映射关系 (前5条和后5条):")
            # 使用to_string()可以更好地对齐显示
            print(mapping_df.head(5).to_string())
            if len(mapping_df) > 5:
                print("...")
                print(mapping_df.tail(5).to_string())
            print("=" * 40 + "\n")

            # 4. 写入临时表并执行更新 (此部分代码不变)
            temp_table_name = "temp_product_id_mapping"
            mapping_df.to_sql(temp_table_name, conn, if_exists='replace', index=False)

            update_step1_sql = text(f"""
                UPDATE {products_table_name} p
                JOIN {temp_table_name} t ON p.id = t.old_id
                SET p.id = p.id + :offset; 
            """)
            conn.execute(update_step1_sql, {"offset": offset})

            update_step2_sql = text(f"""
                UPDATE {products_table_name} p
                JOIN {temp_table_name} t ON p.id = (t.old_id + :offset)
                SET p.id = t.new_id;
            """)
            result = conn.execute(update_step2_sql, {"offset": offset})

            conn.execute(text(f"DROP TABLE IF EXISTS {temp_table_name}"))

            # 我们需要再次确认这里的rowcount
            print(f"成功！ {result.rowcount} 个产品ID已更新。")

        return True

    except Exception as e:
        print(f"数据库操作失败: {e}")
        return False


def process_and_insert_user_events(file_path, db_uri, events_table_name, products_table_name, k):
    """
    【流程已修改】
    1. 计算Top-K商品。
    2. 用Top-K商品ID更新products表。
    3. 插入Top-K商品的交互记录。
    """
    print("--- 开始数据处理总流程 ---")
    start_time = time.time()

    try:
        # --- 步骤 1: 读取数据并计算Top-K热门商品 ---
        print("\n--- 步骤 1: 计算Top-K热门商品 ---")
        df = pd.read_csv(file_path, sep='\t', header=None, names=['UserId', 'ItemId', 'Rating', 'Timestamp'])
        print(f"成功读取 {len(df)} 条总用户事件记录。")

        print(f"正在计算 Top {k} 热门商品...")
        # .index.values 会返回一个numpy数组
        top_k_items = df['ItemId'].value_counts().nlargest(k).index.values

        # --- 步骤 2: 连接数据库，并调用函数更新产品ID ---
        engine = create_engine(db_uri)
        update_success = update_product_ids(engine, top_k_items, products_table_name)

        if not update_success:
            print("\n产品ID更新失败，整个流程已中止。")
            return

        # --- 步骤 3: 筛选并插入Top-K商品的用户事件 ---
        print(f"\n--- 步骤 3: 向 '{events_table_name}' 表插入用户事件 ---")
        df_filtered = df[df['ItemId'].isin(top_k_items)].copy()
        print(f"筛选出 {len(df_filtered)} 条热门商品的用户事件待入库。")

        # 准备待插入的数据
        df_filtered.rename(columns={
            'UserId': 'user_id',
            'ItemId': 'product_id',
            'Timestamp': 'timestamp'
        }, inplace=True)
        df_filtered['event_type'] = 'click'
        df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'], unit='s')

        df_to_insert = df_filtered[['user_id', 'product_id', 'event_type', 'timestamp']]

        # 使用 to_sql 插入数据
        print("正在批量插入用户事件...")
        df_to_insert.to_sql(
            events_table_name,
            con=engine,
            if_exists='append',
            index=False,
            chunksize=10000
        )

        end_time = time.time()
        print(f"\n成功！ {len(df_to_insert)} 条用户事件记录已写入 '{events_table_name}' 表。")
        print(f"数据处理总耗时: {end_time - start_time:.2f} 秒。")

    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")


if __name__ == '__main__':
    process_and_insert_user_events(TSV_FILE_PATH, DB_URI, USER_EVENTS_TABLE, PRODUCTS_TABLE, TOP_K)