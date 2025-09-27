# coding: utf-8 
"""
@Time    : 2025/9/27 19:21
@Author  : Y.H LEE
"""
# data_tools/mysql_to_clickhouse_migrator.py
import pandas as pd
from sqlalchemy import create_engine
from clickhouse_driver import Client
import time

# --- 1. 数据库连接配置 (请根据你的实际情况修改) ---

# MySQL 连接配置
MYSQL_USER = 'root'
MYSQL_PASSWORD = '10086111'
MYSQL_HOST = 'localhost'
MYSQL_PORT = 3306
MYSQL_DB = 'online_mall'

# ClickHouse 连接配置
CLICKHOUSE_HOST = 'localhost'
CLICKHOUSE_PORT = 9000
CLICKHOUSE_DB = 'online_mall'
CLICKHOUSE_USER = 'default'
CLICKHOUSE_PASSWORD = '10086111'  # 如果你设置了密码，请填写

# --- 2. 迁移参数配置 ---

# 从 MySQL 读取数据的表名
SOURCE_TABLE = 'user_behaviors'
# 写入 ClickHouse 的表名
TARGET_TABLE = 'user_behaviors'

# 每次从 MySQL 读取并写入 ClickHouse 的数据行数
CHUNK_SIZE = 50000


def migrate_data():
    """
    将 MySQL 表中的数据分块迁移到 ClickHouse。
    """
    print("--- 开始数据迁移任务 ---")
    start_time = time.time()

    # --- 步骤 1: 创建数据库连接 ---
    try:
        mysql_uri = f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
        mysql_engine = create_engine(mysql_uri)
        print("成功连接到 MySQL。")

        clickhouse_client = Client(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            database=CLICKHOUSE_DB,
            user=CLICKHOUSE_USER,
            password=CLICKHOUSE_PASSWORD
        )
        clickhouse_client.execute('SELECT 1')
        print("成功连接到 ClickHouse。")

    except Exception as e:
        print(f"数据库连接失败: {e}")
        return

    # --- 步骤 2: (可选) 清空 ClickHouse 目标表 ---
    # print(f"正在清空 ClickHouse 表 '{TARGET_TABLE}'...")
    # clickhouse_client.execute(f'TRUNCATE TABLE {TARGET_TABLE}')

    # --- 步骤 3: 分块读取和写入 ---
    total_rows_migrated = 0
    chunk_num = 1

    # ===================================================================
    # ✨✨✨ 核心修改在这里：增加了 ORDER BY timestamp ✨✨✨
    # ===================================================================
    sql_query = f"SELECT user_id, product_id, event_type, timestamp FROM {SOURCE_TABLE} ORDER BY timestamp"

    try:
        for chunk_df in pd.read_sql(sql_query, mysql_engine, chunksize=CHUNK_SIZE):
            print(f"--- 正在处理第 {chunk_num} 批数据 (大小: {len(chunk_df)}行) ---")

            data_to_insert = chunk_df.to_dict('records')

            insert_query = f'INSERT INTO {TARGET_TABLE} (user_id, product_id, event_type, timestamp) VALUES'
            clickhouse_client.execute(insert_query, data_to_insert, types_check=True)

            rows_in_chunk = len(chunk_df)
            total_rows_migrated += rows_in_chunk
            print(f"成功插入 {rows_in_chunk} 行数据。累计已迁移: {total_rows_migrated} 行。")
            chunk_num += 1

    except Exception as e:
        print(f"\n在处理过程中发生错误: {e}")
        print("迁移任务已中断。")
        return

    end_time = time.time()
    print("\n--- 数据迁移任务完成 ---")
    print(f"总计迁移了 {total_rows_migrated} 行数据。")
    print(f"总耗时: {end_time - start_time:.2f} 秒。")


if __name__ == '__main__':
    migrate_data()