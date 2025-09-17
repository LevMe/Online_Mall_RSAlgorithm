# coding: utf-8 
"""
@Time    : 2025/9/17 21:00
@Author  : Y.H LEE
"""
import random
import json
from datetime import datetime, timedelta

categories = {
    1: "Electronics",
    2: "Books",
    3: "Home",
    4: "Phones",
    5: "Computers"
}

# 商品名称池（中英文对应，避免重复）
products = [
    ("高性能游戏本", "Gaming Laptop"),
    ("无线蓝牙耳机", "Bluetooth Earbuds"),
    ("人体工学办公椅", "Office Chair"),
    ("畅销小说套装", "Novel Set"),
    ("旗舰智能手机", "Smartphone"),
    ("4K显示器", "4K Monitor"),
    ("机械键盘", "Mechanical Keyboard"),
    ("智能手表", "Smart Watch"),
    ("扫地机器人", "Robot Vacuum"),
    ("电动牙刷", "Electric Toothbrush"),
    ("路由器", "WiFi Router"),
    ("咖啡机", "Coffee Machine"),
    ("平板电脑", "Tablet"),
    ("空气净化器", "Air Purifier"),
    ("单反相机", "DSLR Camera"),
    ("小说精选", "Novel Collection"),
    ("音响系统", "Speaker System"),
    ("台灯", "Desk Lamp"),
    ("折叠屏手机", "Foldable Phone"),
    ("电竞椅", "Gaming Chair"),
]

# 扩展成 300 个（通过加编号确保唯一）
items = []
for i in range(15):  # 20 * 15 = 300
    for cn, en in products:
        items.append((f"{cn}{i+1}", f"{en}{i+1}"))

random.shuffle(items)

def random_time():
    base = datetime(2025, 8, 20)
    delta = timedelta(days=random.randint(0, 20), hours=random.randint(0, 23), minutes=random.randint(0, 59))
    created = base + delta
    updated = created + timedelta(hours=random.randint(1, 10))
    return created.strftime("%Y-%m-%d %H:%M:%S"), updated.strftime("%Y-%m-%d %H:%M:%S")

with open("insert_products.sql", "w", encoding="utf-8") as f:
    for i, (cn, en) in enumerate(items[:300], start=1):
        price = round(random.uniform(99, 9999), 2)
        stock = random.randint(10, 100)
        sales = random.randint(0, 30)
        category_id = random.randint(1, 5)

        # specs 根据类别变化
        if category_id == 1:  # 电子产品
            specs = {"Battery": f"{random.randint(10,30)}h", "Warranty": "2 years"}
        elif category_id == 2:  # 图书
            specs = {"Author": "Various", "Pages": random.randint(200,1000)}
        elif category_id == 3:  # 家居
            specs = {"Material": random.choice(["Wood", "Metal", "Plastic"]), "Weight": f"{random.randint(2,20)}kg"}
        elif category_id == 4:  # 手机
            specs = {"CPU": "Snapdragon 8 Gen 3", "RAM": f"{random.choice([8,12,16])}GB", "Storage": f"{random.choice([128,256,512])}GB"}
        else:  # 电脑
            specs = {"CPU": "Intel Core i7", "GPU": "NVIDIA RTX 4070", "RAM": "16GB"}

        main_img = f"https://placehold.co/600x400/{random.choice(['FF3B30','34C759','5856D6','FF9500','FF2D55'])}/FFFFFF?text={en.replace(' ','+')}"
        image_urls = json.dumps([main_img])

        created_at, updated_at = random_time()

        sql = f"""INSERT INTO products (id, name, description, price, stock, sales, category_id, main_image_url, image_urls, specs, created_at, updated_at) VALUES ({i}, '{cn}', '自动生成的商品描述。', {price}, {stock}, {sales}, {category_id}, '{main_img}', '{image_urls}', '{json.dumps(specs, ensure_ascii=False)}', '{created_at}', '{updated_at}');\n"""
        f.write(sql)

print("✅ 已生成 insert_products.sql 文件，共 300 条数据。")
