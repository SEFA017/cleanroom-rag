#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import sys
from core.config import PROJECT_ROOT

# ========== 配置 ==========

IMG_DB_DIR =  os.path.join(PROJECT_ROOT, "data", "vector", "img")
DOC_PATH  = os.path.join(IMG_DB_DIR, "document.json")
META_PATH = os.path.join(IMG_DB_DIR, "metadata.json")
VEC_PATH  = os.path.join(IMG_DB_DIR, "vectors.json")
print(DOC_PATH, META_PATH, VEC_PATH)
PHRASE1 = "抱歉，我目前还没有修改图片的能力"
PHRASE2 = "<|observation|>"
# ========================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def len_check():
    """检查三个文件长度是否一致"""
    try:
        documents = load_json(DOC_PATH)
        metadata  = load_json(META_PATH)
        vectors   = load_json(VEC_PATH)
    except FileNotFoundError as e:
        print(f"Error: 未找到文件，{e}")
        sys.exit(1)
    if not (len(documents) == len(metadata) == len(vectors)):
        print("Error: 三个文件长度不一致，无法保证索引对齐。")
        sys.exit(1)
    print(len(documents), len(metadata), len(vectors))

def main():
    # 1. 加载现有数据库
    try:
        documents = load_json(DOC_PATH)
        metadata  = load_json(META_PATH)
        vectors   = load_json(VEC_PATH)
    except FileNotFoundError as e:
        print(f"Error: 未找到文件，{e}")
        sys.exit(1)

    if not (len(documents) == len(metadata) == len(vectors)):
        print("Error: 三个文件长度不一致，无法保证索引对齐。")
        sys.exit(1)

    # 2. 自动检测要删除的图片路径
    to_delete = set()
    for doc, meta in zip(documents, metadata):
        if PHRASE1 in doc or PHRASE2 in doc:
            img_path = meta.get("image_path")
            if img_path:
                to_delete.add(img_path)

    if to_delete:
        print("检测到以下条目包含自动删除短语，将被移除：")
        for p in to_delete:
            print(f"  {p}")
    else:
        print("未检测到自动删除短语，无需自动清理。")

    # 3. 交互式删除——单张图片
    user_image = input("请输入要手动删除的图片路径（回车跳过）：").strip()
    if user_image:
        to_delete.add(user_image)
        print(f"已标记用户指定图片为删除：\n  {user_image}")

    # 4. 交互式删除——指定文件夹下所有图片
    user_folder = input("请输入要删除的图片所在文件夹路径（回车跳过）：").strip()
    if user_folder:
        # 规范化
        folder_norm = os.path.normpath(user_folder)
        folder_norm = folder_norm if folder_norm.endswith(os.sep) else folder_norm + os.sep
        matched = [meta.get("image_path") for meta in metadata
                   if meta.get("image_path", "").startswith(folder_norm)]
        if matched:
            print("检测到以下在指定文件夹下的图片，将被移除：")
            for p in matched:
                print(f"  {p}")
                to_delete.add(p)
        else:
            print("在指定文件夹下未检测到任何图片路径，无需删除。")

    if not to_delete:
        print("没有要删除的条目，退出。")
        return

    # 5. 重新过滤并删除
    new_docs, new_meta, new_vecs = [], [], []
    for doc, meta, vec in zip(documents, metadata, vectors):
        path = meta.get("image_path")
        if path in to_delete:
            print(f"删除条目 -> {path}")
            continue
        new_docs.append(doc)
        new_meta.append(meta)
        new_vecs.append(vec)

    # 6. 覆写回磁盘
    save_json(DOC_PATH, new_docs)
    save_json(META_PATH, new_meta)
    save_json(VEC_PATH, new_vecs)

    print("数据库更新完成。")

if __name__ == "__main__":
    main()
    # len_check()