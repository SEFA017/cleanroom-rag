import os
import sys
from core.config import PROJECT_ROOT
sys.path.append(PROJECT_ROOT)

from core.img_storage_prepare import process_images
from core.text_storage_prepare import text_vector_store

if __name__ == "__main__":

    IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "data", "database", "20250424", "img")
    OUTPUT_MD = os.path.join(PROJECT_ROOT, "data", "database", "20250424", "img", "summary.md")
    img_vector_store_path = os.path.join(PROJECT_ROOT, "data", "vector", "img")
    # process_images(IMAGE_FOLDER, OUTPUT_MD, img_vector_store_path, os.cpu_count() * 2)

    # 处理文本数据
    TEXT_FOLDER = os.path.join(PROJECT_ROOT, "data", "database", "20250424", "md")
    text_vector_store_path = os.path.join(PROJECT_ROOT, "data", "vector", "text")
    # print(f"TEXT_FOLDER: {TEXT_FOLDER}")
    # text_vector_store(TEXT_FOLDER, text_vector_store_path)
