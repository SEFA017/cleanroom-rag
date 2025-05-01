import os
import json
from typing import List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.config import PROJECT_ROOT
from core.Embeddings import ZhipuEmbedding
from core.VectorBase import VectorStore
from dotenv import dotenv_values
from core.LLM import ImageSummaryGLM

# 加载环境变量
env = dotenv_values('.env')
for k, v in env.items():
    os.environ[k] = v

api_key = "your_api_key_here"  # 替换为你的API密钥


class ImageProcessor:
    @staticmethod
    def get_image_files(folder: str) -> List[str]:
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        return [
            os.path.join(root, f)
            for root, _, files in os.walk(folder)
            for f in files
            if os.path.splitext(f)[1].lower() in valid_ext
        ]


def process_single_image(args):
    img_path, glm_vision, embedding = args
    try:
        summary = glm_vision.generate_summary(img_path)
        rel = os.path.relpath(img_path, PROJECT_ROOT)
        if summary:
            vector = embedding.get_embedding(summary)
            return {
                "status": "success",
                "img_path": img_path,
                "summary": summary,
                "vector": vector,
                "metadata": {"image_path": rel}
            }
        print(f"摘要生成失败, 图片路径: {rel}")
        return {"status": "empty_summary", "img_path": img_path}
    except Exception as e:
        print(f"处理异常 [{rel}]: {e}")
        return {"status": "error", "img_path": img_path, "error": str(e)}


def process_images(image_folder: str, output_md: str, vector_store_path: str, workers: int = 4):
    processor = ImageProcessor()
    glm_vision = ImageSummaryGLM(api_key=api_key)
    embedding = ZhipuEmbedding(api_key=api_key)
    img_vector_store = VectorStore()

    # 加载已有路径
    meta_file = os.path.join(vector_store_path, "metadata.json")
    existing = []
    if os.path.exists(meta_file):
        existing = json.load(open(meta_file, 'r', encoding='utf-8'))
    existing_paths = {m.get("image_path") for m in existing}

    img_paths = processor.get_image_files(image_folder)
    new_paths = [p for p in img_paths if os.path.relpath(p, PROJECT_ROOT) not in existing_paths]
    print(f"发现 {len(img_paths)} 张图片，其中 {len(new_paths)} 张待处理")
    if not new_paths:
        return

    tasks = [(p, glm_vision, embedding) for p in new_paths]
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_image, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="处理图片"):
            results.append(fut.result())

    # 持久化
    if os.path.exists(os.path.join(vector_store_path, "document.json")):
        img_vector_store.load_vector_img(vector_store_path)

    md_mode = 'a' if os.path.exists(output_md) else 'w'
    with open(output_md, md_mode, encoding='utf-8') as md:
        if md_mode == 'w':
            md.write("""# 图片摘要库

| 缩略图 | 文件名 | 摘要 | 存储路径 |
|--------|--------|------|---------|
""")
        for r in results:
            rel = os.path.relpath(r['img_path'], PROJECT_ROOT)
            name = os.path.basename(r['img_path'])
            if r['status'] == 'success':
                img_vector_store.add_document(r['summary'], r['metadata'], r['vector'])
                md.write(f"| ![缩略图]({rel}) | {name} | {r['summary']} | `{rel}` |\n")
            else:
                md.write(f"| ![缩略图]({rel}) | {name} | 处理失败 | `{rel}` |\n")

    img_vector_store.persist(vector_store_path)