import os
import sys
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.config import PROJECT_ROOT
from core.VectorBase import VectorStore
from core.utils import ReadFiles
from core.Embeddings import ZhipuEmbedding

api_key = "your_api_key_here"  # 替换为你的API密钥


def text_vector_store(text_path: str, text_vector_path: str):
    vector_store = VectorStore()
    # 增量加载已有数据
    if os.path.exists(os.path.join(text_vector_path, "document.json")):
        vector_store.load_vector_text(text_vector_path)

    file_processor = ReadFiles(text_path)
    new_docs = []
    for file in file_processor.file_list:
        # 跳过已处理的文件
        rel_file = os.path.relpath(file, PROJECT_ROOT)
        if any(meta.get("source_file") == rel_file for meta in vector_store.metadata):
            print(f"跳过已处理文件: {rel_file}")
            continue

        content = ReadFiles.read_file_content(file)
        chunks = ReadFiles.get_chunk2(content)
        for idx, chunk in enumerate(chunks):
            new_docs.append({
                "content": chunk,
                "metadata": {
                    "source_file": rel_file,
                    "chunk_id": idx
                }
            })

    if new_docs:
        embedding = ZhipuEmbedding(api_key=api_key)
        vectors = []
        for doc in tqdm(new_docs, desc="生成文本向量"):
            vectors.append(embedding.get_embedding(doc["content"]))

        # 添加到向量库
        for doc, vec in zip(new_docs, vectors):
            vector_store.add_document(
                content=doc["content"],
                metadata=doc["metadata"],
                embedding=vec
            )

        # 持久化
        vector_store.persist(text_vector_path)