# core/VectorBase_v2.py
import os
import time
import json
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm
from core.config import PROJECT_ROOT
from core.Embeddings import BaseEmbeddings

class VectorStore:
    def __init__(self, document: List[str] = []) -> None:
        self.document = document
        self.metadata: List[Dict[str, Any]] = []  # 元数据存储，路径为相对路径
        self.vectors: List[List[float]] = []

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        self.vectors = []
        for doc in tqdm(self.document, desc="Calculating embeddings"):
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors

    def batch_split_list(self, lst: List[Any], batch_size: int) -> List[List[Any]]:
        return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]

    def get_vector_batch(self, EmbeddingModel: BaseEmbeddings, batch: int) -> List[List[float]]:
        self.vectors = []
        chunks = self.batch_split_list(self.document, batch)
        for chunk in tqdm(chunks, desc="Calculating batch embeddings"):
            self.vectors.extend(EmbeddingModel.get_embeddings(chunk))
        return self.vectors

    def persist(self, path: str):
        os.makedirs(path, exist_ok=True)
        existing = {"document": [], "vectors": [], "metadata": []}
        try:
            for name in existing:
                with open(os.path.join(path, f"{name}.json"), 'r', encoding='utf-8') as f:
                    existing[name] = json.load(f)
        except FileNotFoundError:
            pass

        # collect existing keys
        existing_keys = set()
        for doc, meta in zip(existing['document'], existing['metadata']):
            if 'chunk_id' in meta:
                key = f"{meta['source_file']}#chunk{meta['chunk_id']}"
            else:
                key = meta.get('image_path') or meta.get('source_file') or doc
            existing_keys.add(key)

        new_docs, new_vecs, new_metas = [], [], []
        for doc, vec, meta in zip(self.document, self.vectors, self.metadata):
            if 'chunk_id' in meta:
                key = f"{meta['source_file']}#chunk{meta['chunk_id']}"
            else:
                key = meta.get('image_path') or meta.get('source_file') or doc
            if key not in existing_keys:
                existing_keys.add(key)
                new_docs.append(doc)
                new_vecs.append(vec)
                new_metas.append(meta)

        # write back
        with open(os.path.join(path, 'document.json'), 'w', encoding='utf-8') as f:
            json.dump(existing['document'] + new_docs, f, ensure_ascii=False)
        if new_vecs:
            with open(os.path.join(path, 'vectors.json'), 'w', encoding='utf-8') as f:
                json.dump(existing['vectors'] + new_vecs, f)
        with open(os.path.join(path, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(existing['metadata'] + new_metas, f, ensure_ascii=False)

    def load_vector_text(self, path: str):
        with open(os.path.join(path, 'vectors.json'), 'r', encoding='utf-8') as f:
            self.vectors = json.load(f)
        with open(os.path.join(path, 'document.json'), 'r', encoding='utf-8') as f:
            self.document = json.load(f)
        with open(os.path.join(path, 'metadata.json'), 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

    def load_vector_img(self, path: str):
        # metadata contains relative image_path
        with open(os.path.join(path, 'vectors.json'), 'r', encoding='utf-8') as f:
            self.vectors = json.load(f)
        with open(os.path.join(path, 'document.json'), 'r', encoding='utf-8') as f:
            self.document = json.load(f)
        with open(os.path.join(path, 'metadata.json'), 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

    def get_similarity(self, v1: List[float], v2: List[float]) -> float:
        return BaseEmbeddings.cosine_similarity(v1, v2)

    def query_text(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[Dict[str, Any]]:
        qv = EmbeddingModel.get_embedding(query)
        start_time = time.time()
        sims = []
        for doc, vec, meta in zip(self.document, self.vectors, self.metadata):
            sims.append({'content': doc, 'metadata': meta, 'score_new': self.get_similarity(qv, vec)})
        sims.sort(key=lambda x: x['score_new'], reverse=True)
        print(f'文本检索耗时 {time.time() - start_time:.2f} 秒')
        return sims[:k]

    def query_img(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 5) -> List[Dict[str, Any]]:
        qv = EmbeddingModel.get_embedding(query)
        start_time = time.time()
        sims = []
        for doc, vec, meta in zip(self.document, self.vectors, self.metadata):
            sims.append({'content': doc, 'metadata': meta, 'score_new': self.get_similarity(qv, vec)})
        sims.sort(key=lambda x: x['score_new'], reverse=True)
        print(f"图片检索耗时 {time.time() - start_time:.2f} 秒")
        return sims[:k]

    def add_document(self, content: str, metadata: Dict[str, Any], embedding: List[float]):
        self.document.append(content)
        self.metadata.append(metadata)
        self.vectors.append(embedding)

    def get_img_absolute(self, rel_path: str) -> str:
        """根据相对路径返回项目根目录下的绝对路径"""
        return os.path.join(PROJECT_ROOT, rel_path)


