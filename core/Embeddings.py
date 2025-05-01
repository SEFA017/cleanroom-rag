#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# 指定绝对路径
env_path = r'.env'
load_dotenv(env_path)

class BaseEmbeddings:
    """
    Base class for embeddings
    """
    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api
    
    def get_embedding(self, text: str, model: str) -> List[float]:
        raise NotImplementedError

    def get_embeddings(self, text: List[str], model: str) -> List[List[float]]:
        raise NotImplementedError
    
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude


class ZhipuEmbedding(BaseEmbeddings):
    def __init__(self, path: str = '', is_api: bool = True, embedding_dim=1024, api_key: str = os.getenv("ZHIPUAI_API_KEY")) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from zhipuai import ZhipuAI
            api_key = api_key
            if not api_key:
                raise ValueError("未找到环境变量 ZHIPUAI_API_KEY，请检查.env文件或系统环境变量")
            self.client = ZhipuAI(api_key=api_key)  # 正确传递API Key
        self.embedding_dim = embedding_dim

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
        model="embedding-3",
        input=text,
        )
        return response.data[0].embedding




