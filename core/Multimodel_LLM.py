#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import requests
from core.config import PROJECT_ROOT
sys.path.append(PROJECT_ROOT)

from core.Embeddings import BaseEmbeddings, ZhipuEmbedding
from core.VectorBase import VectorStore
from core.LLM import BaseModel, GLMChat


class HybridRetriever:
    def __init__(self,
                 text_store_path: str,
                 image_store_path: str,
                 threshold: float = 0.8,
                 rerank_api_key: str = None):  # 新增rerank API密钥参数
        # 文本库加载
        self.text_store = VectorStore()
        self.text_store.load_vector_text(text_store_path)
        # 图片库加载
        self.image_store = VectorStore()
        self.image_store.load_vector_img(image_store_path)
        self.embedding_model = ZhipuEmbedding(api_key=rerank_api_key)
        self.threshold = threshold
        self.rerank_api_key = rerank_api_key  # 存储rerank密钥

    def _rerank_documents(self,
                          query: str,
                          documents: list,
                          top_n: int = 5) -> list:
        """执行rerank的私有方法"""
        if not self.rerank_api_key or not documents:
            return documents[:top_n]

        url = "https://open.bigmodel.cn/api/paas/v4/rerank"
        headers = {
            "Authorization": self.rerank_api_key,
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json"
        }
        payload = {
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": True,
            "return_raw_scores": True
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            result = response.json()
            return result.get("results", [])
        except Exception as e:
            print(f"[Rerank Warning] 服务不可用，使用原始排序。错误信息: {str(e)}")
            return [{"index": i} for i in range(len(documents))][:top_n]

    def hybrid_query(self,
                     query: str,
                     text_k: int = 10,
                     top_n: int = 3,
                     img_k: int = 1):
        # 1）文本检索（扩大召回数量）
        text_hits = self.text_store.query_text(
            query, self.embedding_model, k=text_k
        )

        # 2）Rerank精排
        documents = [hit['content'] for hit in text_hits]
        rerank_results = self._rerank_documents(query, documents, top_n)

        # 3）获取最终文本结果
        final_text_hits = []
        for result in rerank_results:
            idx = result['index']
            if idx < len(text_hits):
                final_text_hits.append(text_hits[idx])

        # print(f"[Rerank] 重新排序后的文本结果: {final_text_hits}")
        # print(len(final_text_hits))

        # 4）图片检索（原有逻辑不变）
        img_hits = self.image_store.query_img(
            query, self.embedding_model, k=img_k
        )
        print(f"图片检索结果: {img_hits}")
        print(f"图片检索数量: {len(img_hits)}")
        for hit in img_hits:
            print(hit['score_new'])

        filtered_imgs = [
            hit for hit in img_hits
            if hit['score_new'] >= self.threshold
        ]

        return final_text_hits, filtered_imgs


class EnhancedGLMChat(GLMChat):
    def __init__(self,
                 model_name="glm-z1-air",
                 temperature: float = 0.3,
                 api_key: str = os.getenv("ZHIPUAI_API_KEY")) -> None:
        super().__init__(model_name=model_name, temperature=temperature, api_key=api_key)

    def generate_response(self,
                          question: str,
                          text_context: list,
                          image_context: list,
                          history: list,
                          top_n: int = 3,) -> str:
        # 优化后的上下文组织方式
        context_str = "相关文本：\n" + "\n".join(
            f"[文本{i + 1}] {text}"
            for i, text in enumerate(text_context[:top_n])
        )
        if image_context:
            context_str += "\n\n相关图片线索：\n" + "\n".join(image_context)

        messages = history.copy()
        messages.append({
            "role": "user",
            "content": f"{context_str}\n\n基于以上信息，请回答：{question}"
        })
        return super().chat(prompt=question,history=messages,content=text_context)["content"]

    def format_output_v1(self, answer: str, image_paths: list) -> str:
        # 改进的格式化输出
        if not image_paths or all(not p for p in image_paths):
            return answer

        img_info = []
        for path in image_paths:
            if path:
                dir_name = os.path.basename(os.path.dirname(path))
                img_info.append(f"• 图片路径: {path}\n  来源分类: {dir_name}")

        return (
                f"{answer}\n\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "关联图片信息：\n"
                + "\n".join(img_info)
        )


class final_chat:
    def __init__(self,
                 text_store_path: str,
                 image_store_path: str,
                 model_name="glm-4-flash",
                 temperature: float = 0.3,
                 threshold: float = 0.6,
                 api_key: str = os.getenv("ZHIPUAI_API_KEY")):  # 新增API密钥参数
        self.retriever = HybridRetriever(
            text_store_path, image_store_path, threshold, rerank_api_key=api_key
        )
        self.glm_chat = EnhancedGLMChat(
            model_name=model_name, temperature=temperature, api_key=api_key
        )
        self.history = []
        self.img_path = []

    def get_history(self) -> list:
        return self.history.copy()

    def clear_history(self):
        self.history = []

    def get_img_path(self) -> list:
        return self.img_path.copy()

    def Chat_GLM(self,
                 question: str,
                 additional_context: list = None):
        # 1. 混合检索（包含rerank）
        text_hits, image_hits = self.retriever.hybrid_query(question, top_n=3)

        # 2. 文本上下文处理
        text_context = [hit['content'] for hit in text_hits]
        if additional_context:
            text_context = additional_context + text_context

        # 3. 图片信息处理
        image_contexts = [hit['content'] for hit in image_hits]
        image_paths = [
            hit['metadata'].get('image_path', '')
            for hit in image_hits
        ]
        abs_paths = [os.path.join(PROJECT_ROOT, rp) for rp in image_paths]

        self.img_path = abs_paths

        # 4. 生成响应
        raw_response = self.glm_chat.generate_response(
            question=question,
            text_context=text_context,
            image_context=image_contexts,
            history=self.history
        )

        # 处理不同类型的响应
        if isinstance(raw_response, dict) and raw_response["type"] == "image":
            answer = raw_response["content"]
            is_image = True
        else:
            answer = raw_response
            is_image = False

        # 更新历史记录（需要记录类型）
        self.history.append({"role": "user", "content": question})
        self.history.append({
            "role": "assistant",
            "content": answer,
            "type": "image" if is_image else "text"
        })

        return answer, is_image
