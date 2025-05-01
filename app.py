#!/usr/bin/env python
# -*- coding: utf-8 -*-
import streamlit as st
from PIL import Image
import os
import sys
import tempfile
import requests
from io import BytesIO
from core.config import PROJECT_ROOT
import re

sys.path.append(PROJECT_ROOT)
TEXT_STORE_PATH = os.path.join(PROJECT_ROOT, "data", "vector", "text")
IMAGE_STORE_PATH = os.path.join(PROJECT_ROOT, "data", "vector", "img")

from core.Multimodel_LLM import final_chat
from core.utils import ReadFiles
from core.LLM import ImageChat

# streamlit run app.py

# --- Streamlit配置 ---
st.set_page_config(layout="wide", page_icon="🏥")
st.markdown("""
<style>
    .fixed-input { position: fixed; bottom: 0; left: 0; right: 0; background-color: white; padding: 1rem; box-shadow: 0 -2px 10px rgba(0,0,0,0.1); z-index: 999; }
    .block-container { padding-bottom: 8rem; }
    .chat-box { padding: 1rem; border-radius: 10px; background-color: #f1f3f4; margin-bottom: 1rem; }
    .generated-image { border: 1px solid #e0e0e0; border-radius: 10px; margin: 1rem 0; }
    .sidebar-history-item {padding: 0.5rem;margin-bottom: 0.5rem;border-radius: 10px;background-color: #f7f9fa;border: 1px solid #f7f9fa;}
</style>
""", unsafe_allow_html=True)

# --- 初始化会话状态 ---
if "current_chat" not in st.session_state:
    st.session_state.current_chat = []
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}
if "selected_history" not in st.session_state:
    st.session_state.selected_history = None
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

# --- 工具函数 ---
@st.cache_resource
def init_chatbot(model_name="glm-4-flash", temperature=0.3, threshold=0.5):
    return final_chat(
        TEXT_STORE_PATH,
        IMAGE_STORE_PATH,
        model_name=model_name,
        temperature=temperature,
        threshold=threshold,
        api_key=st.session_state.get("api_key", "")
    )

# --- 页面标题 ---
st.title("🏥 医疗洁净室智能问答系统")
st.subheader("提供医疗洁净室设计、标准、操作规范等全方位知识服务")

# --- 侧边栏 ---
with st.sidebar:
    st.title("⚙️ 系统配置")
    model_choice = st.selectbox(
        "选择大模型引擎",
        ["glm-4-flash", "glm-z1-air", "glm-z1-airx", "glm-4-flash-250414", "glm-4-air-250414", "glm-4-plus", "glm-4v",
         "cogview-4-250304"],
        index=0
    )
    temperature = st.slider("模型温度系数 (0-1)", 0.0, 1.0, 0.3)
    threshold = st.slider("图片匹配阈值 (0-1)", 0.0, 1.0, 0.5)

    st.markdown("---")
    st.subheader("🔑 API Key 配置")
    api_key = st.text_input("请输入您的 智谱 BigModel API Key", type="password", value="")
    st.session_state.api_key = api_key

    st.markdown("---")
    st.subheader("📂 历史对话")
    for title in list(st.session_state.chat_histories.keys()):
        with st.container():
            hist_cols = st.columns([5, 1])
            fixed_title = title[:9]
            with hist_cols[0]:
                if st.button(fixed_title, key=f"load_{title}", use_container_width=True):
                    st.session_state.current_chat = st.session_state.chat_histories[title].copy()
                    st.session_state.selected_history = title
                    st.rerun()
            with hist_cols[1]:
                if st.button("❌", key=f"delete_{title}", use_container_width=True):
                    del st.session_state.chat_histories[title]
                    if st.session_state.selected_history == title:
                        st.session_state.selected_history = None
                        st.session_state.current_chat = []
                    st.rerun()

    st.markdown("---")
    if st.button("🆕 开启新会话"):
        if st.session_state.current_chat:
            first_question = st.session_state.current_chat[0]['question']
            st.session_state.chat_histories[first_question] = st.session_state.current_chat.copy()
        st.session_state.current_chat = []
        st.session_state.selected_history = None
        st.session_state.uploaded_chunks = []
        if "image_content" in st.session_state:
            del st.session_state.image_content
        st.rerun()

# 初始化图片对话模型
img_chat = ImageChat(api_key=st.session_state.get("api_key", ""))

# --- 文件上传区域 ---
with st.expander("📁 上传辅助资料", expanded=False):
    uploaded_file = st.file_uploader("支持PDF/文本/图片格式", type=["pdf", "txt", "md", "png", "jpg", "jpeg"])
    if uploaded_file:
        with st.spinner("🔍 解析文件中..."):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}")
            tmp.write(uploaded_file.read())
            tmp.close()
            try:
                if uploaded_file.type == "application/pdf":
                    raw_text = ReadFiles.read_pdf(tmp.name)
                elif uploaded_file.type == "text/plain":
                    raw_text = ReadFiles.read_text(tmp.name)
                elif uploaded_file.type == "text/markdown":
                    raw_text = ReadFiles.read_markdown(tmp.name)
                else:
                    raw_text = img_chat.generate_response(tmp.name)
                    st.session_state.image_content = raw_text
                if raw_text:
                    chunks = ReadFiles.get_chunk2(raw_text)
                    st.session_state.uploaded_chunks = chunks
                    st.success(f"✅ 成功提取 {len(chunks)} 个知识片段")
            except Exception as e:
                st.error(f"文件解析失败: {e}")

# --- 渲染当前聊天记录 ---
for idx, entry in enumerate(st.session_state.current_chat):
    st.markdown(f"**🙋 用户提问 {idx + 1}:** {entry['question']}")

    #  图片回答
    if entry.get("answer_type") == "image":
        img_path = entry.get("answer_image_path", entry['answer'])
        try:
            if img_path.startswith("http"):
                st.image(img_path, caption="AI 生成图片", use_container_width=True)
            elif os.path.exists(img_path):
                st.image(Image.open(img_path), caption="AI 生成图片", use_container_width=True)
            else:
                st.error("无法加载图片资源")
        except Exception as e:
            st.error(f"图片加载失败: {str(e)}")

    else:
        #  文本回答
        answer = entry['answer']
        st.write(f"<div class='chat-box'>🤖 系统回答：{answer}</div>", unsafe_allow_html=True)

    # 相关参考图片
    related = entry.get("related_images", [])
    if related:
        st.markdown("---")
        st.subheader("📷 相关参考图片")
        cols = st.columns(min(3, len(related)))
        for i, info in enumerate(related):
            with cols[i % 3]:
                try:
                    if os.path.exists(info["path"]):
                        st.image(Image.open(info["path"]),
                                 caption=f"来源：{info['source']}",
                                 use_container_width=True)
                    else:
                        st.warning(f"图片未找到：{os.path.basename(info['path'])}")
                except Exception as e:
                    st.error(f"图片加载错误: {str(e)}")

# --- 聊天输入框 ---
with st.form("chat_input", clear_on_submit=True):
    q_col, btn_col = st.columns([5, 1])
    with q_col:
        question = st.text_area("输入您的问题", height=100, placeholder="请输入关于医疗洁净室的问题...",
                                label_visibility="collapsed")
    with btn_col:
        submitted = st.form_submit_button("🚀 发送")

# --- 处理新问题 ---
if submitted:
    if not st.session_state.api_key.strip():
        st.warning("⚠️ 请先在左侧栏输入有效的 API Key！")
    elif not question.strip():
        st.warning("⚠️ 请输入您的问题后再发送。")
    else:
        if not st.session_state.chatbot or st.session_state.get("current_model") != model_choice:
            st.session_state.chatbot = init_chatbot(model_choice, temperature, threshold)
            st.session_state.current_model = model_choice

        with st.spinner("🔍 检索知识库并生成回答..."):
            extra = st.session_state.get("uploaded_chunks", [])
            img_ctx = [st.session_state.get("image_content", "")] if 'image_content' in st.session_state else []
            history_ctx = [f"Q: {item['question']}\nA: {item['answer']}" for item in st.session_state.current_chat]

            raw_answer, is_image = st.session_state.chatbot.Chat_GLM(
                "\n".join(history_ctx + [question]),
                additional_context=extra + img_ctx
            )

            related_images = []
            for p in st.session_state.chatbot.get_img_path() or []:
                print(os.path.dirname(p))
                if os.path.exists(p):
                    related_images.append({
                        "path": p,
                        "source": os.path.basename(os.path.dirname(p))
                    })

            entry = {
                "question": question,
                "related_images": related_images,
                "answer": raw_answer,
                "answer_type": "image" if is_image else "text"
            }

            if is_image:
                try:
                    if raw_answer.startswith("http"):
                        response = requests.get(raw_answer)
                        img = Image.open(BytesIO(response.content))
                        tmp_path = os.path.join(tempfile.gettempdir(),
                                                f"generated_{len(st.session_state.current_chat)}.png")
                        img.save(tmp_path)
                        entry["answer_image_path"] = tmp_path
                except Exception as e:
                    st.error(f"图片保存失败: {str(e)}")
                    entry["answer_type"] = "text"
                    entry["answer"] = "图片生成失败，请重试"

            st.session_state.current_chat.append(entry)
            st.rerun()