from core.config import PROJECT_ROOT
import sys
sys.path.append(PROJECT_ROOT)
from core.Multimodel_LLM import final_chat
import os

# 配置向量数据库路径
TEXT_STORE_PATH = os.path.join(PROJECT_ROOT, "data", "vector", "text")
IMAGE_STORE_PATH = os.path.join(PROJECT_ROOT, "data", "vector", "img")

api_key = "your api key"  # 替换为你的API密钥

def main():
    # 初始化聊天机器人
    chatbot = final_chat(TEXT_STORE_PATH, IMAGE_STORE_PATH, api_key=api_key)
    current_session_history = []

    while True:
        question = input("\n请输入问题（输入 'exit' 退出）: ").strip()
        if question.lower() == 'exit':
            break

        # —— 修改点：capture and print the answer ——
        answer, is_image = chatbot.Chat_GLM(question)

        if is_image:
            print("\n🤖 系统生成了一张图片，你可以用浏览器打开下面这个 URL：")
            print(answer)
        else:
            print(f"\n🤖 系统回答：{answer}")
        # ————————————————————————————————

        # 记录当前会话
        current_session_history.extend([
            {"role": "user",    "content": question},
            {"role": "assistant","content": answer}
        ])

        # 让用户选择接下来的操作
        while True:
            choice = input(
                "\n请选择操作：\n"
                "1. 继续当前会话\n"
                "2. 重新开始对话\n"
                "3. 退出对话\n"
                "输入数字选择: "
            ).strip()
            if choice == '1':
                break
            elif choice == '2':
                current_session_history = []
                chatbot.clear_history()
                print("已开启新对话，历史已清空")
                break
            elif choice == '3':
                print("正在退出程序…")
                return
            else:
                print("输入无效，请重新输入")


if __name__ == "__main__":
    main()