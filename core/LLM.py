import os
from typing import List
from zhipuai import ZhipuAI
from dotenv import dotenv_values
import base64

# 加载.env文件中的环境变量
env_variables = dotenv_values('.env')
for var in env_variables:
    os.environ[var] = env_variables[var]

# 文本对话提示模板
PROMPT_TEMPLATE = {
    "RAG_PROMPT_TEMPALTE":
        r"""你是一位洁净室设计专家，擅长医疗洁净用房和制药洁净车间的规范化设计。主要功能是回答用户提出的问题。

        你的基本回答逻辑是：
        使用以下上下文内容（可能包含相关信息）来回答用户的问题。注意，如果上下文包含用户问题的答案，请在你自己生成的内容相应位置标注是在哪里检索得到的；
        同时，不要说类似“根据上下文内容””根据您提供的信息”“根据相关文本”之类的话，直接说“根据数据库内容”。
        如果上下文不包含答案，请根据你自己的知识进行回答，可以进行合理的推测，但需要说明你的回答是AI生成的，而不是在数据库中检索得到的。
        此外，你需要注意以下几点：

        第一，你需要按照以下步骤进行回答：
        1. 标记关键证据：这一点是你所有回答的重点。一定要牢记！！！一定要给出信息来源的具体内容！！！
           请首先从“上下文”中识别最相关的内容，并在回答中用【文本1】、【文本2】……进行标注。
           **特别注意，由于用户看不到上下文内容，所以你还需要在回答中给出证据对应的内容。
           例如，如果你在回答中引用了上下文中的某个表格（假设为表3-3），你需要在回答中给出表格的标题和内容。**
           这一点是你所有回答的重点。一定要牢记！！！一定要给出信息来源的具体内容！！！
        2. 链式思维推理：基于标记的证据，分步说明你的推理过程，每一步都要说明依据（对应哪个证据）。
        3. 综合生成回答：最后给出完整的设计建议或规范解析，并在每个要点后引用相应证据标号。


        第二，你首先需要判断用户问题的类型。一般来说，问题可以分为两类：一类是关于洁净室设计相关的问题，另一类是关于洁净室资料（如相关的设计标准规范、文献案例等等）相关的问题。
        如果问题是关于洁净室设计相关的，你需要给出尽可能详细的回答，包含设计要点、设计原则、设计标准等等，如果需要的话，还可以给出一些定量的计算结果。
        如果问题是关于洁净室资料相关的，你同样需要给出尽可能详细的回答，包含设计标准规范的名称、版本、适用范围、要求等等。如果用户没有特殊说明，对于这类问题的回答，不需要你有太多的创新，只需要完成信息检索功能即可。
        如果方便的话，还可以用类似图片（比如直观展示房间布局、设备安装示意等等）或表格（比如各要素的归纳等等）的形式总结你的回答。另外，如果你判断用户的问题不属于这两类，你可以根据自己的理解回答用户问题。

        第三，如果上下文中的信息是表格形式的，你需要以可视化的形式展示，方便理解。总之就是要以人类可读的方式回答问题。
        也就是说，不要直接把上下文内容原封不动地复制粘贴到回答中，同时注意去除上下文中的无用信息，比如“\n”或者“\t”等等。
        
        

        第四，除非用户明确要求用指定语言，否则总是使用中文回答。

        第五，如果你觉得用户的问题不够清晰，可以首先生成你的回答。然后在最后主动询问用户以获取更多信息。
        例如用户想了解如何设计一个洁净室，但没有提供洁净室的面积、类型、洁净度等级等信息，你可以首先给出一个比较宽泛的回答，然后主动询问用户这些详细的信息，进而方便给出更精确的回答。

        最后，如果你的回答中包含公式，请按照markdown语法进行排版，并且用 `$…$`（行内公式）或 `$$…$$`（块级公式）将公式包裹起来。确保公式可读性。
        
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        有用的回答:""",
    "NO_CONTEXT_PROMPT_TEMPALTE":
        """请根据你的知识回答以下问题。如果问题超出你的知识范围，可以进行合理的推测，但需要说明你的回答是AI生成的，而不是在数据库中检索得到的。总是使用中文回答。
        问题: {question}
        有用的回答:""",
}

# 图片摘要提示模板
IMAGE_SUMMARY_PROMPT_TEMPLATE = """你是一个摘要生成助手，用于分析和描述医疗洁净室相关的图片。请根据图片内容生成一个简洁明了的摘要，以便于查询信息。请详细分析并描述该图片。
注意，这张图片与医疗洁净室相关，可能是设计相关的规范标准，也可能是一些建筑布局等等。任何医疗洁净室相关的元素都有可能出现在图片中。你的摘要可以从以下要素考虑：
如果是建筑相关图片：
1. 主要功能区域划分
2. 空气净化系统特征
3. 建筑材料与表面处理
4. 设备布局与人员动线
5. 其他显著设计特点
如果是设计标准相关图片：
1. 设计标准或规范名称
2. 适用范围与要求
3. 关键设计参数
4. 设计图例与符号说明
5. 其他重要信息
如果是其他类型的图片：
1. 图片内容概述
2. 相关的医疗洁净室元素
3. 其他相关信息
请根据图片内容进行详细描述，确保摘要准确、全面。"""

# 图片问答提示模板
IMAGE_CHAT_PROMPT_TEMPLATE = """请根据图片内容回答以下问题。注意，这张图片与洁净室相关，可能是设计相关的规范标准，也可能是一些建筑布局等等。
任何医疗洁净室相关的元素都有可能出现在图片中。请你仔细分析图片内容，给出任何相关的回答，并且要尽量详细。以便于结合上下文进行回答。"""


class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: List[str]) -> str:
        pass

    def load_model(self):
        pass


class ImageSummaryGLM(BaseModel):
    """专门用于生成图片摘要的GLM模型"""

    def __init__(self, model: str = "glm-4v", api_key: str = os.getenv("ZHIPUAI_API_KEY")):
        """初始化模型"""
        super().__init__()
        self.client = ZhipuAI(api_key=api_key)
        self.model = model

    def generate_summary(self, image_path: str) -> str:
        """生成图片摘要"""
        try:
            # 编码图片为base64
            base64_img = self._encode_image(image_path)

            # 调用API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
                        {"type": "text", "text": IMAGE_SUMMARY_PROMPT_TEMPLATE}
                    ]
                }],
                temperature=0.1,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"图片摘要生成失败: {str(e)}")
            return None

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """内部方法：将图片编码为base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


class ImageChat(BaseModel):
    """专门用于识别图片的GLM模型"""

    def __init__(self, model: str = "glm-4v-plus-0111", api_key: str = os.getenv("ZHIPUAI_API_KEY")):
        super().__init__()
        self.client = ZhipuAI(api_key=api_key)
        self.model = model

    def _encode_image(self, image_path: str) -> str:
        """内部方法：将图片编码为base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_response(self, image_path: str, content: str = "描述图片内容") -> str:
        """生成图片回答"""
        try:
            # 编码图片为base64
            base64_img = self._encode_image(image_path)

            # 调用API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
                        {"type": "text", "text": IMAGE_CHAT_PROMPT_TEMPLATE + content}
                    ]
                }],
                temperature=0.1,
                max_tokens=1024
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"图片识别失败: {str(e)}")
            return None


class GLMChat(BaseModel):
    def __init__(self, model_name: str = "glm-z1-air", temperature: float = 0.3,
                 api_key: str = os.getenv("ZHIPUAI_API_KEY")) -> None:
        super().__init__()
        self.client = ZhipuAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature

    def chat(self, prompt: str, history: List[dict], content: List[str]) -> dict:
        # 1. 根据是否有 context 选择模板
        if content:
            template = PROMPT_TEMPLATE["RAG_PROMPT_TEMPALTE"]
            print("使用RAG上下文模版进行回答")
            filled = template.format(question=prompt, context="\n".join(content))
        else:
            print("使用无上下文模版进行回答")
            template = PROMPT_TEMPLATE["NO_CONTEXT_PROMPT_TEMPALTE"]
            filled = template.format(question=prompt)

        messages = [{"role": "system", "content": filled}]

        # 2. 将历史对话添加到消息中
        history.append(messages[0])

        try:
            if self.model_name == "cogview-4-250304":
                response = self.client.images.generations(
                    model="cogview-4-250304",
                    prompt=filled,  # 用填充后的 prompt 发给图片模型
                    quality="hd",
                )
                return {"type": "image", "content": response.data[0].url}
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=history,
                    temperature=self.temperature,
                    max_tokens=32000
                )
                return {"type": "text", "content": response.choices[0].message.content}
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            return {"type": "text", "content": "暂时无法回答该问题"}