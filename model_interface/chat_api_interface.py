import torch
import os
import logging
import getpass
from openai import OpenAI
from abc import ABC
import json
from typing import List
from sci_inov.config import settings
class ChatAPIInterface(ABC):
    """通用的对话API接口抽象类"""
    
    def __init__(self, model_name: str, base_url: str = None, api_key_env_name: str = None):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key_env_name = api_key_env_name
        self.api_key = self.get_api_key()
        self.client = self._init_client()
    
    def _init_client(self):
        """初始化API客户端"""
        return OpenAI(
            api_key=self.api_key, 
            base_url=self.base_url,
        )
    
    def get_api_key(self) -> str:
        """获取API密钥"""
        logging.info(f"正在初始化API key")
        api_key = os.getenv(self.api_key_env_name)
        if api_key:
            return api_key
        print(f"未找到环境变量 {self.api_key_env_name}")
        api_key = getpass.getpass(f"请输入您的API key: ")
        
        if not api_key:
            raise ValueError("API key不能为空")
            
        return api_key
    
    def chat(self, query: str, chat_history=None, system_prompt="You are a helpful assistant."):
        """发起对话"""
        messages = []
        if chat_history is not None:
            messages.extend(chat_history)
            
        messages.extend([
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': query}
        ])
        
        completion = self.client.chat.completions.create(
            model = self.model_name,
            messages = messages
        )
        return self.unwrap_output(completion)

    def tool_call(self, query: str, system_prompt: str = "You are a helpful assistant."):
        pass
    
    def unwrap_output(self, output):
        """解析API输出"""
        return output.choices[0].message.content

class QwenAPIInterface(ChatAPIInterface):  
    def __init__(self, model_name="qwen-max", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"):
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key_env_name="DASHSCOPE_API_KEY"
        )

    def tool_call(self, query: str, tools: list, system_prompt: str = "You are a helpful assistant.") -> List[dict]:
        """
        调用通义千问的 function calling，返回完整的工具名 + 参数字典列表
        返回示例: 
        [
            {
                "name": "search_knowledge_base",
                "arguments": {"query": "Transformer", "category": "code"}
            }
        ]
        """
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": query}
            ],
            tools=tools,
            tool_choice="auto"  # 关键：让模型自由决定是否调用工具
        )

        message = completion.choices[0].message
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            logging.info("LLM 决定不调用任何工具")
            return []  # 空列表表示不调用

        results = []
        for tool_call in message.tool_calls:
            if tool_call.type != "function":
                continue
                
            func = tool_call.function
            try:
                # 安全解析 JSON 参数
                args = json.loads(func.arguments)
            except json.JSONDecodeError as e:
                logging.error(f"工具参数 JSON 解析失败: {func.arguments} | 错误: {e}")
                args = {}  # 降级为空参数
            
            results.append({
                "name": func.name,
                "arguments": args
            })

            logging.info(f"Tool 调用成功 → {func.name} | 参数: {args}")

        return results
class LocalChatInterface(ChatAPIInterface):
    """
    本地 vLLM 对话接口
    """
    def __init__(self):
        super().__init__(
            model_name=settings.LOCAL_LLM_MODEL_NAME,
            base_url=settings.LOCAL_LLM_BASE_URL,
            api_key_env_name="LOCAL_LLM_API_KEY" # 这里实际上会被 get_api_key 覆盖
        )

    def get_api_key(self) -> str:
        """重写获取 Key 的逻辑，直接读取配置"""
        return settings.LOCAL_LLM_API_KEY

    def tool_call(self, query: str, tools: list, system_prompt: str = "You are a helpful assistant.") -> List[dict]:
        """
        本地模型的 Function Calling 实现。
        vLLM 兼容 OpenAI 格式，因此逻辑与 QwenAPIInterface 基本一致。
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": query}
                ],
                tools=tools,
                tool_choice="auto" 
            )
            
            message = completion.choices[0].message
            if not hasattr(message, "tool_calls") or not message.tool_calls:
                logging.info("Local LLM 决定不调用任何工具")
                return []

            results = []
            for tool_call in message.tool_calls:
                if tool_call.type != "function":
                    continue
                    
                func = tool_call.function
                try:
                    args = json.loads(func.arguments)
                except json.JSONDecodeError as e:
                    logging.error(f"工具参数解析失败: {func.arguments}")
                    args = {}
                
                results.append({
                    "name": func.name,
                    "arguments": args
                })
            return results

        except Exception as e:
            logging.error(f"Local LLM 工具调用出错: {e}")
            return []
if __name__ == "__main__":
    model = QwenAPIInterface()
    model.chat("你好")