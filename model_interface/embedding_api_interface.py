import torch
import os
import logging
import getpass
from openai import OpenAI
from abc import ABC
from typing import List
# 记得导入 settings
from rag.sci_inov.config import settings

class EmbedAPIInterface(ABC):
    """通用的向量化API接口抽象类"""
    
    def __init__(self, model_name: str, base_url: str = None, embed_dim: int = None, api_key_env_name: str = None):
        self.model_name = model_name
        self.base_url = base_url
        self.embed_dim = embed_dim
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
        # 如果是本地模式且没有设置环境变量，可以直接跳过检查或返回默认值
        if self.api_key_env_name in ["LOCAL_LLM_API_KEY", "LOCAL_EMBED_API_KEY"]:
             api_key = os.getenv(self.api_key_env_name)
             return api_key if api_key else "EMPTY"

        logging.info(f"正在初始化API key")
        api_key = os.getenv(self.api_key_env_name)
        if api_key:
            return api_key
        print(f"未找到环境变量 {self.api_key_env_name}")
        api_key = getpass.getpass(f"请输入您的API key: ")
        
        if not api_key:
            raise ValueError("API key不能为空")
            
        return api_key
    
    def embed(self, query: str | List[str], squeeze=False):
        """发起向量化请求"""
        # 1. 这里必须先定义 kwargs 字典！
        kwargs = {
            "model": self.model_name,
            "input": query,  # Both str and list
            "encoding_format": "float"
        }
        
        # 2. 只有当 embed_dim 不为 None 时，才把 dimensions 加进去
        # (vLLM 的 BGE 模型不支持 dimensions 参数，所以 LocalEmbedInterface 会设为 None)
        if self.embed_dim is not None:
            kwargs["dimensions"] = self.embed_dim

        # 3. 使用 **kwargs 解包参数传给 create 方法
        completion = self.client.embeddings.create(**kwargs)
        return self.unwrap_output(completion, squeeze)
    
    def unwrap_output(self, output):
        """解析API输出"""
        return output.data

class QwenEmbedAPIInterface(EmbedAPIInterface):  
    def __init__(self, model_name="text-embedding-v3", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", embed_dim=1024):
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            embed_dim=embed_dim,
            api_key_env_name="DASHSCOPE_API_KEY"
        )
    
    def unwrap_output(self, output, squeeze=False):
        if not output or not output.data:
            return None
        if squeeze and len(output.data) == 1:
            return output.data[0].embedding
        embed_list = [e.embedding for e in output.data]
        return embed_list

class LocalEmbedInterface(EmbedAPIInterface):
    """
    本地 vLLM Embedding 接口 (兼容 OpenAI 格式)
    """
    def __init__(self):
        super().__init__(
            model_name=settings.LOCAL_EMBED_MODEL_NAME,
            base_url=settings.LOCAL_EMBED_BASE_URL,
            # 关键修改：设为 None，避免向 vLLM 传递 dimensions 参数
            embed_dim=None, 
            api_key_env_name="LOCAL_EMBED_API_KEY"
        )
    
    def get_api_key(self) -> str:
        return settings.LOCAL_EMBED_API_KEY
        
    def unwrap_output(self, output, squeeze=False):
        """解析 vLLM/OpenAI 格式的 embedding 输出"""
        if not output or not output.data:
            return None
        if squeeze and len(output.data) == 1:
            return output.data[0].embedding
        embed_list = [e.embedding for e in output.data]
        return embed_list

if __name__ == "__main__":
    # 测试代码
    try:
        model = QwenEmbedAPIInterface()
        # 如果没有 API KEY 这里可能会报错，可以注释掉
        # rep = model.embed("好吃")
        # print(len(rep))
    except Exception as e:
        print(f"测试跳过: {e}")