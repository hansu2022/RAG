import os
import json
from langchain.embeddings.base import Embeddings
from volcenginesdkarkruntime import Ark
from volcenginesdkarkruntime._exceptions import ArkAPIError
from typing import List

os.environ['ARK_API_KEY'] = "a0d24422-e641-48b6-b566-8fc17bcb7c9a"


# 环境变量配置（请替换为实际值）
os.environ["KIMI_API_KEY"] = "your-kimi-api-key"
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

class DoubaoEmbeddings(Embeddings):
    """火山引擎文本嵌入模型"""
    
    def __init__(
            self,
            model="doubao-embedding-large-text-240915", 
            embed_dim=4096, 
            record=False
        ):
        self.api_key = os.environ.get("ARK_API_KEY")
        self.client = Ark(api_key=self.api_key)
        self.model = model
        self.embed_dim = embed_dim
        self.record = record
        self.query = []
        self.response = []
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]

        try:
            resp = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
        except ArkAPIError as e:
            print(e)

        if self.record:
            self.query.append(texts)
            self.response.append(resp)
        
        return [item.embedding for item in resp.data]
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def clear_cache():
        self.query.clear()
        self.response.clear()




if __name__ == "__main__":
    embed = DoubaoEmbeddings()
    result = embed.embed_documents(["天很蓝", "海很深"])
    print(len(result))
    print(len(embed.response))