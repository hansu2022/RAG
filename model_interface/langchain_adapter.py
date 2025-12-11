from typing import List
from langchain_core.embeddings import Embeddings
from sci_inov.config import settings
from model_interface.embedding_api_interface import QwenEmbedAPIInterface, LocalEmbedInterface
class QwenLangChainEmbeddings(Embeddings):
    """
    将项目中的 QwenEmbedAPIInterface 包装成 LangChain 标准接口
    以便在 ingest.py 中使用
    """
    def __init__(self):
        # ✅ 根据配置选择正确的模型
        if settings.USE_LOCAL_MODEL:
            self.client = LocalEmbedInterface()
        else:
            self.client = QwenEmbedAPIInterface()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain 用于批量文档向量化的标准方法"""
        # 你的 embed 接口支持 List[str]
        # 注意：Qwen API 单次 Batch 往往有限制，OpenAI 库通常会自动处理，
        # 但如果量太大建议在上层 ingest.py 控制 batch size
        return self.client.embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """LangChain 用于查询向量化的标准方法"""
        # 这里的 squeeze=True 对应你代码里的逻辑，返回单条向量
        return self.client.embed(text, squeeze=True)