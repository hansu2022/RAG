import sys
sys.path.append("..")

import logging
from typing import List, Optional

# 基础配置
from rag.milvus_base import MilvusDBBase
from rag.model_interface.chat_api_interface import QwenAPIInterface, LocalChatInterface
from rag.model_interface.embedding_api_interface import QwenEmbedAPIInterface, LocalEmbedInterface
from rag.sci_inov.tool_call import tools, TOOL_PROMPT
from rag.sci_inov.config import settings

logging.basicConfig(level=logging.INFO)

class MilvusSciInovDB(MilvusDBBase):
    def __init__(self, uri=settings.MILVUS_URI, token=settings.MILVUS_TOKEN, **kwargs):
        # 初始化基类
        self.col_name = settings.COLLECTION_NAME
        super().__init__(uri, token, col_name=self.col_name, **kwargs)
        

        if settings.USE_LOCAL_MODEL:
            logging.info(f"🚀 [MilvusDB] 使用本地模型: LLM={settings.LOCAL_LLM_MODEL_NAME}, Embed={settings.LOCAL_EMBED_MODEL_NAME}")
            self.embed_model = LocalEmbedInterface()
            # 这里的 chat_model 主要用于 Milvus 内部的工具调用决策（如果需要的话）
            self.chat_model = LocalChatInterface()
        else:
            logging.info("☁️ [MilvusDB] 使用云端 Qwen 模型")
            self.embed_model = QwenEmbedAPIInterface()
            self.chat_model = QwenAPIInterface()
        
        # 检查集合是否存在（由 ingest.py 创建）
        if not self.client.has_collection(self.col_name):
            logging.info(f"⚠️ 集合 {self.col_name} 不存在，正在自动创建...")
            self.auto_create_collection()
        else:
            self.client.load_collection(self.col_name)
            logging.info(f"✅ 已加载知识库: {self.col_name}")
    def auto_create_collection(self):
        """自动创建集合"""
        try:
            embeddings = QwenLangChainEmbeddings()
            # 这里的 connection_args 需要 token，我们从 self.token 获取 (基类中应已保存)
            # 如果基类没有保存 token 到 self.token，这里直接用 settings.MILVUS_TOKEN
            
            vectorstore = Milvus.from_texts(
                texts=["Init"], 
                embedding=embeddings,
                collection_name=self.col_name,
                connection_args={"uri": self.uri, "token": self.token}, 
                auto_id=False, 
                primary_field="id",
                enable_dynamic_field=True,
                ids=["init_001"], 
                index_params={"index_type": "HNSW", "metric_type": "L2", "params": {"M": 8, "efConstruction": 64}}
            )
            vectorstore.delete(["init_001"])
            logging.info(f"✅ 集合 {self.col_name} 自动创建成功！")
        except Exception as e:
            logging.error(f"❌ 自动创建集合失败: {e}")

    # =========================================================
    #  核心修复：添加这些空方法，解决 "Can't instantiate abstract class" 报错
    # =========================================================
    def set_schema(self):
        pass

    def set_indices(self):
        pass

    def set_f_attr(self, f_dict=None):
        pass

    def insert_item(self, data):
        pass
    # =========================================================

    def embed_queries(self, q: str | List[str]):
        return self.embed_model.embed(q, squeeze=False)

    def search_knowledge_base(self, query: str, category: Optional[str] = None, top_k: int = 5):
        """
        统一搜索入口
        """
        logging.info(f"🔍 搜索: '{query}' | 分类过滤: {category}")
        
        if isinstance(query, str):
            query = [query]

        try:
            query_vectors = self.embed_queries(query)
        except Exception as e:
            logging.error(f"向量化失败: {e}")
            return []

        # 构建过滤表达式
        filter_expr = ""
        if category:
            if category in [settings.CATEGORY_PAPERS, settings.CATEGORY_CODE, settings.CATEGORY_GENERAL]:
                 filter_expr = f'category == "{category}"'
            else:
                 logging.warning(f"检测到未知分类标签: {category}，忽略过滤条件")
                 filter_expr = ""

        try:
            res = self.client.search(
                collection_name=self.col_name,
                data=query_vectors,
                anns_field="vector",  # LangChain 默认向量字段名
                filter=filter_expr,
                limit=top_k,
                output_fields=["text", "source", "category", "doc_id"] 
            )

            results = []
            for hits in res:
                formatted_hits = []
                for hit in hits:
                    entity = hit['entity']
                    formatted_hits.append({
                        "content": entity.get("text", ""),
                        "source": entity.get("source", "Unknown"),
                        "category": entity.get("category", "general"),
                        "score": hit['distance']
                    })
                results.append(formatted_hits)
            
            # --- 日志代码---
            final_res = results[0] if results else []
            logging.info(f"📚 [Milvus] 检索到 {len(final_res)} 条记录:")
            for idx, item in enumerate(final_res):
                # 只打印前100个字符避免刷屏
                preview = item['content'][:100].replace('\n', ' ') + "..."
                logging.info(f"   [{idx+1}] Score:{item['score']:.4f} | Source:{item['source']} | Content: {preview}")
            # --- 日志代码---

            return final_res

        except Exception as e:
            logging.error(f"❌ 搜索失败: {e}")
            return []

    def search(self, query: str | List[str], top_k=5):
        """
        RAG Agent 调用的主入口
        """
        if isinstance(query, str):
            query = [query]
            
        all_results = []
        for q in query:
            # 1. 让 LLM 决定调用哪个工具 + 带什么参数
            try:
                tool_calls = self.chat_model.tool_call(q, tools, TOOL_PROMPT)
            except Exception as e:
                logging.error(f"Tool call error: {e}")
                tool_calls = []
            
            category = None
            should_search = False
            
            # 解析 tool_call 结果
            # 注意：需确保 chat_api_interface.py 的 tool_call 返回的是包含 'arguments' 的字典列表
            if tool_calls:
                for call in tool_calls:
                    if isinstance(call, dict) and call.get("name") == "search_knowledge_base":
                        should_search = True
                        args = call.get("arguments", {})
                        category = args.get("category")
                        break
                    elif isinstance(call, str) and call == "search_knowledge_base":
                         # 兼容旧版本只返回字符串的情况
                         should_search = True
                         break
            
            # 保底策略：如果问题像是在问知识，即使没调工具也强行搜
            if not should_search:
                keywords = ["是什么", "解释", "原理", "代码", "论文", "介绍", "如何", "怎么"]
                if any(k in q for k in keywords):
                    should_search = True

            # 2. 执行搜索
            if should_search:
                logging.info(f"执行检索 -> Query: {q}, Category: {category}")
                results = self.search_knowledge_base(q, category=category, top_k=top_k)
                all_results.append(results)
            else:
                logging.info("LLM 决定不检索，且未触发保底策略")
                all_results.append([])
                
        return all_results

if __name__ == "__main__":
    db = MilvusSciInovDB()
    # 简单的冒烟测试
    print(db.search("Transformer 代码"))