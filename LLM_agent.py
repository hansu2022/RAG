import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.sci_inov.milvus_si import MilvusSciInovDB
import rag.milvus_base
import rag.model_interface.chat_api_interface
import json
import logging
from abc import ABC, abstractmethod


logging.basicConfig(level=logging.INFO)

# LLM基类，提供聊天功能，function call功能需要在子类中实现
class LLM_agent(ABC):
    def __init__(self, model_interface, history_path=None, use_api=False, device="cpu", system_prompt="You are a helpful assistant."):
        if use_api:
            self.model = model_interface()
        else:
            self.model = model_interface(device)
        self.device = device
        self.history_path = history_path
        self.system_prompt = system_prompt

        self.init_chat()

    def init_chat(self):
        # 聊天记录为字典，key为session_id，value为某一对话的聊天记录List[dict]
        self.chat_history = {}
        if self.history_path is not None:
            try:
                # 确保路径是相对于项目根目录的
                self.history_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.history_path)
                self.load_chat_history(self.history_path)
            except Exception as e:
                logging.info(f"未找到聊天记录: {self.history_path}, 错误: {str(e)}")

    @abstractmethod
    def chat(self, query: str, session_id: str):
        pass

    def load_chat_history(self, history_path: str):
        with open(history_path, 'r') as f:
            self.chat_history = json.load(f)

    def save_chat_history(self, history_path: str, create_dir: bool = False, overwrite: bool = False):
        """
        保存聊天记录到文件
        
        Args:
            history_path (str): 保存路径
            create_dir (bool): 如果目录不存在，是否创建目录
            overwrite (bool): 如果文件已存在且不为空，是否覆盖
        """
        
        # 检查路径是否合法
        try:
            os.path.normpath(history_path)
        except Exception as e:
            logging.error(f"无效的保存路径: {history_path}, 错误: {str(e)}")
            return False
            
        # 检查目录是否存在
        dir_path = os.path.dirname(history_path)
        if dir_path and not os.path.exists(dir_path):
            if create_dir:
                try:
                    os.makedirs(dir_path)
                    logging.info(f"创建目录: {dir_path}")
                except Exception as e:
                    logging.error(f"创建目录失败: {dir_path}, 错误: {str(e)}")
                    return False
            else:
                logging.error(f"目录不存在且未启用自动创建: {dir_path}")
                return False
                
        # 检查文件是否存在且不为空
        if os.path.exists(history_path) and os.path.getsize(history_path) > 0:
            if not overwrite:
                logging.warning(f"文件已存在且不为空，跳过保存: {history_path}")
                return False
            else:
                logging.info(f"覆盖已存在的文件: {history_path}")
                
        # 保存聊天记录
        try:
            with open(history_path, 'w') as f:
                json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
            logging.info(f"成功保存聊天记录到: {history_path}")
            return True
        except Exception as e:
            logging.error(f"保存聊天记录失败: {str(e)}")
            return False

    def delete_chat_history(self, session_id: str = None, message_index: int = None) -> bool:
        """
        删除指定的聊天记录
        
        Args:
            session_id (str): 要删除的会话ID，如果为None则删除所有会话
            message_index (int): 要删除的消息索引，如果为None则删除整个会话
            
        Returns:
            bool: 删除是否成功
        """
        try:
            if session_id is None:
                # 删除所有会话
                self.chat_history.clear()
            else:
                if message_index is None:
                    # 删除整个会话
                    self.chat_history.pop(session_id, None)
                else:
                    # 删除指定会话中的特定消息
                    if session_id in self.chat_history:
                        session = self.chat_history[session_id]
                        # 确保索引有效且是成对的消息（用户和助手）
                        if 0 <= message_index < len(session) and message_index % 2 == 0:
                            # 删除用户消息和对应的助手回复
                            del session[message_index:message_index+2]
                            if not session:  # 如果会话为空，删除整个会话
                                self.chat_history.pop(session_id)
            
            # 如果设置了保存路径，则保存更新后的聊天记录
            if self.history_path:
                self.save_chat_history(self.history_path, create_dir=True, overwrite=True)
                
            return True
        except Exception as e:
            logging.error(f"删除聊天记录失败: {str(e)}")
            return False


# 使用Qwen API的Agent
class QwenAPIAgent(LLM_agent):
    def __init__(self, history_path="storage/qwen_chat_history.json", system_prompt="You are a helpful assistant."):
        super().__init__(
            rag.model_interface.chat_api_interface.QwenAPIInterface,
            history_path=history_path,
            use_api=True,
            system_prompt=system_prompt
        )
        # 初始化数据库连接 (Agent 需要它来调用搜索工具)
        self.db = MilvusSciInovDB() # 初始化 Milvus 数据库实例
        
    def _format_context(self, search_results):
        """
        [修改] 适配新的通用知识库返回格式
        将检索结果格式化为 LLM 可读的字符串
        """
        if not search_results:
            return ""
        
        context_str = "检索到的相关参考资料：\n"
        
        for i, item in enumerate(search_results, 1):
            # 兼容新旧字段，防止报错
            content = item.get('content', item.get('summary', ''))  # 新字段 content, 旧字段 summary
            source = item.get('source', item.get('url', '未知来源'))
            category = item.get('category', 'general') # 新增的分类标签
            score = item.get('score', 0.0)
            
            # 拼接更紧凑的上下文
            context_str += (
                f"【资料 {i}】(类型: {category})\n"
                f"来源: {source}\n"
                f"内容摘要: {content}\n"
                f"--------------------------------\n"
            )
            
        return context_str + "\n"


    def chat(self, query: str, session_id: str, save_history: bool = True, system_prompt: str = None) -> str:
        # 获取当前会话的聊天记录
        session_history = self.chat_history.get(session_id, [])
        applied_system_prompt = system_prompt if system_prompt is not None else self.system_prompt
        
        # --- RAG 核心逻辑开始 ---
        
        # 1. 触发 LLM 工具调用决策并执行检索
        # 这里的 self.db.search 包含了 LLM 的第一次调用 (Tool Call)
        # 它会判断是否需要搜索，并返回搜索结果列表 (res 可能是 List[List[dict]])
        search_res_list = self.db.search(query, top_k=3)
        
        # 提取第一个查询的结果（如果有的话）
        search_results = search_res_list[0] if search_res_list and search_res_list[0] else []
        
        # 2. 格式化检索结果作为上下文
        retrieved_context = self._format_context(search_results)
        
        # 3. 构造新的 Prompt：将用户查询和检索到的上下文结合
        if retrieved_context:
            final_query = f"{retrieved_context}请严格根据以上检索到的知识，回答用户问题：{query}"
        else:
            final_query = query # 如果没有检索到，使用原始查询
        
        # 4. 调用模型获取回复（这是 LLM 的第二次调用，进行答案生成）
        response = self.model.chat(
            query=final_query, # 使用带有上下文的最终查询
            chat_history=session_history,
            system_prompt=applied_system_prompt
        )
        
        # --- RAG 核心逻辑结束 ---
        
        # 5. 更新聊天记录
        session_history.extend([
            {'role': 'user', 'content': query},
            {'role': 'assistant', 'content': response}
        ])
        self.chat_history[session_id] = session_history

        # 6. 保存聊天记录
        if save_history and self.history_path:
            self.save_chat_history(self.history_path, create_dir=True, overwrite=True)

        return response

    def force_save_history(self):
        """强制保存当前的历史记录"""
        if self.history_path:
            return self.save_chat_history(self.history_path, create_dir=True, overwrite=True)
        return False


if __name__ == "__main__":
    agent = QwenAPIAgent()
    agent.chat("你好", "100")
    agent.chat("介绍一位AI专家,不超过3句话.", "100")
    agent.chat("我们已经进行了几轮对话?", "100")
    agent.chat("我用你提到的方法煎了牛排,但是不好吃,你能安慰我两句吗?", "123")
    #print(agent.chat("我们一共进行了几轮对话？", "123"))
    #agent.delete_chat_history(session_id="123", message_index=0)
    print(agent.chat_history)