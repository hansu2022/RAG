# rag/rag_api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging

# 导入RAG Agent
from rag.LLM_agent import QwenAPIAgent, LocalRAGAgent
from rag.sci_inov.config import settings

logging.basicConfig(level=logging.INFO)

# 1. 初始化 FastAPI 应用
app = FastAPI(title="RAG Agent Service (Qwen + Milvus)")

# 2. 初始化 RAG Agent (全局单例)
try:
    if settings.USE_LOCAL_MODEL:
        logging.info(f"正在初始化本地 RAG Agent (LLM: {settings.LOCAL_LLM_MODEL_NAME})...")
        rag_agent = LocalRAGAgent(
        system_prompt="你是一位专业的科研信息检索和问答助手。请始终利用检索到的最新信息（如果存在）来回答用户的问题，并保持中文回复的专业性。"
        )
    else:
        logging.info("正在初始化云端 Qwen RAG Agent...")
        rag_agent = QwenAPIAgent(
        system_prompt="你是一位专业的科研信息检索和问答助手。请始终利用检索到的最新信息（如果存在）来回答用户的问题，并保持中文回复的专业性。"
        )
    
    logging.info("RAG Agent 初始化成功。")
except Exception as e:
    logging.error(f"RAG Agent 初始化失败: {e}")
    rag_agent = None

# 3. 定义请求体 Pydantic Model
class ChatRequest(BaseModel):
    query: str
    session_id: str | None = "default_session"
    # 允许用户覆盖系统提示词 (system_prompt)
    system_prompt: str | None = None

# 4. 定义 API 路由
@app.post("/rag/chat")
async def chat_endpoint(request: ChatRequest):
    if rag_agent is None:
        raise HTTPException(status_code=503, detail="RAG 服务未准备好，Agent 初始化失败。")

    try:
        # 调用 Agent 的 chat 方法，该方法会进行工具调用（Milvus搜索）和最终生成
        response = rag_agent.chat(
            query=request.query,
            session_id=request.session_id,
            save_history=True,
            system_prompt=request.system_prompt
        )
        return {"success": True, "response": response, "session_id": request.session_id}
    except Exception as e:
        logging.error(f"处理 RAG 请求失败: {e}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


# 5. 运行入口 (可选，但方便直接运行)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
