import os
from typing import List, ClassVar
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv
load_dotenv()
# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- 常量定义 (供外部引用，避免实例化 AppConfig 也能访问) ---
VALID_CATEGORIES = ["papers", "code", "general"]

class AppConfig(BaseSettings):
    # --- 基础开关 ---
    LOCAL_MODE: bool = Field(default=False, description="本地部署")
    
    # --- 基础路径配置 ---
    PROJECT_ROOT: str = Field(default=BASE_DIR, description="项目根目录")
    
    # 默认数据目录
    DOC_DIR: str = Field(
        default=os.path.join(BASE_DIR, "sci_inov", "data"),
        description="文档数据存放目录"
    )

    # 日志文件路径 (解决 AttributeError)
    LOG_FILE: str = Field(default="rag_sync.log", description="日志文件路径")

    # --- Milvus 数据库配置 ---
    MILVUS_URI: str = Field(default="http://localhost:19530", description="Milvus 连接地址")
    MILVUS_TOKEN: str = Field(default="root:Milvus", description="Milvus 认证 Token")
    COLLECTION_NAME: str = Field(default="Science_Knowledge", description="Milvus 集合名称")
    
    # --- 文本切分与处理配置 ---
    CHUNK_SIZE: int = Field(default=800, description="文本切分块大小")
    CHUNK_OVERLAP: int = Field(default=100, description="文本切分重叠大小")

    # --- 同步策略---
    FULL_SYNC: bool = Field(default=False, description="是否开启全量同步模式 (True: 删除旧数据重写, False: 增量更新)")
    
    # --- 性能配置 (解决 ingest.py 引用缺失) ---
    BATCH_SIZE_COUNT: int = Field(default=10, description="批量处理文档数量上限")
    BATCH_SIZE_BYTES: int = Field(default=2 * 1024 * 1024, description="批量处理字节大小上限 (2MB)")
    MAX_WORKERS: int = Field(default=4, description="并行处理线程数")
    
    # --- 业务分类标签 ---
    CATEGORY_PAPERS: str = "papers"
    CATEGORY_CODE: str = "code"
    CATEGORY_GENERAL: str = "general"
    
    # --- 模型配置 ---
    EMBEDDING_MODEL_NAME: str = "text-embedding-v3"
    # 未来可扩展：EMBEDDING_MODEL_PATH: str = "/path/to/local/model"

    # --- Pydantic 配置 ---
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore" 
    )

# 实例化单例
settings = AppConfig()

if __name__ == "__main__":
    print(f"Loaded Settings from: {settings.PROJECT_ROOT}")
    print(f"Log File: {settings.LOG_FILE}")
    print(f"Categories: {VALID_CATEGORIES}")