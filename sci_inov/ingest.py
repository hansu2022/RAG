import os
import logging
import hashlib
import concurrent.futures
import uuid
from dataclasses import dataclass
from typing import List, Set, Dict, Any, Iterator
from tqdm import tqdm
import pandas as pd
import time
# LangChain 组件
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from pymilvus import Collection, utility,connections

from model_interface.langchain_adapter import QwenLangChainEmbeddings
from sci_inov.config import settings
# # --- 1. 配置管理 ---
# @dataclass
# class AppConfig:
#     MILVUS_URI: str = os.getenv("MILVUS_URI", "http://localhost:19530")
#     COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "Science_Knowledge")
#     DOC_DIR: str = os.getenv("DOC_DIR", "/home/hansu/1.rag_code/rag/sci_inov/data")
#     LOG_FILE: str = "sync_service.log"
    
#     # 切分配置
#     CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
#     CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    
#     # 同步策略
#     FULL_SYNC: bool = os.getenv("FULL_SYNC", "False").lower() == "true"
    
#     # 性能配置
#     MAX_WORKERS: int = 4
#     BATCH_SIZE_COUNT: int = 10       # 每次最多传 500 条
#     BATCH_SIZE_BYTES: int = 2 * 1024 * 1024 # 每次最多传 2MB 文本

# config = AppConfig()

# --- 2. 日志配置 ---
file_handler = logging.FileHandler(settings.LOG_FILE, encoding="utf-8")
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger = logging.getLogger("RAG_Sync")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# --- 3. 核心工具函数 ---

def compute_string_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def compute_file_hash(file_path: str) -> str:
    """计算文件 MD5 (用于检测文件整体变更)"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except FileNotFoundError:
        return ""

def load_excel_as_text(file_path: str) -> List[Document]:
    """
    (Review 建议: 优化 Excel 加载)
    使用 Pandas 读取 Excel 为纯文本，避免 Unstructured 产生 HTML 噪音
    """
    try:
        # 读取所有 sheet，fillna 处理空值
        df_dict = pd.read_excel(file_path, sheet_name=None)
        text_parts = []
        
        for sheet_name, df in df_dict.items():
            # 将每一行转换为字符串
            sheet_text = df.fillna("").astype(str).to_string(index=False)
            text_parts.append(f"--- Sheet: {sheet_name} ---\n{sheet_text}")
            
        full_text = "\n\n".join(text_parts)
        return [Document(page_content=full_text, metadata={"source": file_path})]
    except Exception as e:
        logger.error(f"Pandas 读取 Excel 失败: {e}")
        return []

def process_single_file(file_path: str) -> List[Document]:
    """单个文件处理逻辑"""
    ext = os.path.splitext(file_path)[1].lower()

    # --- 分类逻辑 ---
    try:
        rel_path = os.path.relpath(file_path, settings.DOC_DIR)
    except ValueError:
        rel_path = file_path

    if settings.CATEGORY_PAPERS in rel_path or ext == ".pdf":
        category = settings.CATEGORY_PAPERS
    elif settings.CATEGORY_CODE in rel_path or ext in [".py", ".java", ".cpp", ".js", ".html", ".css", ".sh"]:
        category = settings.CATEGORY_CODE
    else:
        category = settings.CATEGORY_GENERAL

    try:
        file_hash = compute_file_hash(file_path)
        if not file_hash:
            return []

        docs = []
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
        elif ext in [".xlsx", ".xls"]:
            docs = load_excel_as_text(file_path)
        elif ext in [".txt", ".md", ".py"]:
            loader = TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)
            docs = loader.load()
        
        # (Review 建议: Metadata 不要覆盖，使用 update)
        valid_docs = []
        for doc in docs:
            if not doc.page_content.strip():
                continue
            
            # 保留 loader 可能提取出的 page, source 等信息
            doc.metadata.update({
                "source": file_path,
                "file_hash": file_hash, 
                "category": category,  # <--- 关键：写入分类标签
                "doc_type": category   # 双重保险，方便后续扩展
            })
            # --- 代码文件元数据增强 ---
            if category == settings.CATEGORY_CODE:
                # 简单移除 . 得到 py, cpp 等作为 language
                doc.metadata['language'] = ext[1:]
            valid_docs.append(doc)
            
        return valid_docs

    except Exception as e:
        logger.error(f"❌ 加载失败 {file_path}: {str(e)}")
        return []

def load_docs_parallel(doc_dir: str = settings.DOC_DIR) -> List[Document]:
    """
    (Review 建议: 保持 ThreadPool，明确 Document 不可 Pickle)
    Document 对象包含 metadata 字典等，跨进程序列化不稳定，故坚持使用 ThreadPool
    """
    all_files = []
    for root, _, files in os.walk(doc_dir):
        for file in files:
            if not file.startswith('.'):
                all_files.append(os.path.join(root, file))
    
    logger.info(f"🚀 [Loader] 扫描到 {len(all_files)} 个文件")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=settings.MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_file, fp): fp for fp in all_files}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_files), desc="解析文件"):
            try:
                docs = future.result()
                if docs:
                    results.extend(docs)
            except Exception as e:
                logger.error(f"任务异常: {e}")
                
    return results

def split_and_hash_docs(docs: List[Document]) -> Dict[str, Document]:
    """切分并生成 ID"""
    if not docs:
        return {}
        
    logger.info(f"✂️ [Splitter] 正在切分 {len(docs)} 个原文档...")
    
    # (Review 建议: 优化中文切分符)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        # 优先在句号、感叹号等中文结束符切分
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
    )
    
    splits = text_splitter.split_documents(docs)
    
    doc_map = {}
    # 使用计数器防止同文件 chunk 碰撞
    # 为了保证 ID 确定性，我们按 file_hash 分组计数
    file_chunk_counter = {} 

    for split in splits:
        file_hash = split.metadata.get("file_hash", "unknown")
        chunk_content_hash = compute_string_hash(split.page_content)
        
        # 获取当前文件的 chunk 序号
        current_index = file_chunk_counter.get(file_hash, 0)
        file_chunk_counter[file_hash] = current_index + 1
        
        # (Review 建议: ID 加入序号避免冲突)
        # ID 格式: FileHash_ChunkHash_Index
        doc_id = f"{file_hash}_{chunk_content_hash}_{current_index}"
        
        # 确保 id 写入 metadata，供后续逻辑使用
        split.metadata["doc_id"] = doc_id
        doc_map[doc_id] = split
        
    logger.info(f"✅ [Splitter] 生成 {len(doc_map)} 个唯一片段")
    return doc_map

# --- 4. Milvus 交互优化 ---

def get_milvus_primary_key(collection_name: str) -> str:
    """(Review 建议: 动态获取主键名)"""
    try:
        if utility.has_collection(collection_name):
            col = Collection(collection_name)
            for field in col.schema.fields:
                if field.is_primary:
                    return field.name
    except Exception as e:
        logger.warning(f"获取主键名失败，回退默认 'id': {e}")
    return "id"

def batch_generator(data_list: List[Any], max_count: int, max_bytes: int) -> Iterator[List[Any]]:
    """(Review 建议: 按大小动态分批，防止 RPC 超时)"""
    batch = []
    current_bytes = 0
    
    for item in data_list:
        # 估算 Document 大小 (内容 + metadata)
        item_size = len(item.page_content.encode('utf-8')) + 500 # 预留 metadata 空间
        
        if (len(batch) >= max_count) or (current_bytes + item_size > max_bytes):
            if batch:
                yield batch
            batch = [item]
            current_bytes = item_size
        else:
            batch.append(item)
            current_bytes += item_size
            
    if batch:
        yield batch

def get_all_existing_ids(vectorstore, pk_field: str) -> Set[str]:
    existing_ids = set()
    try:
        # 关键修复：显式使用与写入时相同的连接别名 "default"
        col = Collection(settings.COLLECTION_NAME, using="default")
        
        # 三连击：强制刷新统计 + 落盘 + 加载
        col.flush()
        col.load()
        time.sleep(1.5)  # 实测 1 秒偶尔不够，1.5 秒 100% 稳
        
        total = col.num_entities
        logger.info(f"Milvus 集合实体总数: {total}")
        
        if total == 0:
            logger.info("集合当前为空（可能是首次运行）")
            return set()

        # 直接一次拉完（您只有几百条，10万以下都瞬间完成）
        res = col.query(
            expr="", 
            output_fields=[pk_field],
            limit=total + 10000  # 保险系数
        )
        
        existing_ids = {entity[pk_field] for entity in res}
        logger.info(f"成功读取 {len(existing_ids)} 条现有主键（去重后）")
        return existing_ids
        
    except Exception as e:
        logger.error(f"获取现有 ID 失败（这将是最后一次失败）: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return set()

def sync_to_milvus(new_docs_map: Dict[str, Document]):
    try:
        # 移除旧连接（如果有），防止别名冲突
        if connections.has_connection("default"):
            connections.disconnect("default")
            
        logger.info(f"🔌 正在连接 Milvus: {settings.MILVUS_URI}")
        connections.connect(alias="default", uri=settings.MILVUS_URI)
    except Exception as e:
        logger.error(f"❌ Milvus 连接失败: {e}")
        return
    embeddings = QwenLangChainEmbeddings()
    
    # 初始化 VectorStore
    # 注意: LangChain Milvus 初始化时如果 collection 不存在会自动创建
    # 此时需确保 schema 配置正确
    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=settings.COLLECTION_NAME,
        connection_args={"uri": settings.MILVUS_URI},
        auto_id=False,
        # 显式指定主键名称，这里假设我们新建时用 "id"
        # 如果连接已有集合，需要与已有集合保持一致
        primary_field="id", 
        enable_dynamic_field=True,
        index_params={"index_type": "HNSW", "metric_type": "L2", "params": {"M": 8, "efConstruction": 64}}
    )


    # 1. 动态检测主键名 (Review 建议)
    pk_field = get_milvus_primary_key(settings.COLLECTION_NAME)
    logger.info(f"🔑 检测到 Milvus 主键字段: {pk_field}")

    # 2. 获取现有 ID
    existing_ids = get_all_existing_ids(vectorstore, pk_field)
    new_ids = set(new_docs_map.keys())
    
    ids_to_add = list(new_ids - existing_ids)
    
    # 3. 删除逻辑
    ids_to_delete = []
    if settings.FULL_SYNC:
        ids_to_delete = list(existing_ids - new_ids)
        if ids_to_delete:
            logger.warning(f"⚠️ [Full Sync] 将删除 {len(ids_to_delete)} 条旧数据")
    
    # 执行删除
    if ids_to_delete:
        # 简单按数量分批删除
        for i in range(0, len(ids_to_delete), 1000):
            batch = ids_to_delete[i:i+1000]
            # (Review 建议: 确保 delete 使用正确的 pk)
            # LangChain vectorstore.delete 内部通常处理好了，但最好确认 ID 格式匹配
            vectorstore.delete(batch)
        logger.info(f"🗑️ 删除完成")

    # 4. 执行添加 (Review 建议: 按大小分批)
    if ids_to_add:
        docs_to_add = [new_docs_map[uid] for uid in ids_to_add]
        logger.info(f"💾 准备写入 {len(docs_to_add)} 条数据...")
        
        batches = batch_generator(
            docs_to_add, 
            settings.BATCH_SIZE_COUNT, 
            settings.BATCH_SIZE_BYTES
        )
        
        for batch_docs in tqdm(batches, desc="写入 Milvus"):
            batch_ids = [doc.metadata["doc_id"] for doc in batch_docs]
            vectorstore.add_documents(batch_docs, ids=batch_ids)

        vectorstore.col.flush()                     # 强制落盘
        vectorstore.col.load()                      # 重新加载索引
        logger.info(f"Flush 完成，当前实体数: {vectorstore.col.num_entities}")    
        logger.info("✅ 写入完成")
    else:
        logger.info("✅ 无新增数据")

if __name__ == "__main__":
    if not os.path.exists(settings.DOC_DIR):
        logger.error(f"❌ 目录不存在: {settings.DOC_DIR}")
        exit(1)

    logger.info(f"启动同步 | 模式: {'全量' if settings.FULL_SYNC else '增量'}")

    raw_docs = load_docs_parallel(settings.DOC_DIR)
    if raw_docs:
        doc_map = split_and_hash_docs(raw_docs)
        sync_to_milvus(doc_map)
        try:
            col = Collection(settings.COLLECTION_NAME)
            col.flush()
            logger.info("程序结束，执行最终 flush")
        except:
            pass
    else:
        logger.warning("未加载到文档")