from pymilvus import MilvusClient
client = MilvusClient(uri="http://localhost:19530")
if client.has_collection("Science_Knowledge"):
    client.drop_collection("Science_Knowledge")
    print("✅ 已删除旧集合，可以重新运行 ingest.py 了")