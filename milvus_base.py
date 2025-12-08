from abc import ABC, abstractmethod
import logging
from pymilvus import MilvusClient


DEFAULT_URI = "http://localhost:19530"
TOKEN = "root:Milvus"

class MilvusDBBase(ABC):
    def __init__(self, 
                 uri=DEFAULT_URI, 
                 token=TOKEN, 
                 col_name="rag", 
                 **kwargs):
        self.client = MilvusClient(
            uri=uri,
            token=token,
            timeout=None
        )
        self.uri = uri
        self.col_name = col_name
        self.display_field = []
        self.set_f_attr()

    def init_collection(self, overwrite=False):
        # Collection存在, 只有overwrite为True时, 重新创建Collection
        if self.client.has_collection(self.col_name):
            if not overwrite:
                logging.info("Milvus collection named {} at {} exists. \n Initialization skipped.".format(self.col_name, self.uri))
                return
            else:
                self.client.drop_collection(self.col_name)
        
        # Collection不存在, 直接创建
        schema = self.set_schema()
        indices = self.set_indices()
        self.client.create_collection(
            collection_name = self.col_name,
            schema = schema,
            index_params = indices
        )

    @abstractmethod
    def set_schema(self):
        pass

    @abstractmethod
    def set_indices(self):
        pass

    @abstractmethod
    def set_f_attr(self, f_dict=None):
        pass

    def remove_collection(self):
        if self.client.has_collection(self.col_name):
            self.client.drop_collection(self.col_name)

    @abstractmethod
    def insert_item(self, data):
        pass
    
    def delete_item(self, ids=None, filter=""):
        if self.client.has_collection(self.col_name):
            del_count_dict = self.client.delete(
                collection_name=self.col_name,
                ids=ids,
                filter=filter
            )
            return del_count_dict
        return None

    def display(self):
        res = self.client.query(
            collection_name = self.col_name,
            filter = "id >= 0 ",
            output_fields = self.display_field,
        )
        for d in res:
            print(d)

    def unwrap_search_result(self, res):
        count = 1
        for r in res:
            print("======Results for Query {}======".format(count))
            for item in r:
                print("Search Result: ")
                print(item)
            count += 1