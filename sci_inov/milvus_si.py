import sys
sys.path.append("..")

import logging
from typing import List
import time

from rag.milvus_base import MilvusDBBase, DEFAULT_URI, TOKEN
from rag.model_interface.model_interface import ImageBindInterface
from rag.model_interface.chat_api_interface import QwenAPIInterface
from rag.model_interface.embedding_api_interface import QwenEmbedAPIInterface
from rag.sci_inov.tool_call import tools, TOOL_PROMPT

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient, DataType, Function, FunctionType, AnnSearchRequest, WeightedRanker, RRFRanker


logging.basicConfig(level = logging.INFO)

SI_COL_NAME = "SIDB"
EXPERT_COL_NAME = "SIDB_experts"
PAPER_COL_NAME = "SIDB_papers"

# 假设向量维度，与 ImageBindInterface 输出一致
DEFAULT_EMBED_DIM = 1024 # 从 __init__ 中提取，或根据 ImageBindInterface 确认

class MilvusSciInovDB(MilvusDBBase):
    def __init__(self, 
                uri=DEFAULT_URI, 
                token=TOKEN, 
                expert_col_name=EXPERT_COL_NAME,
                paper_col_name=PAPER_COL_NAME,
                embed_dim=DEFAULT_EMBED_DIM,
                **kwargs):
        # Initialize base class with one of the collection names or a generic one
        super().__init__(uri, token, col_name=expert_col_name, **kwargs) # Expert collection as default for base
        self.expert_col_name = expert_col_name
        self.paper_col_name = paper_col_name
        self.embed_dim = embed_dim
        #self.embed_model = ImageBindInterface()
        self.embed_model = QwenEmbedAPIInterface()
        self.chat_model = QwenAPIInterface()
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50).split_text
    
    def _set_expert_schema(self):
        schema = MilvusClient.create_schema(
            auto_id=True, # Milvus will auto-create 'id' INT64 PK field
            enable_dynamic_field=True, 
            description="Schema for SIDB experts"
        )

        analyzer_params = {
            "tokenizer": "jieba"
        }
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True) # Milvus will auto-generate IDs
        schema.add_field(field_name="name", datatype=DataType.VARCHAR, max_length=255, description="Expert's name", enable_match=True, enable_analyzer=True, analyzer_params=analyzer_params)
        schema.add_field(field_name="dept", datatype=DataType.VARCHAR, max_length=255, description="Department/Faculty", enable_match=True, enable_analyzer=True, analyzer_params=analyzer_params)
        schema.add_field(field_name="lab", datatype=DataType.VARCHAR, max_length=255, description="Laboratory", enable_match=True, enable_analyzer=True, analyzer_params=analyzer_params)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=255, description="Professional title")
        schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=1024, description="Personal page URL (renamed)")
        schema.add_field(field_name="research", datatype=DataType.VARCHAR, max_length=1024, description="Research interests keywords")
        schema.add_field(field_name="research_vector", datatype=DataType.FLOAT_VECTOR, dim=self.embed_dim, description="Vector embedding of research interests")
        schema.add_field(field_name="summary", datatype=DataType.VARCHAR, max_length=65535, description="Professor's overall summary", enable_analyzer=True, analyzer_params=analyzer_params)
        schema.add_field(field_name="summary_vector", datatype=DataType.FLOAT_VECTOR, dim=self.embed_dim, description="Vector for summary")
        schema.add_field(field_name="summary_bm25", datatype=DataType.SPARSE_FLOAT_VECTOR, description="BM25 for summary")
        
        schema.add_function(Function(name="summary_bm25_func", function_type=FunctionType.BM25,
                                     input_field_names=["summary"], output_field_names="summary_bm25"))
        return schema

    def _set_paper_schema(self):
        schema = MilvusClient.create_schema(
            auto_id=True, # Milvus will auto-create 'id' INT64 PK field
            enable_dynamic_field=True, 
            description="Schema for SIDB papers"
        )

        analyzer_params = {
            "tokenizer": "jieba"
        }
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True) # Paper's own ID
        schema.add_field(field_name="expert_id", datatype=DataType.INT64, description="FK to expert's Milvus ID")
        schema.add_field(field_name="paper", datatype=DataType.VARCHAR, max_length=1024, description="Paper title", enable_analyzer=True, enable_match=True, analyzer_params=analyzer_params) # Added enable_analyzer for BM25
        schema.add_field(field_name="author", datatype=DataType.VARCHAR, max_length=65535, description="Authors (e.g., comma-separated)") 
        schema.add_field(field_name="abstract", datatype=DataType.VARCHAR, max_length=65535, description="Paper abstract", enable_analyzer=True, analyzer_params=analyzer_params)
        schema.add_field(field_name="abstract_vector", datatype=DataType.FLOAT_VECTOR, dim=self.embed_dim, description="Vector for abstract")
        schema.add_field(field_name="abstract_bm25", datatype=DataType.SPARSE_FLOAT_VECTOR, description="BM25 for abstract")
        schema.add_field(field_name="paper_bm25", datatype=DataType.SPARSE_FLOAT_VECTOR, description="BM25 for paper title") # New field for paper title BM25

        schema.add_function(Function(name="abstract_bm25_func", function_type=FunctionType.BM25,
                                     input_field_names=["abstract"], output_field_names="abstract_bm25"))
        schema.add_function(Function(name="paper_bm25_func", function_type=FunctionType.BM25, # New function for paper_bm25
                                     input_field_names=["paper"], output_field_names="paper_bm25"))
        return schema

    def _set_expert_indices(self):
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="research_vector", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="summary_vector", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="summary_bm25", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        # Potentially index 'url' or 'name' if frequently filtered on, e.g., index_params.add_index(field_name="name", index_type="STL_SORT")
        return index_params

    def _set_paper_indices(self):
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="expert_id", index_type="STL_SORT") 
        index_params.add_index(field_name="abstract_vector", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="abstract_bm25", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        index_params.add_index(field_name="paper_bm25", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25") # Index for new paper_bm25 field
        return index_params
    
    def init_collection(self, overwrite=False):
        collections_to_init = [
            {
                "name": self.expert_col_name,
                "schema_fn": self._set_expert_schema,
                "index_fn": self._set_expert_indices,
                "description": "Expert"
            },
            {
                "name": self.paper_col_name,
                "schema_fn": self._set_paper_schema,
                "index_fn": self._set_paper_indices,
                "description": "Paper"
            }
        ]

        for col_info in collections_to_init:
            col_name = col_info["name"]
            logging.info(f"Initializing {col_info['description']} collection: {col_name}...")
            
            if self.client.has_collection(col_name):
                if overwrite:
                    logging.warning(f"{col_info['description']} collection '{col_name}' exists and overwrite is True. Dropping collection.")
                    self.client.drop_collection(col_name)
                else:
                    logging.info(f"{col_info['description']} collection '{col_name}' already exists and overwrite is False. Skipping creation. Loading collection.")
                    self.client.load_collection(col_name)
                    continue # Move to next collection
            
            schema_obj = col_info["schema_fn"]()
            self.client.create_collection(collection_name=col_name, schema=schema_obj)
            logging.info(f"{col_info['description']} collection '{col_name}' created.")
            
            index_params_obj = col_info["index_fn"]()
            logging.info(f"Creating index for {col_info['description']} collection '{col_name}'...")
            self.client.create_index(collection_name=col_name, index_params=index_params_obj)
            logging.info(f"Index created for {col_info['description']} collection '{col_name}'.")
            
            logging.info(f"Loading {col_info['description']} collection '{col_name}'...")
            self.client.load_collection(col_name)
            logging.info(f"{col_info['description']} collection '{col_name}' loaded.")

        # Keep these for clarity, even if set_f_attr also defines them
        self.display_field_experts = ["id", "name", "dept", "lab", "title", "url", "research", "summary"]
        self.search_field_experts = ["name", "dept", "lab", "title", "url", "research", "summary"]
        self.display_field_papers = ["id", "expert_id", "paper", "author", "abstract"]
        self.search_field_papers = ["paper", "abstract", "author"]

    def set_schema(self):
        logging.warning("set_schema() called directly on MilvusSciInovDB, but schema is now handled by init_collection for specific collections.")
        return self._set_expert_schema() # Or raise NotImplementedError

    def set_indices(self):
        logging.warning("set_indices() called directly on MilvusSciInovDB, but indices are now handled by init_collection for specific collections.")
        return self._set_expert_indices() # Or raise NotImplementedError
    
    def set_f_attr(self, f_dict=None):
        # This is now less relevant as fields are collection-specific.
        # Define them here directly so they exist when base class __init__ calls this method.
        self.display_field_experts = ["id", "name", "dept", "lab", "title", "url", "research", "summary"]
        self.search_field_experts = ["name", "research", "summary"] 
        # These are specific to MilvusSciInovDB, not directly used by base if base refers to self.display_field
        self.display_field_papers = ["id", "expert_id", "paper", "author", "abstract"]
        self.search_field_papers = ["paper", "abstract", "author"]

        # Configure default fields for the base class (related to the primary/expert collection)
        self.id_field = "id" 
        self.display_field = self.display_field_experts 
        self.search_field = self.search_field_experts
        logging.info("set_f_attr() in MilvusSciInovDB configured fields for the 'experts' collection as default for base class.")

    def insert_item(self, data):
        # This will be substantially updated in a later step.
        # For now, make it clear it needs to handle the new structure.
        logging.info("insert_item called. Preparing to insert experts and papers.")
        
        expert_data_to_insert = []
        paper_data_to_insert = []

        # Example: data = {"experts": [expert1_dict, ...], "papers_by_expert_name": {"expert_name": [paper1_dict, ...]}}
        # Or data = [{"type": "expert", ...}, {"type": "paper", ...}] - this is more complex for linking
        # Let's assume `data` is a dictionary: {'experts': expert_list, 'papers': paper_list_with_temp_expert_ref}
        # and papers will have a temporary key to link to an expert (e.g., expert_name or an index)
        # which will be converted to expert's Milvus ID after experts are inserted.
        
        # This is a placeholder and will be fully implemented in step 6
        
        experts_input = data.get('experts', [])
        papers_input = data.get('papers', []) # papers might need expert_id populated later

        # Stage 1: Insert Experts
        if experts_input:
            prepared_experts = []
            for item in experts_input:
                entry = {
                    "name": item.get("name", ""),
                    "dept": item.get("dept", ""),
                    "lab": item.get("lab", ""),
                    "title": item.get("title", ""),
                    "url": item.get("personal_page_addr", ""), # renamed field
                    "research": item.get("research", ""),
                    "summary": item.get("summary", "") 
                }
                if entry["research"]:
                    entry["research_vector"] = self.embed_query(entry["research"])
                else:
                    entry["research_vector"] = [0.0] * self.embed_dim
                if entry["summary"]:
                    entry["summary_vector"] = self.embed_query(entry["summary"])
                else:
                    entry["summary_vector"] = [0.0] * self.embed_dim
                prepared_experts.append(entry)

            if prepared_experts:
                logging.info(f"Inserting {len(prepared_experts)} experts into '{self.expert_col_name}'...")
                expert_insert_res = self.client.insert(collection_name=self.expert_col_name, data=prepared_experts)
                expert_ids = expert_insert_res['ids']
                logging.info(f"Successfully inserted experts. ")
                # Create a mapping from expert name (or another unique key from input) to Milvus ID
                # This is crucial for linking papers. Assuming input experts have a 'name' that's unique for this batch.
                expert_name_to_id_map = {expert_data['name']: expert_id for expert_data, expert_id in zip(experts_input, expert_ids)}
            else:
                expert_name_to_id_map = {}
                expert_ids = []


        # Stage 2: Insert Papers (linking to expert_ids)
        if papers_input and expert_ids: # Ensure there are experts to link to
            prepared_papers = []
            for item in papers_input:
                # Assume paper item has a key like 'expert_name_ref' to link to the expert by name
                expert_name_ref = item.get("expert_name_ref") # This key must exist in paper data
                milvus_expert_id = expert_name_to_id_map.get(expert_name_ref)

                if milvus_expert_id is None:
                    logging.warning(f"Could not find Milvus ID for expert reference '{expert_name_ref}' for paper '{item.get('paper')}'. Skipping paper.")
                    continue

                entry = {
                    "expert_id": milvus_expert_id,
                    "paper": item.get("paper", ""), # title
                    "author": item.get("author", ""),
                    "abstract": item.get("abstract", "")
                }
                if entry["abstract"]:
                    entry["abstract_vector"] = self.embed_query(entry["abstract"])
                else:
                    entry["abstract_vector"] = [0.0] * self.embed_dim
                prepared_papers.append(entry)
            
            if prepared_papers:
                logging.info(f"Inserting {len(prepared_papers)} papers into '{self.paper_col_name}'...")
                self.client.insert(collection_name=self.paper_col_name, data=prepared_papers)
                logging.info(f"Successfully inserted papers.")
        elif papers_input and not expert_ids:
             logging.warning("Paper data provided, but no expert IDs available from expert insertion. Papers will not be inserted.")


        logging.info("Data insertion process finished.")


    '''def embed_query(self, q):
        assert isinstance(q, str)
        return self.embed_model.unwrap_output(self.embed_model.embed_input(self.embed_model.process_text(q)))[0].squeeze()
    
    def embed_queries(self, q):
        assert isinstance(q, list)
        embed_list = self.embed_model.unwrap_output(self.embed_model.embed_input(self.embed_model.process_text(q)))
        return embed_list[0]'''
    
    def embed_query(self, q: str | List[str]):
        return self.embed_model.embed(q, squeeze=True)
    
    def embed_queries(self, q: str | List[str]):
        return self.embed_model.embed(q, squeeze=False)

    def embed_documents(self, docs):
        if isinstance(docs, str):
            docs = [docs]
        return self.embed_queries(docs)
    
    def unwrap_search_result(self, res, verbose=False, squeeze=True):
        count = 1
        res_list = []
        for r in res:
            if verbose:
                print("======Search results for Query {}======".format(count))
            cur_list = []
            for item in r:
                cur_list.append(item["entity"])
                if verbose:
                    print(item["entity"])
            res_list.append(cur_list)
            count += 1
        if len(res) == 1 and squeeze:
            return res_list[0]
        return res_list

    # New Search Functions Start Here
    def _fetch_experts_by_ids(self, expert_ids: list, output_fields: list = None) -> list:
        if not expert_ids:
            return []
        if output_fields is None:
            output_fields = self.display_field_experts
        # Ensure IDs are of a consistent type, e.g., int, if they came from different sources
        safe_expert_ids = [int(eid) for eid in expert_ids if eid is not None]
        if not safe_expert_ids:
            return []
        # Milvus IN operator expects a list of values for the field.
        # The expression should be like: "id in [123, 456]"
        # Max length for an IN expression can be a concern for very large lists. Chunk if necessary.
        # Max elements for IN operator is 65536 by default.
        expr = f"id in {safe_expert_ids}"
        try:
            res = self.client.query(collection_name=self.expert_col_name, filter=expr, output_fields=output_fields)
            return res if res else []
        except Exception as e:
            logging.error(f"Error fetching experts by IDs: {e}")
            return []

    def _fetch_papers_by_expert_ids(self, expert_ids: list, output_fields: list = None) -> dict:
        """Fetches papers and groups them by expert_id."""
        papers_by_expert = {eid: [] for eid in expert_ids}
        if not expert_ids:
            return papers_by_expert
        if output_fields is None:
            output_fields = self.display_field_papers
        
        safe_expert_ids = [int(eid) for eid in expert_ids if eid is not None]
        if not safe_expert_ids: return papers_by_expert
        
        expr = f"expert_id in {safe_expert_ids}"
        try:
            all_papers = self.client.query(collection_name=self.paper_col_name, filter=expr, output_fields=output_fields, limit=1000) # Adjust limit as needed
            for paper in all_papers:
                papers_by_expert.get(paper['expert_id'], []).append(paper)
            return papers_by_expert
        except Exception as e:
            logging.error(f"Error fetching papers by expert IDs: {e}")
            return papers_by_expert # Return partially filled or empty dict

    def search_by_expert_name(self, names, top_k_papers_per_expert=5):
        if isinstance(names, str):
            names = [names]
        
        # For exact name matching:
        # Milvus doesn't support list of ORs directly in a single `expr` for query easily.
        # expr = " or ".join([f'name == "{name_val}"' for name_val in names]) # This can get long
        # A better way for multiple exact matches is `name in ["name1", "name2"]`
        # However, MilvusPy version and server capabilities for complex expressions should be checked.
        # Safest might be to query for each name if the list is small, or use `in` if supported well.
        # Using `name in [...]`
        quoted_names = [f'\"{n}\"' for n in names]
        expr = f"name in [{','.join(quoted_names)}]"
        
        logging.info(f"Searching for experts with expression: {expr}")
        try:
            matched_experts = self.client.query(
                collection_name=self.expert_col_name,
                filter=expr,
                output_fields=self.display_field_experts
            )
        except Exception as e:
            logging.error(f"Error querying experts by name: {e}")
            return []

        if not matched_experts:
            logging.info("No experts found matching the given names.")
            return []
        
        results = []
        expert_ids_found = [expert['id'] for expert in matched_experts]
        papers_for_these_experts = self._fetch_papers_by_expert_ids(expert_ids_found, output_fields=self.display_field_papers)

        for expert in matched_experts:
            expert_id = expert['id']
            # Fetch papers for this expert (already fetched all, just look up)
            related_papers = papers_for_these_experts.get(expert_id, [])
            # Optionally, also search author field if strict linking by expert_id is not enough
            # This would require another query or a more complex initial paper query.
            # For now, papers are linked by expert_id.
            results.append({"expert": expert, "papers": related_papers[:top_k_papers_per_expert]})
        
        return results

    def search_by_lab(self, lab: str, top_k=10):
        expr = f'lab == "{lab}"'
        try:
            experts = self.client.query(
                collection_name=self.expert_col_name,
                filter=expr,
                output_fields=self.display_field_experts,
                limit=top_k
            )
            return experts if experts else []
        except Exception as e:
            logging.error(f"Error searching experts by lab '{lab}': {e}")
            return []

    def search_by_dept(self, dept: str, top_k=10):
        expr = f'dept == "{dept}"'
        try:
            experts = self.client.query(
                collection_name=self.expert_col_name,
                filter=expr,
                output_fields=self.display_field_experts,
                limit=top_k
            )
            return experts if experts else []
        except Exception as e:
            logging.error(f"Error searching experts by department '{dept}': {e}")
            return []

    def search_by_research(self, query_text: str, top_k=5, coeff=100):
        if isinstance(query_text, str):
            query_text = [query_text]
        query_vector = self.embed_queries(query_text)
        logging.info(f"Searching by research for query: '{query_text}'")


        # Search 1: Vector search on research_vector
        search_req_research_vec = AnnSearchRequest(
            data=query_vector,
            anns_field="research_vector",
            param={"metric_type": "COSINE"}, # Ensure metric matches index
            limit=top_k * 2 # Fetch more to give reranker options
        )

        # Search 2: BM25 search on summary_bm25
        # For BM25 search using hybrid_search, the data is the query string itself.
        search_req_summary_bm25 = AnnSearchRequest(
            data=query_text, # Query text for BM25
            anns_field="summary_bm25",
            param={"metric_type": "BM25"}, # Metric for BM25, though often implicit for SPARSE_INVERTED_INDEX
            limit=top_k * 2
        )

        reranker = RRFRanker(coeff)

        try:
            hybrid_res = self.client.hybrid_search(
                collection_name=self.expert_col_name,
                reqs=[search_req_research_vec, search_req_summary_bm25],
                ranker=reranker,
                limit=top_k,
                output_fields=self.display_field_experts
            )
            # hybrid_search returns a list of lists of hits. Assuming one query.
            return self.unwrap_search_result(hybrid_res, True)
        except Exception as e:
            logging.error(f"Error during hybrid search by research: {e}")
            return []

    def search_by_paper_name(self, name_query: str, top_k=10):
        if isinstance(name_query, str):
            name_query = [name_query]
        # Using BM25 search on the 'paper_bm25' field (title)
        logging.info(f"Searching papers by name (BM25) for: '{name_query}'")
        try:
            search_res = self.client.search(
                collection_name=self.paper_col_name,
                data=name_query, # Query text for BM25
                anns_field="paper_bm25",
                limit=top_k,
                output_fields=self.display_field_papers
            )
            # search returns a list of lists of hits. Assuming one query.
            return self.unwrap_search_result(search_res, True)
        except Exception as e:
            logging.error(f"Error searching papers by name '{name_query}': {e}")
            return []

    def search_by_paper_abstract(self, query_text: str | List[str], top_k=10):
        # Using BM25 search on the 'abstract_bm25' field
        logging.info(f"Searching papers by abstract (BM25) for: '{query_text}'")
        if isinstance(query_text, str):
            query_text = [query_text]
        try:
            search_res = self.client.search(
                collection_name=self.paper_col_name,
                data=query_text, # Query text for BM25
                anns_field="abstract_bm25",
                limit=top_k,
                output_fields=self.search_field_papers
            )
            return self.unwrap_search_result(search_res, True)
        except Exception as e:
            logging.error(f"Error searching papers by abstract: {e}")
            return []

    def search(self, query: str | List[str], top_k=10):
        if isinstance(query, str):
            query = [query]
        res = []
        for q in query:
            tool_names = self.chat_model.tool_call(q, tools, TOOL_PROMPT)
            if len(tool_names) == 0:
                res.append([])
            else:
                for tool in tool_names:
                    if tool == "search_by_research":
                        res.append(self.search_by_research(q, top_k))
                    elif tool == "search_by_paper_abstract":
                        res.append(self.search_by_paper_abstract(q, top_k))
                    else:
                        logging.warning(f"Tool {tool} not implemented")
                        res.append([])
        return res
    
if __name__ == "__main__":
    database = MilvusSciInovDB()

    qs = ["告诉我和量子力学有关的论文"]
    r = database.search(qs)
    print(r)
