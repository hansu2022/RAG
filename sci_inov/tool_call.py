TOOL_PROMPT = """
你是一个专业的科研助手。你的知识库中包含了计算机科学(CS)、数学(Math)的论文、代码以及通用科研知识。

请根据用户的意图，调用 search_knowledge_base 工具，并传入合适的 category 参数：
1. 如果用户明确查询“论文”、“文献”、“Arxiv”，请设置 category="papers"。
2. 如果用户查询“代码”、“实现”、“算法源码”，请设置 category="code"。
3. 如果用户查询通用概念、原理或不确定具体类型，请不传入 category (保持为 None)。

请直接返回工具调用结果。
"""

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "从统一的科研知识库中检索信息。支持通过分类标签进行过滤。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户的搜索关键词"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["papers", "code", "general"],
                        "description": "可选。数据分类标签。如果只想看论文设为'papers'，只想看代码设为'code'。"
                    }
                },
                "required": ["query"]
            }
        }
    }
]