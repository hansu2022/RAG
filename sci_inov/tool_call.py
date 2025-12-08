TOOL_PROMPT = """
你是一个科研助手，可以查询论文和专家学者信息。
当用户有查询特定领域专家的意向时，请调用search_by_research工具。
当用户有查询特定论文的意向时，请调用search_by_paper_abstract工具。
如果用户没有查询特定领域专家或论文的意向，你直接回复None。
"""

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_by_paper_abstract",
            "description": "查询论文。将用户query与论文摘要信息匹配，返回相关论文的标题、作者等信息。",
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_research",
            "description": "查询专家学者信息。将用户query与专家学者研究方向匹配，返回相关专家学者的姓名、研究方向、主页等信息。",
        }
    }
]

