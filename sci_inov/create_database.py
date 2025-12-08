import sys
sys.path.append("..")

from rag.sci_inov.milvus_si import MilvusSciInovDB
import time
import logging


logging.basicConfig(level = logging.INFO)

# testing_doc = [
#     {
#         "text": 
#             "Tri Dao: Assistant Professor of Computer Science at Princeton University.\\n Chief Scientist at Together AI. \\n Previously: PhD, Department of Computer Science, Stanford University \\n research Interests \\n Machine learning and systems, with a focus on efficient training and long-range context:\\n Efficient Transformer training and inference. \\n Sequence models with long-range memory. \\n Structured sparsity for compact deep learning models.",
#         "source": "https://tridao.me/"
#     },
#     {
#         "text":
#             "梁文锋是广东省湛江市吴川市覃巴镇米历岭村，父母为吴川梅菉小学教师。其从小成绩优异，1996年，从梅菉小学直升至吴川市第一中学，在数学表现出极大天赋，初中时期甚至开始学习大学的数学课程。2002年，以全校第一成绩被浙江大学电子信息工程专业录取，2007年，攻读浙江大学信息与通信工程专业硕士，主攻机器视觉研究。 2007年—2008年环球金融危机期间，梁文锋与同学组建了一个团队，探索如何通过机器学习进行量化交易。 2009年，曾以实习生身份入职上海艾麒信息，后经推荐直接担任新技术部经理。",
#         "source": "https://zh.wikipedia.org/wiki/%E6%A2%81%E6%96%87%E9%94%8B"
#     }
# ]

new_testing_data = [
    {
        "name": "张三",
        "dept": "计算机科学与技术系",
        "lab": "人工智能研究所",
        "title": "教授",
        "research": "机器学习, 自然语言处理, 计算机视觉",
        "personal_page_addr": "http://example.com/zhangsan",
        "summary": "张三教授是人工智能领域的资深专家，专注于机器学习、自然语言处理和计算机视觉的研究与应用，致力于推动AI技术的发展"
    },
    {
        "name": "李四",
        "dept": "物理学院",
        "lab": "量子计算实验室",
        "title": "副教授",
        "research": "量子信息, 量子纠缠, 量子算法",
        "personal_page_addr": "http://example.com/lisi",
        "summary": "李四副教授在量子计算领域有深入研究，主攻量子信息理论、量子纠缠现象及新型量子算法的开发。他积极探索量子技术在材料科学和药物设计中的应用潜力。",
    },
    {
        "name": "王五",
        "dept": "生命科学学院",
        "lab": "基因编辑与合成生物学中心",
        "title": "研究员",
        "research": "CRISPR-Cas9, 基因编辑技术, 遗传病治疗, 农业生物技术",
        "personal_page_addr": "http://example.com/wangwu",
        "summary": "王五研究员是基因编辑领域的专家，重点研究CRISPR-Cas9技术的优化及其在遗传疾病治疗和农业生物技术改良中的应用。他的工作对于推动精准医疗和可持续农业发展具有重要意义。",
    },
    {
        "name": "赵六",
        "dept": "材料科学与工程系",
        "lab": "先进能源材料实验室",
        "title": "教授",
        "research": "储能材料, 锂离子电池, 超级电容器, 太阳能电池",
        "personal_page_addr": "http://example.com/zhaoliu",
        "summary": "赵六教授是先进能源材料领域的带头人，专注于新型储能技术，如高性能锂离子电池和超级电容器的研发，以及高效太阳能电池材料的性能优化。",
    },
    {
        "name": "孙七",
        "dept": "经济管理学院",
        "lab": "金融科技与数据分析中心",
        "title": "讲师",
        "research": "金融大数据分析, 算法交易, 区块链金融应用",
        "personal_page_addr": "http://example.com/sunqi",
        "summary": "孙七讲师的研究聚焦于金融科技，特别是金融市场的大数据分析、算法交易策略以及区块链技术在金融创新中的应用。他致力于将前沿技术与金融实践相结合。",
    }
]

def generate_mock_papers(expert_names):
    mock_papers = []
    paper_titles_templates = [
        "探索{area}的新进展", "{area}中的深度学习应用", "基于{method}的{area}分析", 
        "{area}的未来展望与挑战", "一种新颖的{area}框架", "{area}的计算模型研究",
        "机器学习在{area}的创新应用", "大数据驱动的{area}研究", "{area}系统的优化与设计",
        "{area}的跨学科研究范式"
    ]
    authors_templates = [lambda name: f"{name}, 同事A, 同事B", lambda name: f"同事X, {name}, 同事Y"]
    abstract_templates = [
        lambda title, name: f"本文 ({title}) 由{name}团队撰写，深入探讨了相关技术的核心原理和潜在应用，并提出了一种创新的解决方案。我们通过实验验证了该方案的有效性。",
        lambda title, name: f"{name}及其合作者在本研究 ({title}) 中，回顾了当前领域的主要挑战，并介绍了一种新的方法论。该方法在多个数据集上表现优异，为未来研究开辟了新方向。"
    ]
    research_areas_map = {
        "张三": ["人工智能", "自然语言处理"],
        "李四": ["量子物理", "量子算法"],
        "王五": ["基因编辑", "生物信息学"],
        "赵六": ["能源存储", "新材料"],
        "孙七": ["金融模型", "算法交易"]
    }

    paper_id_counter = 0
    for i in range(10): # Generate 10 papers
        expert_name = expert_names[i % len(expert_names)] # Cycle through experts
        expert_areas = research_areas_map.get(expert_name, ["通用领域"])
        area = expert_areas[i % len(expert_areas)]
        
        title_template = paper_titles_templates[paper_id_counter % len(paper_titles_templates)]
        paper_title = title_template.format(area=area, method="创新方法") # Added method for template
        
        author_template = authors_templates[paper_id_counter % len(authors_templates)]
        paper_authors = author_template(expert_name)
        
        abstract_template = abstract_templates[paper_id_counter % len(abstract_templates)]
        paper_abstract = abstract_template(paper_title, expert_name)
        
        mock_papers.append({
            "expert_name_ref": expert_name, # Temporary key for linking
            "paper": paper_title,
            "author": paper_authors,
            "abstract": paper_abstract
        })
        paper_id_counter += 1
    return mock_papers

# new_paper_data = [] # Already initialized earlier

if __name__ == "__main__":
    sci_inov_database = MilvusSciInovDB()
    sci_inov_database.init_collection(overwrite = True)
    
    expert_names_for_papers = [e["name"] for e in new_testing_data]
    generated_papers = generate_mock_papers(expert_names_for_papers)

    # Data for insertion should be a dict with keys 'experts' and 'papers'
    combined_data_to_insert = {
        "experts": new_testing_data,
        "papers": generated_papers
    }

    sci_inov_database.insert_item(combined_data_to_insert) 
    time.sleep(5) 
    # sci_inov_database.display() # display() needs to be adapted for two collections
    # Example of how to display (implement this in MilvusSciInovDB or call client.query directly):
    logging.info("--- Displaying Experts ---")
    expert_res = sci_inov_database.client.query(collection_name=sci_inov_database.expert_col_name, filter="", output_fields=sci_inov_database.display_field_experts, limit=5)
    for r in expert_res: logging.info(r)
    
    logging.info("--- Displaying Papers ---")
    paper_res = sci_inov_database.client.query(collection_name=sci_inov_database.paper_col_name, filter="", output_fields=sci_inov_database.display_field_papers, limit=10)
    for r in paper_res: logging.info(r)