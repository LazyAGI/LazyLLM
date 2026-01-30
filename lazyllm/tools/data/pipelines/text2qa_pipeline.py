from lazyllm import pipeline
from lazyllm.tools.data import Text2qa

def build_generate_pipeline():
    with pipeline() as ppl:
        ppl.text_to_chunks = Text2qa.text_to_chunks()
        ppl.chunk_to_qa = Text2qa.chunk_to_qa()
        ppl.qa_scorer = Text2qa.qa_scorer()
        ppl.qa_filter = Text2qa.score_filter()
    return ppl

def build_regenerate_pipeline():
    with pipeline() as ppl:
        # 输入drop的qa对，再根据里面的chunk进行重新生成
        ppl.chunk_to_qa = Text2qa.chunk_to_qa()
        ppl.qa_scorer = Text2qa.qa_scorer()
        ppl.qa_filter = Text2qa.score_filter()
    return ppl

def build_text2qa_pipeline(input):
    generate_ppl = build_generate_pipeline()
    regenerate_ppl = build_regenerate_pipeline()
    keep, drop = generate_ppl(input)

    while drop:
        new_keep, new_drop = regenerate_ppl(drop)
        keep.extend(new_keep)
        drop = new_drop
    
    return keep



if __name__ == "__main__":
    test_data = {
        "content": """# 站场工程技术规范
本章节依据《铁路车站及枢纽设计规范》（TB 10091-2017）编制。在格尔木南站的改造工程中，站场扩建必须严格遵守地基承载力要求。\n
根据设计图纸（图号：GK-ZH-04），1号至5号到发线的有效长度应统一调整为1050米。在信号楼布线方面，所有电缆径路应避开排水沟。对于施工中发现的软土路基，必须立即启动应急预案。\n
随便什么xxx。
""",
        "file_name": 'test.md'
    }
    print("开始执行 Pipeline 测试...\n")
    qas = build_text2qa_pipeline(test_data)
    print(qas)


