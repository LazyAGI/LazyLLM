import json
from typing import Any, Dict

from .base_prompt import PromptABC

DOMAIN_PRESETS: Dict[str, Dict[str, Any]] = {
    'finance': {
        'instruction_zh': '你是一位专业的金融分析师。请根据给定的上下文，提供准确、专业且合规的金融分析或解答。',
        'instruction_en': (
            'You are a professional financial analyst. Please provide accurate, '
            'professional and compliant financial analysis or answers based on '
            'the given context.'
        ),
        'filters': [
            {'type': 'word_count', 'min_words': 20, 'max_words': 5000},
            {'type': 'char_count', 'min_chars': 50, 'max_chars': 50000},
            {'type': 'null_content'},
        ],
        'pretrain_keywords': [
            '股票', '债券', '投资', '财报', '金融', '货币', '利率', '汇率',
            '基金', '期货', '期权', '资产', '信贷', '风控', '银行', '证券',
            '保险', '上市', '融资', '估值', '收益', '分红', '市盈率', '杠杆',
            '通胀', '央行', 'GDP', '财政', '税收', '审计',
        ],
        'pretrain_filters': [
            {'type': 'word_count', 'min_words': 50, 'max_words': 50000},
            {'type': 'char_count', 'min_chars': 200, 'max_chars': 500000},
            {'type': 'null_content'},
        ],
    },
    'medical': {
        'instruction_zh': '你是一位医疗信息助手。请注意：提供的信息仅供参考，不构成医疗建议、诊断或治疗。如有医疗问题，请咨询专业医生。',
        'instruction_en': (
            'You are a medical information assistant. Please note: The information '
            'provided is for reference only and does not constitute medical advice, '
            'diagnosis or treatment. Please consult a professional doctor for medical issues.'
        ),
        'filters': [
            {'type': 'word_count', 'min_words': 30, 'max_words': 10000},
            {'type': 'char_count', 'min_chars': 100, 'max_chars': 100000},
            {'type': 'null_content'},
        ],
        'pretrain_keywords': [
            '疾病', '症状', '治疗', '药物', '医院', '医生', '患者', '健康',
            '诊断', '手术', '处方', '临床', '病理', '免疫', '感染', '并发症',
            '预后', '康复', '护理', '检查', '化验', '影像', '病毒', '细菌',
            '抗生素', '疫苗', '基因', '遗传', '中医', '西医',
        ],
        'pretrain_filters': [
            {'type': 'word_count', 'min_words': 50, 'max_words': 80000},
            {'type': 'char_count', 'min_chars': 200, 'max_chars': 800000},
            {'type': 'null_content'},
        ],
    },
    'legal': {
        'instruction_zh': '你是一位法律信息助手。请注意：提供的信息仅供参考，不构成法律建议或意见。如有具体法律问题，请咨询专业律师。',
        'instruction_en': (
            'You are a legal information assistant. Please note: The information '
            'provided is for reference only and does not constitute legal advice or '
            'opinion. Please consult a professional lawyer for specific legal issues.'
        ),
        'filters': [
            {'type': 'word_count', 'min_words': 40, 'max_words': 15000},
            {'type': 'char_count', 'min_chars': 150, 'max_chars': 150000},
            {'type': 'null_content'},
        ],
        'pretrain_keywords': [
            '法律', '合同', '诉讼', '法院', '律师', '法规', '条款', '侵权',
            '刑法', '民法', '行政法', '宪法', '仲裁', '判决', '裁定', '起诉',
            '被告', '原告', '证据', '辩护', '赔偿', '违约', '知识产权', '商标',
            '专利', '著作权', '劳动法', '合规', '监管', '执法',
        ],
        'pretrain_filters': [
            {'type': 'word_count', 'min_words': 80, 'max_words': 100000},
            {'type': 'char_count', 'min_chars': 300, 'max_chars': 1000000},
            {'type': 'null_content'},
        ],
    },
    'education': {
        'instruction_zh': '你是一位专业的教育专家。请针对以下问题提供清晰、准确且有建设性的教育指导、解释或解答。',
        'instruction_en': (
            'You are a professional education expert. Please provide clear, accurate '
            'and constructive educational guidance, explanations or answers to the '
            'following question.'
        ),
        'filters': [
            {'type': 'word_count', 'min_words': 10, 'max_words': 8000},
            {'type': 'char_count', 'min_chars': 30, 'max_chars': 80000},
            {'type': 'null_content'},
        ],
        'pretrain_keywords': [
            '教育', '学习', '教学', '课程', '学生', '教师', '学校', '考试',
            '知识', '培训', '教材', '作业', '成绩', '毕业', '学位', '论文',
            '研究', '实验', '思维', '方法论', '素质教育', '高考', '考研',
        ],
        'pretrain_filters': [
            {'type': 'word_count', 'min_words': 30, 'max_words': 50000},
            {'type': 'char_count', 'min_chars': 100, 'max_chars': 500000},
            {'type': 'null_content'},
        ],
    },
    'customer_service': {
        'instruction_zh': '你是一位专业的客服人员。请以友善、专业的态度回答客户的问题，提供准确、有帮助的信息。',
        'instruction_en': (
            'You are a professional customer service representative. Please provide '
            'helpful, polite and accurate responses to customer inquiries.'
        ),
        'filters': [
            {'type': 'word_count', 'min_words': 5, 'max_words': 3000},
            {'type': 'char_count', 'min_chars': 10, 'max_chars': 30000},
            {'type': 'null_content'},
        ],
        'pretrain_keywords': [
            '客服', '服务', '客户', '投诉', '咨询', '订单', '售后', '支持',
            '退款', '换货', '保修', '反馈', '满意度', '工单', '响应',
        ],
        'pretrain_filters': [
            {'type': 'word_count', 'min_words': 10, 'max_words': 20000},
            {'type': 'char_count', 'min_chars': 30, 'max_chars': 200000},
            {'type': 'null_content'},
        ],
    },
    'general': {
        'instruction_zh': '你是一个有帮助的助手。请回答以下问题。',
        'instruction_en': 'You are a helpful assistant. Please answer the following question.',
        'filters': None,
        'pretrain_keywords': [],
        'pretrain_filters': None,
    },
}


DOMAIN_INSTRUCTION_EN: Dict[str, str] = {
    'finance': (
        'You are a professional financial analyst. Please provide accurate and '
        'professional financial analysis or answers based on the given context.'
    ),
    'medical': (
        'You are a medical information assistant. The information provided is for '
        'reference only and does not constitute medical advice. Please consult a '
        'professional doctor for medical issues.'
    ),
    'legal': (
        'You are a legal information assistant. The information provided is for '
        'reference only and does not constitute legal advice. Please consult a '
        'professional lawyer for specific legal issues.'
    ),
    'education': (
        'You are a professional education expert. Please provide clear, accurate and '
        'constructive educational guidance or answers.'
    ),
    'customer_service': (
        'You are a professional customer service representative. Please provide '
        'helpful, polite and accurate responses to customer inquiries.'
    ),
    'general': 'You are a helpful assistant. Please answer the following question.',
}


class DomainFinetuneExtractionPrompt(PromptABC):
    def __init__(self, lang: str = 'zh', extract_format: str = 'qa', num_samples: int = 3):
        self.lang = lang
        self.extract_format = extract_format
        self.num_samples = num_samples

    def build_system_prompt(self) -> str:
        if self.lang in ('zh', 'cn', 'chinese'):
            return (
                '你是一名擅长从垂直领域文档中构造训练样本的数据标注专家。'
                '请严格根据用户输入中的说明，从给定文本中抽取高质量的问答对或指令-输出对，'
                '并以 JSON 格式输出。'
            )
        return (
            'You are a data annotation expert skilled at constructing training samples '
            'from domain-specific documents. Follow the user instructions to extract '
            'high-quality QA or instruction-output pairs and output them in JSON format.'
        )

    def _build_qa_template(self, is_zh: bool) -> str:
        if is_zh:
            return (
                '请从以下垂直领域文本中提取{num_samples}个高质量的问答对，用于大语言模型微调。\n'
                '要求：\n'
                '1. 问题应具体、自然，有实际价值，源于文本内容\n'
                '2. 答案应完整、准确，直接基于文本，不要添加文本以外的内容\n'
                '3. 问答对应覆盖文本的不同要点\n'
                '4. 严格按JSON格式输出：{{"qa_pairs": [{{"question": "...", "answer": "..."}}]}}\n\n'
                '文本内容：\n{text}\n\n'
                '直接输出JSON，不要添加任何说明或注释：'
            )
        return (
            'Extract {num_samples} high-quality question-answer pairs from the following domain text '
            'for LLM fine-tuning.\n'
            'Requirements:\n'
            '1. Questions should be specific, natural, and valuable, derived from the text\n'
            '2. Answers should be complete and accurate, directly based on the text\n'
            '3. QA pairs should cover different key points in the text\n'
            '4. Output strictly in JSON format: {{"qa_pairs": [{{"question": "...", "answer": "..."}}]}}\n\n'
            'Text content:\n{text}\n\n'
            'Output JSON directly without any explanation:'
        )

    def _build_instruction_template(self, is_zh: bool) -> str:
        if is_zh:
            return (
                '请从以下垂直领域文本中提取{num_samples}个高质量的指令-输出对，用于大语言模型微调。\n'
                '要求：\n'
                '1. instruction 为清晰的任务描述或系统指令\n'
                '2. input 为用户的具体输入或问题（可为空字符串）\n'
                '3. output 为高质量的模型输出，内容准确、来源于文本\n'
                '4. 严格按JSON格式输出：{{"samples": [{{"instruction": "...", "input": "...", "output": "..."}}]}}\n\n'
                '文本内容：\n{text}\n\n'
                '直接输出JSON，不要添加任何说明或注释：'
            )
        return (
            'Extract {num_samples} high-quality instruction-output pairs from the following domain text '
            'for LLM fine-tuning.\n'
            'Requirements:\n'
            '1. instruction: a clear task description or system prompt\n'
            '2. input: specific user input or question (can be empty string)\n'
            '3. output: high-quality model response, accurate and grounded in the text\n'
            '4. Output strictly in JSON format: '
            '{{"samples": [{{"instruction": "...", "input": "...", "output": "..."}}]}}\n\n'
            'Text content:\n{text}\n\n'
            'Output JSON directly without any explanation:'
        )

    def build_prompt(self, text: str, max_input_chars: int = 3000) -> str:
        is_zh = self.lang in ('zh', 'cn', 'chinese')
        tpl = (
            self._build_qa_template(is_zh)
            if self.extract_format == 'qa'
            else self._build_instruction_template(is_zh)
        )
        return tpl.format(num_samples=self.num_samples, text=text[:max_input_chars])


class DomainFinetuneFieldMappingPrompt(PromptABC):
    def __init__(self, lang: str = 'zh'):
        self.lang = lang

    def build_system_prompt(self) -> str:
        if self.lang in ('zh', 'cn', 'chinese'):
            return (
                '你是一名熟悉多种数据集格式的大语言模型微调数据工程师。'
                '你需要将不同来源、不同字段命名的数据记录统一映射到 '
                'instruction / input / output 三个字段。'
            )
        return (
            'You are a data engineer familiar with various dataset formats for LLM fine-tuning. '
            'Your task is to map heterogeneous records to the standard instruction / input / output format.'
        )

    def build_prompt(self, record: dict, max_data_chars: int = 2000) -> str:
        data_str = json.dumps(record, ensure_ascii=False, indent=2)
        data_str = data_str[:max_data_chars]
        if self.lang in ('zh', 'cn', 'chinese'):
            template = (
                '以下是一条来自垂直领域数据集的记录（JSON格式）。请分析各字段的语义含义，'
                '将其映射到大语言模型微调所需的标准格式（instruction, input, output）。\n\n'
                '数据记录：\n{data_str}\n\n'
                '请以JSON格式输出映射结果：\n'
                '{{"instruction": "任务描述或系统指令", "input": "用户输入或问题（可为空字符串）", '
                '"output": "期望的模型输出或答案"}}\n\n'
                '映射规则：\n'
                '- instruction：提取或生成简洁的任务描述\n'
                '- input：提取用户的问题/查询内容（若无合适字段则留空）\n'
                '- output：提取期望的答案/回复/结论\n\n'
                '直接输出JSON，不要任何额外说明：'
            )
        else:
            template = (
                'The following is a record from a domain-specific dataset (JSON format). '
                'Analyze the semantic meaning of each field and map it to the standard LLM '
                'fine-tuning format (instruction, input, output).\n\n'
                'Data record:\n{data_str}\n\n'
                'Output mapping in JSON format:\n'
                '{{"instruction": "task description or system prompt", '
                '"input": "user input or question (can be empty string)", '
                '"output": "expected model output or answer"}}\n\n'
                'Mapping rules:\n'
                '- instruction: extract or generate a concise task description\n'
                '- input: extract user query/question (leave empty if no suitable field)\n'
                '- output: extract the expected answer/response/conclusion\n\n'
                'Output JSON directly without any explanation:'
            )
        return template.format(data_str=data_str)


__all__ = [
    'DOMAIN_PRESETS',
    'DOMAIN_INSTRUCTION_EN',
    'DomainFinetuneExtractionPrompt',
    'DomainFinetuneFieldMappingPrompt',
]
