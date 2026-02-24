from .base_prompt import PromptABC


class MultiHopQABuilderPrompt(PromptABC):
    '''Build multi-hop QA pairs requiring reasoning across multiple facts.'''
    def __init__(self, lang: str = 'en'):
        self.lang = lang
        self.system_text = self.build_system_prompt()

    def build_system_prompt(self) -> str:
        if self.lang == 'en':
            return '''\
                As an expert in multi-hop question answering, you must adhere to rigorous standards:

                █ Essential Guidelines
                1. Locate 2-3 interconnected pieces of information within the provided context
                2. Create sophisticated questions that demand reasoning across multiple facts
                3. Your reasoning process should:
                   - Include 2-3 sequential logical stages (with numbering)
                   - Demonstrate evident cause-effect or progressive connections
                   - Every stage needs to cite particular facts explicitly
                4. The ultimate response should integrate conclusions from all reasoning stages

                █ Response Format
                1. Output exclusively valid JSON following this exact structure:
                {
                    "question": "Multi-fact reasoning question",
                    "reasoning_steps": [
                        {"step": "First step (must use Fact 1)"},
                        {"step": "Second step (must link Fact 2)"}
                    ],
                    "answer": "Synthesized final answer",
                    "supporting_facts": ["Verbatim Fact 1", "Verbatim Fact 2"],
                    "type": "domain_tag"
                }
                2. Supporting facts need to:
                   - Match the original context word-for-word
                   - Provide direct evidence for their respective steps
                   - Paraphrasing is strictly forbidden

                █ Exclusion Rules
                Do not proceed if:
                - Reasoning steps number less than 2
                - Supporting facts appear without proper references
                - Content outside JSON format is present
                '''
        else:
            return '''\
                作为多跳问答领域的资深专家，您需要严格遵守以下操作规范：

                █ 基本准则
                1. 需要在给定上下文中找出2-3个相互关联的信息点
                2. 构建需要综合多个信息点进行推理的复杂问题
                3. 推理链条应当具备：
                    - 不少于2-3个有序的逻辑环节
                    - 每个环节需标注清晰序号
                    - 环节之间应体现因果关系或递进逻辑
                4. 最终回答需融合所有推理环节的结论

                █ 输出要求
                1. 只能输出符合以下结构的纯JSON格式：
                {
                    "question": "需要跨事实推理的问题",
                    "reasoning_steps": [
                        {"step": "第一推理步骤（必须引用事实1）"},
                        {"step": "第二推理步骤（必须关联事实2）"}
                    ],
                    "answer": "整合所有步骤的最终答案",
                    "supporting_facts": ["原文事实1", "原文事实2"],
                    "type": "领域标签"
                }

                █ 拒绝条件
                出现以下情况时应拒绝生成：
                - 推理环节数量不足2个
                - 支撑事实缺少对应引用
                - JSON格式外存在额外内容
                '''

    def build_prompt(self, text: str) -> str:
        if self.lang == 'en':
            return f'''\
            Create high-quality multi-hop question-answer pairs based on:

            Context:
            {text}

            Mandatory conditions:
            1. Identify precisely 2-3 connected pieces of information
            2. The question should require reasoning that spans multiple facts
            3. Follow this exact JSON format (preserve all quotes and braces):
            {{
                "question": "...",
                "reasoning_steps": [
                    {{"step": "Must explicitly use Fact 1"}},
                    {{"step": "Must explicitly link Fact 2"}}
                ],
                "answer": "...",
                "supporting_facts": ["Verbatim Fact 1", "Verbatim Fact 2"],
                "type": "..."
            }}
            '''
        else:
            return f'''\
                请根据以下文本内容创建高质量的多跳问答对：

                文本内容：
                {text}

                必须满足的条件：
                1. 从上述文本中准确提取2-3个相互关联的信息点
                2. 生成的问题应当展现跨信息点推理的复杂特征
                3. 严格按照以下JSON格式输出：
                {{
                    "question": "...",
                    "reasoning_steps": [
                        {{"step": "必须明确引用事实1"}},
                        {{"step": "必须明确关联事实2"}}
                    ],
                    "answer": "...",
                    "supporting_facts": ["事实1原文", "事实2原文"],
                    "type": "..."
                }}
            '''


__all__ = [
    'MultiHopQABuilderPrompt',
]
