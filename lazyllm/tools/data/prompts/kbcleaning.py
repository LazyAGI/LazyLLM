from .base_prompt import PromptABC


class KnowledgeCleanerPrompt(PromptABC):
    def __init__(self, lang: str = 'en', strict_mode: bool = True):
        self.lang = lang
        self.strict_mode = strict_mode

    def build_prompt(self, raw_content: str) -> str:
        if self.lang == 'en':
            self.prompt_header = f'''
As a precise Knowledge Processing Specialist, you must follow these guidelines RIGOROUSLY:

1. Eliminate unnecessary tags while keeping:
- Semantically meaningful tags such as <table>, <code>
- Attributes that carry significance

2. Standardize special characters:
- Normalize quotation marks and hyphenation
- Transform ellipsis marks (...)

3. URL management:
- Keep footnote URLs intact
- Retain display text content

4. Document structure:
- Preserve paragraph and list separations
- Maintain code indentation levels
- Restrict blank lines (maximum=2)

5. Reference handling:
- Images → "[Image: alt_text]"
- Signatures → "[Signature]"

6. Code sections: {"(strict)" if self.strict_mode else ""}
- {"Enforce closure" if self.strict_mode else "Maintain original"}
- Label incomplete sections as /*...*/

7. Complete accuracy:
- DO NOT alter facts or numbers
- DO NOT rephrase terminology
- DO NOT modify table layouts

8. Security measures:
- PII: Mask phone numbers/IDs/emails
- Classified: Label as 〖SEC∶classified〗
- Illegal: Substitute with 〖ILLEGAL∶removed〗

Response must be enclosed within <cleaned_start> and <cleaned_end>.
'''
        else:
            self.prompt_header = f'''
你是一位细致入微的知识处理专家。请严格遵循以下准则处理原始材料：

1. 删除多余的HTML/XML标记，但需保留：
- 具有语义价值的标签如 <table>、<code>、<formula>
- 所有包含意义的属性信息

2. 统一特殊字符格式：
- 将特殊引号转换为标准引号
- 将长横线改为短横线
- 将中文省略号改为英文省略号

3. 超链接管理：
- 脚注/参考文献中的URL维持不变
- 去除超链接标签但保留可见文本

4. 文档结构：
- 维持原有段落/列表的分隔
- 保持代码/引用的缩进结构

5. 引用元素处理：
- 图片引用改为【引用图片：描述文本】
- 签名区域标注为【签名引用】

6. 代码段处理：{"（严格模式）" if self.strict_mode else ""}

7. 完全保真：
- 不得增减任何事实信息、数值或实体名称

8. 安全措施：
- 个人敏感信息需要脱敏处理
- 机密内容替换为【涉密内容已加密】

输出内容需以<cleaned_start>开始，<cleaned_end>结束。
'''

        if self.lang == 'en':
            processing_steps = '''
Workflow Steps:
1. [Tag Classification] Categorize markup elements
2. [Reference Isolation] Separate images/tables
3. [Character Review] Document special characters
4. [Hierarchy Validation] Verify document structure
5. [Result Generation] Produce refined text
'''.strip()
            output_requirement = 'Response should include ONLY refined text enclosed by <cleaned_start> and <cleaned_end>.'
        else:
            processing_steps = '''
工作流程：
1. [标签归类] 识别并区分所有标记元素
2. [引用分离] 提取图片/表格/签名等引用部分
3. [字符审查] 记录特殊字符变化
4. [层级验证] 检查文档结构
5. [结果生成] 输出处理后的文本
'''.strip()
            output_requirement = '响应内容应仅包含处理后的文本，以<cleaned_start>开始，<cleaned_end>结束，不得包含其他内容。'

        return f'''
{self.prompt_header}

待清洗内容：
{raw_content}

{processing_steps}

{output_requirement}
'''.strip()


class MathbookQuestionExtractPrompt(PromptABC):
    def __init__(self):
        pass

    def build_prompt(self):
        return '''You are provided with a set of images:

• page_n.jpg – represents the n-th page of a mathematics textbook
• page_n+1.jpg – represents the (n+1)-th page of the same textbook
• index.jpg files (e.g. 1.jpg, 2.jpg, …) – contains all figures, diagrams or illustrations present on those two pages

Your assignment:

1. Identify every exercise (mathematical problem) that contains at least one line or component on page_n.jpg. \
You must extract the problem using its original language, without translation.
2. When a problem spans across page_n.jpg and page_n+1.jpg, include the complete problem (utilize page_n+1.jpg solely \
to finish it).
3. Exclude any problem that appears only on page_n+1.jpg.
4. For each identified problem, find any referenced figures within the index.jpg files and place \
the precise filename in <image>...</image> (for instance <image>3.jpg</image>) at the appropriate position \
within the problem text.
5. Combine all identified problems into a single string, using the literal token <SPACE> as separator. \
Example:
   PROBLEM_TEXT_1<SPACE>PROBLEM_TEXT_2<SPACE>PROBLEM_TEXT_3
6. If no eligible problems exist on page_n.jpg, return two consecutive spaces: "<SPACE><SPACE>".

Make sure figure tags precisely correspond to the provided image filenames and avoid adding extra separators or \
punctuation marks.
      '''


__all__ = [
    'KnowledgeCleanerPrompt',
    'MathbookQuestionExtractPrompt',
]
