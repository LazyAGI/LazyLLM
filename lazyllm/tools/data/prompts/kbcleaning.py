'''Prompts for knowledge cleaning pipeline operators'''
from .base_prompt import PromptABC


class KnowledgeCleanerPrompt(PromptABC):
    '''
    知识清洗提示词生成器，支持中英文多语言适配
    Specialized in refining raw content with multilingual support.
    '''
    def __init__(self, lang: str = 'en', strict_mode: bool = True):
        self.lang = lang
        self.strict_mode = strict_mode

    def build_prompt(self, raw_content: str) -> str:
        '''生成知识清洗的思维链提示词'''
        if self.lang == 'en':
            self.prompt_header = f'''
You are a meticulous Knowledge Refinement Engineer. Apply these rules STRICTLY:

1. Remove redundant tags but retain:
- Semantic tags like <table>, <code>
- Meaningful attributes

2. Normalize special characters:
- Standardize quotes and dashes
- Convert ellipsis (...)

3. URL handling:
- Preserve footnote URLs
- Extract display texts

4. Text structure:
- Maintain paragraph/list breaks
- Keep code indentation
- Limit empty lines (max=2)

5. Reference processing:
- Images → "[Image: alt_text]"
- Signatures → "[Signature]"

6. Code blocks: {"(strict)" if self.strict_mode else ""}
- {"Force closure" if self.strict_mode else "Preserve raw"}
- Mark fragments as /*...*/

7. Absolute fidelity:
- NO fact/number modifications
- NO term paraphrasing
- NO table structure changes

8. Security Processing:
- PII: Phone/ID/Email must be masked
- Classified: Mark as 〖SEC∶classified〗
- Illegal: Replace with 〖ILLEGAL∶removed〗

Output must be between <cleaned_start> and <cleaned_end>.
'''
        else:
            self.prompt_header = f'''
你是一名严谨的知识清洗工程师。请严格按照以下规则处理原始内容：

1. 移除冗余HTML/XML标签，但保留：
- 语义化标签如 <table>、<code>、<formula>
- 所有携带意义的属性值

2. 规范化特殊字符：
- 将花引号转为标准引号
- 将长破折号替换为短横线
- 中文省略号转为英文省略号

3. 链接处理：
- 脚注/参考文献中的URL保持原样
- 移除超链接包装但保留显示文本

4. 文本结构：
- 保持原始段落/列表的换行
- 保留代码/引用的缩进层级

5. 引用内容处理：
- 图片引用转换为【引用图片：描述文本】
- 签名区块标记为【签名引用】

6. 代码块处理：{"（严格模式）" if self.strict_mode else ""}

7. 绝对保真：
- 禁止增删任何事实、数字或命名实体

8. 安全处理：
- 个人隐私需脱敏
- 涉密内容替换为【涉密内容已加密】

输出必须以<cleaned_start>开头，<cleaned_end>结尾。
'''

        if self.lang == 'en':
            processing_steps = '''
Processing Steps:
1. [Tag Analysis] Classify markup tags
2. [Reference Extraction] Isolate images/tables
3. [Character Audit] Log special chars
4. [Structure Check] Validate hierarchy
5. [Final Output] Generate cleaned text
'''.strip()
            output_requirement = 'Response must contain ONLY cleaned text between <cleaned_start> and <cleaned_end>.'
        else:
            processing_steps = '''
处理步骤：
1. [标签分析] 识别并分类所有标记标签
2. [引用提取] 分离图片/表格/签名等引用内容
3. [字符审核] 记录特殊字符变更
4. [结构检查] 验证文本层级
5. [最终输出] 生成清洗后文本
'''.strip()
            output_requirement = '响应必须只包含清洗后文本，以<cleaned_start>开头，<cleaned_end>结尾，无其他内容。'

        return f'''
{self.prompt_header}

待清洗内容：
{raw_content}

{processing_steps}

{output_requirement}
'''.strip()


class MathbookQuestionExtractPrompt(PromptABC):
    '''Prompt for extracting questions from math textbook images.'''
    def __init__(self):
        pass

    def build_prompt(self):
        return '''You are given a collection of images:

• page_n.jpg – the n-th page of a math textbook
• page_n+1.jpg – the (n+1)-th page of the same book
• index.jpg files (e.g. 1.jpg, 2.jpg, …) – all figures, diagrams or illustrations appearing on those two pages

Your task:

1. Extract every exercise (math problem) that has at least one line or element on page_n.jpg. \
You should extract the problem in its original language, do not translate it.
2. If a problem is split across page_n.jpg and page_n+1.jpg, include it in full (using page_n+1.jpg only \
to complete it).
3. Do not extract any problem that appears exclusively on page_n+1.jpg.
4. For each extracted problem, locate any referenced figures among the index.jpg files and insert \
the exact filename in <image>...</image> (for example <image>3.jpg</image>) at the correct place \
in the problem text.
5. Return all extracted problems concatenated into one string, using the literal token <SPACE> to separate them. \
For example:
   PROBLEM_TEXT_1<SPACE>PROBLEM_TEXT_2<SPACE>PROBLEM_TEXT_3
6. If no qualifying problems are found on page_n.jpg, return two consecutive spaces: "<SPACE><SPACE>".

Ensure that figure tags exactly match the provided image filenames and that no extra separators or \
punctuation are added.
      '''


__all__ = [
    'KnowledgeCleanerPrompt',
    'MathbookQuestionExtractPrompt',
]
