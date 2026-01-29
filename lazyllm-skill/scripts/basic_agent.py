from lazyllm.common.utils import compile_func
from lazyllm import OnlineModule, WebModule
from lazyllm.tools import CodeGenerator, ReactAgent, fc_register

@fc_register('tool')
def generate_code_from_query(query: str) -> str:
    '''
    Generate and execute Python code to fulfill a user's natural language request.

    This tool uses LLM to generate a single-function Python script according to the user's query.
    The generated function will then be safely compiled and executed, and the result (e.g., image path)
    will be returned directly.

    Args:
        query (str): The natural language instruction from the user,
                     for example: "Draw a temperature change chart of Beijing in the past month".

    Returns:
        str: The execution result of the generated function (e.g., image path or computed value).
    '''
    prompt = '''
    请生成一个仅包含单个函数定义的 Python 代码，用于完成用户的需求。

    编写要求如下：
    1. 不允许导入或使用以下模块：requests、os、sys、subprocess、socket、http、urllib、pickle 等。
    2. 仅可使用 matplotlib、datetime、random、math 等安全标准库。
    3. 如果任务涉及网络请求或外部 API，请使用随机数或固定数据进行模拟。
    4. 函数必须有明确的返回值，并返回最终结果（如图片路径或计算结果）。
    5. 如果是绘图任务，绘图时禁止使用中文字符（标题、坐标轴、标签均使用英文），
    请将图片保存到路径 `/home/mnt/WorkDir/images` 中，
    返回值必须是图片的完整保存路径。
    6. 代码中不得包含函数调用示例或打印语句。
    '''
    gen = CodeGenerator(llm, prompt)
    code = gen(query)

    compiled_func = compile_func(code)

    try:
        result = compiled_func()
    except Exception as e:
        result = f'执行生成代码时出错: {e}'

    return result

llm = OnlineModule(source='deepseek', model='deepseek-chat')
agent = ReactAgent(llm, tools=['generate_code_from_query'])
WebModule(agent, port=12347, title='Code Agent', static_paths='/home/mnt/WorkDir/images').start().wait()
