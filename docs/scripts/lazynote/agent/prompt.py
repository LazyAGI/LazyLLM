"""
存放 GitAgent 使用的各种 prompt
"""


README_PROMPT = """请根据以下项目结构生成一个标准的 README.md 文件：  

项目结构：  
{project_structure}  

请尽可能涵盖以下部分（根据项目特点酌情取舍）：  
1. 项目名称和简介：简要说明项目的主要功能和应用场景。  
2. 功能特性：列出项目的核心功能和亮点。  
3. 安装说明：提供安装步骤和依赖要求。  
4. 使用方法：描述如何运行或使用项目，包含示例代码或命令。  
5. 项目结构：展示项目的目录结构和文件组织方式。 

注意：
1. 请使用 **{language}** 生成内容，并合理组织，确保简明清晰且实用。  
2. 可调用工具获取模块的文档字符串，辅助生成内容。
3. 直接输出可写入 README 文件的内容，不包含任何其他说明。
"""


GITIGNORE_PROMPT = """请为以下类型的项目生成一个标准的 .gitignore 文件：

项目信息：
{project_info}

请包含：
1. 语言特定的忽略规则
2. IDE 配置文件
3. 操作系统生成的文件
4. 构建输出和缓存文件
5. 环境文件

注意：
1. 直接输出可以粘贴到.gitignore 文件中的内容, 不要包含任何其他说明
"""

LICENSE_PROMPT = """请为这个开源项目推荐一个合适的开源许可证，并提供完整的许可证文本。
建议考虑：
1. MIT License
2. Apache License 2.0
3. GNU GPL v3
4. BSD License
"""


MKDOCS_PROMPT = """你是一个专业的 Python 文档生成助手。请根据项目的结构信息按照以下要求为项目自动生成文档：  

1. **生成模块 API 文档：**  
   - 通过模块名调用相关工具获取模块文档，并生成每个模块的独立 Markdown 文件（`.md`），可按需将多个类合并至同一文件。  
   - 文档应包含以下部分（根据模块内容灵活调整）：  
     - 模块名称与简介  
     - 类和方法列表  
     - 每个类和方法的详细说明（包括参数和返回值）  
     - 示例代码（如有）  
   - 使用工具调用，将生成的文档保存到 `docs/{language}/API` 目录，合理组织目录结构。  

2. **生成最佳实践文档：**
   - 根据你对项目的理解，生成项目的最佳实践文档(可生成多个)，文档文件格式为 Markdown（`.md`）：
   - 使用工具调用，将生成的文档保存到 `docs/{language}/BestPractice` 目录，并合理组织目录结构。  

3. **生成主页文档：**
   - 生成项目文档的首页文件，‘index.md’：
     - 项目名称和简介
     - 文档结构概览
     - 快速开始指引（如有）
   - 使用工具调用，将生成的文档保存到 `docs/{language}` 目录。  

4. **创建 MkDocs 配置文件：**  
   - 按照给定模板生成一个标准的 `mkdocs.yml` 文件：    
     - 文档导航设置，需包含已经生成的全部的 API 文档和最佳实践文档。
     - 导航标签一律设置为英文。
     - 导航目录为相对于 `docs/{language}` 的路径。 
   - 使用工具调用，将`mkdocs.yml` 配置文件保存到项目的根目录下。 
   
**注意：**  
- 使用 {language_type} 生成内容。
- 生成的文档和配置文件应符合 MkDocs 规范，确保能通过 `mkdocs serve` 正常预览和构建。
- 由于工具调用获取模块文档可能较长，建议自行逐个模块处理，避免超出 token 限制。

**项目结构信息：**   
{project_structure}
MkDocs 配置文件模板：
{mkdocs_config}
"""


TRANSLATE_PROMPT = """请将以下Markdown文档从中文翻译成英文，保持所有Markdown格式和代码块不变，只翻译文本内容：
    
{zh_content}

要求：
1. 保持所有Markdown语法格式
2. 保持所有代码块内容不变
3. 只翻译中文文本内容, 且不翻译模块名、注释和代码块
4. 保持原有的文档结构
5. 使用专业的技术文档语言风格
"""


def generate_mkdocs_config(site_name, docs_dir):
    mkdocs_config = {
            'site_name': site_name,
            'site_description': f'API documentation for {site_name}',
            'docs_dir': docs_dir,

            'theme': {
                'name': 'material',
                'palette': {
                    'primary': 'indigo',
                    'accent': 'pink'
                },
                'font': {
                    'text': 'Roboto',
                    'code': 'Roboto Mono'
                }
            },

            'nav': [
                {'Home': 'index.md'},
                {'API Reference': []}
            ],
        }
    return mkdocs_config
  
  
  
CUSTOM_KMDOCS_PROMPT = """你是一个专业的 Python 文档生成助手，负责根据项目信息按照要求为项目自动生成文档，你有。

# 工具：
  - get_module_doc：根据模块名获取模块的 docstring，用于获取模块信息以便理解项目。
  - write_doc：将给定内容写入指定路径的文件，用于写生成的文档。

你需要按照合适的顺序使用工具，逐步完成任务。可能需要将任务分解为子任务，依次调用工具获取模块文档、生成文档文件并进行组织。

# 输出格式
请使用与问题相同的语言回答，并遵循以下格式：

想法：用户当前的语言是：（用户的语言）。我需要使用工具来帮助回答问题。

{TOKENIZED_PROMPT}

答案：在此处填写你的答案（使用与用户问题相同的语言）。

当前对话：以下是当前对话，由人工和助手消息交织而成。请逐步思考。

文档生成要求
模块 API 文档：

通过调用 get_module_doc 获取模块 docstring。

包含模块名称、简介、类和方法列表、详细说明和示例代码。

使用 write_doc 将生成的文档保存至 docs/{language}/API 目录，合理组织目录结构。

最佳实践文档：

基于项目特性生成最佳实践文档，使用 write_doc 保存至 docs/{language}/BestPractice 目录。

首页文档：

包含项目名称、简介、文档结构和快速开始指引，使用 write_doc 保存至 docs/{language}/index.md。

MkDocs 配置文件：

生成标准的 mkdocs.yml 配置文件，包含 API 文档和最佳实践文档的导航，使用 write_doc 保存至 docs 目录。
  """

PLUGIN_CONFIG= [
      {'i18n': 
          {'docs_structure': 'folder','languages': 
              [
                  {
                      'locale': 'en',
                      'default': True,
                      'name': 'English',
                      'build': True
                  },
                  {
                      'locale': 'zh',
                      'name': '简体中文',
                      'build': True
                  }
              ]
          }
      }
  ]