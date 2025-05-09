"""
存放 GitAgent 使用的各种 prompt
"""

# README 生成的 prompt 模板
README_PROMPT = """请为以下项目生成一个标准的 README.md 文件：
        
项目结构：
{project_structure}

请包含以下部分：
1. 项目名称和简介
2. 功能特性
3. 安装说明
4. 使用方法
5. 项目结构
6. 贡献指南
7. 许可证信息
"""

# .gitignore 生成的 prompt 模板
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

# LICENSE 生成的 prompt 模板
LICENSE_PROMPT = """请为这个开源项目推荐一个合适的开源许可证，并提供完整的许可证文本。
建议考虑：
1. MIT License
2. Apache License 2.0
3. GNU GPL v3
4. BSD License
"""