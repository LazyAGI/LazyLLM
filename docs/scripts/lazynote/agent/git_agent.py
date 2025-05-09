import os
import sys
from typing import List, Dict, Optional
import json
import yaml
import importlib
import inspect
import pkgutil
from lazyllm.module.onlineChatModule.onlineChatModule import OnlineChatModule
from lazynote.editor import BaseEditor
from lazynote.manager.llm_manager import LLMDocstringManager
from .prompt import README_PROMPT, GITIGNORE_PROMPT, LICENSE_PROMPT

class GitAgent:
    def __init__(self, project_path: str):
        self.project_path = os.path.abspath(project_path)
        self.docstring_manager = LLMDocstringManager(pattern='fill', skip_on_error=True)
        self.module_dict = {}  
        if not os.path.exists(self.project_path):
            raise ValueError(f"项目路径不存在: {self.project_path}")
        if self.project_path not in sys.path:
            sys.path.append(self.project_path)
        self.llm = OnlineChatModule(source='deepseek', stream=False)

    def standardize_project(self, if_gen_docs = True) -> None:
        """
        将项目标准化为 Git 项目
        """
        self._gen_module_dict() 
        if if_gen_docs:
            self._generate_docs() 
            self._update_module_dict()
        self._generate_mkdocs()
        self._generate_readme()
        self._generate_requirements()
        self._generate_gitignore()
        # self._generate_license()
        print("✨项目标准化完成")
        
    
    def _update_module_dict(self):
        """
        更新项目路径下所有模块的docstring
        """
        project_modules = self.module_dict.keys()
        modules_to_del = []
        for name in sys.modules:
            if any(name.startswith(mod) for mod in project_modules):
                modules_to_del.append(name)
        for module in modules_to_del:
            del sys.modules[module]
        self._gen_module_dict()
    
    def _gen_module_dict(self):
        """
        读取项目路径下所有模块信息
        """
        print("😊 正在分析项目结构...")
        processed_packages = set()
        for root, dirs, files in os.walk(self.project_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'tests', 'docs']]

            if '__init__.py' in files:
                rel_path = os.path.relpath(root, self.project_path)
                module_name = rel_path.replace(os.sep, '.')
                try:
                    module = importlib.import_module(module_name)
                    skip_modules = ['docs', 'test']
                    if '.'.join(module_name.split('.')[:-1]) in processed_packages:
                        continue
                    self.module_dict |= self._process_package(module, skip_modules)       
                    processed_packages.add(module_name)
                except Exception as e:
                    print(f"处理包 {module_name} 时出错: {str(e)}")

            for file in files:
                if file.endswith('.py') and file not in {'setup.py', 'main.py', '__init__.py', 'conftest.py', 'wsgi.py', 'asgi.py'}:
                    rel_dir = os.path.relpath(root, self.project_path)
                    package_name = rel_dir.replace(os.sep, '.')
                    if package_name in processed_packages:
                        continue
                    print(f"处理模块 {file}")
                    
                    module_path = os.path.join(root, file)
                    module_name = os.path.splitext(file)[0]
                    try:
                        spec = importlib.util.spec_from_file_location(module_name, module_path)
                        module = importlib.util.module_from_spec(spec)
                        self.module_dict |= self._process_module(module)
                                    
                    except Exception as e:
                        print(f"处理模块 {module_name} 时出错: {str(e)}")
        
        print("✅ 项目结构分析完成...") 
    
    def _process_module(self, module, f_module_name=""):
        m_dict = {}
        for name, obj in inspect.getmembers(module, inspect.isclass):
            m_dict[f"{f_module_name}.{name}"] = {'doc':obj.__doc__, 'obj': obj}
            for method_name, method_obj in inspect.getmembers(obj, inspect.isfunction):
                m_dict[f"{f_module_name}.{name}.{method_name}"] = {'doc':method_obj.__doc__, 'obj': method_obj}
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            m_dict[f"{f_module_name}.{name}"] = {'doc':obj.__doc__, 'obj': obj}
        return m_dict
            
    def _process_package(self, module, skip_modules):
        m_dict = {}
        m_dict[module.__name__] = {'obj': module}
        m_dict[module.__name__]['children'] = {}
        processed_module = set()
        for importer, modname, ispkg in pkgutil.walk_packages(module.__path__, module.__name__ + "."):
            if any(modname.startswith(skip_mod) for skip_mod in skip_modules):
                continue
            if ispkg:
                m_dict[module.__name__]['children'] |= self._process_package(importer.find_module(modname).load_module(modname), skip_modules)
                processed_module.add(modname)
                continue
            try:
                submodule = importlib.import_module(modname)
                if any(modname.startswith(mod) for mod in processed_module):
                    continue
                m_dict[module.__name__]['children'] |= self._process_module(submodule, f_module_name=modname)
            except Exception as e:
                print(f"Skipping {modname} due to import error", e)
        return m_dict
    
    def _generate_docs(self) -> None:
        """生成文档"""
        print("😊 正在自动生成注释...")
        for _, module in self.module_dict.items():
            if 'children' in module:
                self.docstring_manager.traverse(module['obj'])
            else:
                self.docstring_manager.modify_docstring(module['obj'])
        print("✅ 注释生成已完成...")

    def _generate_mkdocs(self) -> None:
        """生成 mkdocs 文档"""
        docs_dir = os.path.join(self.project_path, "docs")
        if not os.path.exists(docs_dir):
            os.makedirs(docs_dir)
        
        docs_dir_zh = os.path.join(docs_dir, "zh")
        if not os.path.exists(docs_dir_zh):
            os.makedirs(docs_dir_zh)
            
        api_dir = os.path.join(docs_dir_zh, "api")
        os.makedirs(api_dir, exist_ok=True)
        
        # 生成 mkdocs.yml 配置文件
        mkdocs_config = {
            'site_name': os.path.basename(self.project_path),
            # 'site_url': f'https://{os.path.basename(self.project_path)}.docs',
            # 'site_author': 'GitAgent',
            'site_description': f'API documentation for {os.path.basename(self.project_path)}',
            # 'copyright': f'© {datetime.now().year} GitAgent',
            'docs_dir': f'{docs_dir_zh}',

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
                {'API Reference': []}  # 将在处理模块时填充
            ],
        }
        
        # 生成首页
        index_content = f"""# {os.path.basename(self.project_path)}

这是由 GitAgent 自动生成的项目文档。

## API 文档

请查看 [API Reference](api/) 部分获取详细的API文档。
"""
        with open(os.path.join(docs_dir_zh, "index.md"), "w", encoding="utf-8") as f:
            f.write(index_content)
        
        def process_module_dict(module_dict, nav_list=None):
            if nav_list is None:
                nav_list = []
            
            for name, info in module_dict.items():
                module_path = name.replace('.', '/')
                module_dir = os.path.join(api_dir, module_path)
                os.makedirs(os.path.dirname(module_dir), exist_ok=True)
                
                # 生成模块文档
                with open(f"{module_dir}.md", "w", encoding="utf-8") as f:
                    f.write(f"# {name}\n\n")
                    if info.get('doc'):
                        f.write(f"{info['doc']}\n\n")
                    
                    f.write("## API 文档\n\n")
                    
                    if 'children' in info:
                        sub_nav = []
                        process_module_dict(info['children'], sub_nav)
                        if sub_nav:
                            nav_list.append({name: sub_nav})
                    else:
                        f.write(f"### {name.split('.')[-1]}\n\n")
                        if info.get('doc'):
                            f.write(f"```python\n{info['doc']}\n```\n\n")
                        nav_list.append({name: f"api/{module_path}.md"})
        
        # 处理所有模块并生成导航结构
        api_nav = []
        process_module_dict(self.module_dict, nav_list=api_nav)
        mkdocs_config['nav'][1]['API Reference'] = api_nav
        
        # 保存 mkdocs.yml 配置文件
        with open(os.path.join(docs_dir, "mkdocs.yml"), "w", encoding="utf-8") as f:
            yaml.dump(mkdocs_config, f, allow_unicode=True, sort_keys=False)
            
        # 启动 mkdocs 服务器
        current_dir = os.getcwd()
        try:
            import subprocess
            import atexit
            os.chdir(self.project_path)
            print("✅ 文档生成已完成，正在启动 mkdocs 服务")
            mkdocs_process = subprocess.Popen(
                ["mkdocs", "serve", "-f", os.path.join(docs_dir, "mkdocs.yml"), "-a", "0.0.0.0:8333"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # 注册清理函数
            def cleanup():
                mkdocs_process.terminate()
                mkdocs_process.wait()
            
            atexit.register(cleanup)
            print("mkdocs 服务器已启动，请访问 http://localhost:8333")
        except Exception as e:
            print(f"启动 mkdocs 服务器时出错: {str(e)}")
        finally:
            os.chdir(current_dir)

    def _generate_project_tree(self) -> str:
        """生成项目目录树的Markdown表示"""
        tree = []
        for root, dirs, files in os.walk(self.project_path):
            level = root.replace(self.project_path, '').count(os.sep)
            indent = '  ' * level
            tree.append(f"{indent}- {os.path.basename(root)}/")
            for file in files:
                if not file.startswith('.'):
                    tree.append(f"{indent}  - {file}")
        return '\n'.join(tree)
      
    def _generate_readme(self) -> None:
        """生成 README.md 文件"""
        print("😊 正在生成README.md...")
        readme_path = os.path.join(self.project_path, "README.md")
        if os.path.exists(readme_path):
            print("✅  README.md 已存在，跳过生成")
            return

        # 分析项目结构
        project_structure = self._analyze_project_structure()
        # 使用 LLM 生成 README 内容
        prompt = README_PROMPT.format(project_structure=json.dumps(project_structure, indent=2, ensure_ascii=False))
        readme_content = self.llm(prompt)
        
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        print("✅ README.md 生成已完成")

    def _generate_gitignore(self) -> None:
        """生成 .gitignore 文件"""
        print("😊 正在生成.gitignore...")
        gitignore_path = os.path.join(self.project_path, ".gitignore")
        if os.path.exists(gitignore_path):
            print("✅ .gitignore 已存在，跳过生成")
            return

        project_info = self._analyze_project_type()

        prompt = GITIGNORE_PROMPT.format(project_info=json.dumps(project_info, indent=2, ensure_ascii=False))
        gitignore_content = self.llm(prompt)
        
        with open(gitignore_path, "w", encoding="utf-8") as f:
            f.write(gitignore_content)
        print("✅ .gitignore 生成已完成")

    def _generate_license(self) -> None:
        """生成 LICENSE 文件"""
        license_path = os.path.join(self.project_path, "LICENSE")
        if os.path.exists(license_path):
            print("LICENSE 已存在，跳过生成")
            return

        # 使用 LLM 选择合适的许可证
        license_content = self.llm(LICENSE_PROMPT)
        
        with open(license_path, "w", encoding="utf-8") as f:
            f.write(license_content)

    def _generate_requirements(self) -> None:
        """生成 requirements.txt 文件"""
        
        req_path = os.path.join(self.project_path, "requirements.txt")
        if os.path.exists(req_path):
            print("✅ requirements.txt 已存在，跳过生成")
            return
        print("😊 正在生成requirements.txt...")
        dependencies = self._analyze_dependencies()
        project_modules = self.module_dict.keys()
        for dep in dependencies:
            if any(dep.startswith(mod) for mod in project_modules):
                dependencies.remove(dep)
        with open(req_path, "w", encoding="utf-8") as f:
            f.write("\n".join(dependencies))
        print("✅ requirements.txt 生成已完成")

    def _analyze_project_structure(self) -> Dict:
        """
        分析项目结构，基于module_dict生成项目的模块组织结构
        """
        def pro_dict(d):
            for name, module in d.items():
                if 'obj' in module:
                    del module['obj']
                if 'children' in module:
                    pro_dict(module['children'])
        pro_dict(self.module_dict)
        return self.module_dict
    
    def _analyze_dependencies(self) -> List[str]:
        """分析项目依赖"""
        dependencies = set()
        for root, _, files in os.walk(self.project_path):
            for file in files:
                if file.endswith(".py"):
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        content = f.read()
                        # 简单的导入语句分析
                        for line in content.split("\n"):
                            if line.startswith(("import ", "from ")):
                                module = line.split()[1].split(".")[0]
                                if not self._is_standard_library(module):
                                    dependencies.add(module)
        return list(dependencies)

    def _analyze_project_type(self) -> Dict:
        """分析项目类型"""
        file_extensions = set()
        for root, _, files in os.walk(self.project_path):
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext:
                    file_extensions.add(ext[1:])  # 移除点号
        
        return {
            "languages": list(file_extensions),
            "has_setup_py": os.path.exists(os.path.join(self.project_path, "setup.py")),
            "has_requirements": os.path.exists(os.path.join(self.project_path, "requirements.txt")),
            "has_tests": any(d.startswith("test") for d in os.listdir(self.project_path))
        }

    @staticmethod
    def _is_standard_library(module_name: str) -> bool:
        """检查是否为 Python 标准库"""
        return module_name in sys.stdlib_module_names

def main():
    if len(sys.argv) != 2:
        print("使用方法: python git_agent.py <project_path>")
        sys.exit(1)
    
    project_path = sys.argv[1]
    agent = GitAgent(project_path)
    agent.standardize_project()

if __name__ == "__main__":
    main()