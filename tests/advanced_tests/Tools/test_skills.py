import os
import shutil

import lazyllm
from lazyllm.tools import ReactAgent
from lazyllm.cli.skills import skills as skills_cli
from lazyllm.tools.agent.skill_manager import SkillManager


def _make_skill(base_dir: str, folder_name: str, meta_name: str) -> str:
    skill_dir = os.path.join(base_dir, folder_name)
    os.makedirs(skill_dir, exist_ok=True)
    skill_md = os.path.join(skill_dir, 'SKILL.md')
    with open(skill_md, 'w', encoding='utf-8') as f:
        f.write(
            '---\n'
            f'name: {meta_name}\n'
            f'description: {meta_name} skill for tests\n'
            '---\n'
            f'# {meta_name}\n'
            'Test skill\n'
        )
    return skill_dir


class TestSkills(object):
    @classmethod
    def setup_class(cls):
        cls._home = lazyllm.config['home']
        cls._skills_dir = lazyllm.config['skills_dir']
        cls._src_root = os.path.join(cls._home, '_test_skills_src')
        os.makedirs(cls._src_root, exist_ok=True)
        cls._alpha_folder = 'test-alpha'
        cls._beta_folder = 'test-beta'
        cls._alpha_name = 'test-alpha'
        cls._beta_name = 'test-beta'
        _make_skill(cls._src_root, cls._alpha_folder, cls._alpha_name)
        _make_skill(cls._src_root, cls._beta_folder, cls._beta_name)

    @classmethod
    def teardown_class(cls):
        for folder in (cls._alpha_folder, cls._beta_folder):
            path = os.path.join(cls._skills_dir, folder)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
        if os.path.isdir(cls._src_root):
            shutil.rmtree(cls._src_root, ignore_errors=True)

    def test_skills_cli(self):
        skills_cli(['init'])
        assert os.path.isdir(self._skills_dir)

        skills_cli(['import', self._src_root])
        assert os.path.isdir(os.path.join(self._skills_dir, self._alpha_folder))
        assert os.path.isdir(os.path.join(self._skills_dir, self._beta_folder))

    def test_skill_manager(self):
        manager = SkillManager(dir=self._skills_dir)
        listing = manager.list_skill()
        assert self._alpha_name in listing
        assert self._beta_name in listing

    def test_react_agent_with_skills(self):
        llm = lazyllm.TrainableModule('Qwen2.5-32B-Instruct')
        agent = ReactAgent(llm=llm, skills=[self._alpha_name, self._beta_name])
        res = agent('what skills do you have?')
        assert self._alpha_name in res
        assert self._beta_name in res
