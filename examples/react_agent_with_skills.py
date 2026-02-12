import os
import tempfile

import lazyllm
from lazyllm.tools import ReactAgent
from lazyllm.tools.agent.skill_manager import SkillManager
from lazyllm.cli.skills import skills as skills_cli


def _prepare_sample_skill(root: str) -> str:
    skill_dir = os.path.join(root, 'demo-skill')
    os.makedirs(skill_dir, exist_ok=True)
    skill_md = os.path.join(skill_dir, 'SKILL.md')
    with open(skill_md, 'w', encoding='utf-8') as f:
        f.write(
            "---\n"
            "name: demo-skill\n"
            "description: A demo skill for showing how to import and use skills.\n"
            "---\n"
            "\n"
            "# demo-skill\n"
            "This is a minimal demo skill.\n"
        )
    return skill_dir


def main():
    # 1) Prepare a local skill folder and import it into the default skills directory
    with tempfile.TemporaryDirectory() as tmp:
        skill_dir = _prepare_sample_skill(tmp)
        # CLI equivalents:
        #   lazyllm skills init
        #   lazyllm skills add /path/to/skill_folder
        skills_cli(['init'])
        skills_cli(['add', skill_dir])

    # 2) Use SkillManager to show available skills
    manager = SkillManager()
    lazyllm.LOG.info(manager.list_skill())

    # 3) Run ReactAgent with skills enabled
    llm = lazyllm.TrainableModule('Qwen2.5-32B-Instruct')
    agent = ReactAgent(
        llm=llm,
        tools=[],
        skills=['demo-skill'],
    )
    res = agent('What skills do you have?')
    lazyllm.LOG.info(res)


if __name__ == '__main__':
    main()
