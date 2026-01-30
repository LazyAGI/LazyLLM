import argparse
import subprocess
import tempfile
import shutil
from lazyllm import LOG
from pathlib import Path
from dataclasses import dataclass

DEFAULT_SKILL_REPO = 'https://github.com/LazyAGI/LazyLLM.git'
FALLBACK_SKILL_REPO = 'https://gitcode.com/LazyAGI/LazyLLM.git'

SKILL_RELATIVE_PATH = Path('docs/lazyllm-skill')

def install_skill(commands):
    parser = argparse.ArgumentParser(
        prog='lazyllm install --skill',
        description='Install lazyllm-skill for an agent'
    )

    parser.add_argument(
        '--agent',
        required=True,
        help='Target agent (e.g., opencode, cursor, claude..)'
    )

    parser.add_argument(
        '--project',
        action='store_true',
        help='Install at project level (default: user level)'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Timeout in seconds for git clone (default: 300)'
    )

    args = parser.parse_args(commands)
    agent = args.agent
    project_level = args.project
    timeout = args.timeout

    if get_agent_config(agent) is None:
        agents = ', '.join(get_all_agents().keys())
        raise ValueError(f"Unknown agent '{agent}'. Available: {agents}")

    tmp_dir = Path(tempfile.mkdtemp(prefix='lazyllm-skill-'))
    try:
        _download_repo(tmp_dir, timeout=timeout)

        skill_src = tmp_dir / SKILL_RELATIVE_PATH
        if not skill_src.exists():
            raise RuntimeError(
                f'lazyllm-skill not found at {skill_src}'
            )

        target_root = get_path(agent, project_level)
        target_dir = target_root / 'lazyllm-skill'

        target_dir.parent.mkdir(parents=True, exist_ok=True)

        if target_dir.exists():
            shutil.rmtree(target_dir)

        shutil.copytree(skill_src, target_dir)

        LOG.info(f'lazyllm-skill installed to: {target_dir}')

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def get_user_path(agent: str) -> Path:
    config = get_agent_config(agent)
    if config is None:
        raise ValueError(f'Unknown agent: {agent}')

    if not config.supports_home or config.home_dir is None:
        raise ValueError(f"Agent '{agent}' does not support user-level installation")

    return config.home_dir

def get_project_path(agent: str, project_dir: Path | None = None) -> Path:
    config = get_agent_config(agent)
    if config is None:
        raise ValueError(f'Unknown agent: {agent}')

    if project_dir is None:
        project_dir = Path.cwd()

    return (project_dir / config.project_dir).resolve()

def get_path(agent: str, project_level: bool = False, project_dir: Path | None = None) -> Path:
    if project_level:
        return get_project_path(agent, project_dir)
    else:
        return get_user_path(agent)

def list_all_agents() -> dict[str, dict[str, str | None]]:
    return get_all_agents()

def _clone_repo(repo: str, dst: Path, timeout: int = 300):
    cmd = ['git', 'clone', repo, str(dst)]
    subprocess.check_call(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout
    )

def _download_repo(tmp_dir: Path, timeout: int = 300):
    try:
        _clone_repo(DEFAULT_SKILL_REPO, tmp_dir, timeout=timeout)
    except Exception:
        _clone_repo(FALLBACK_SKILL_REPO, tmp_dir, timeout=timeout)


@dataclass
class AgentConfig:
    name: str
    display_name: str
    home_dir: Path | None
    project_dir: Path
    supports_home: bool

# All agent configurations
AGENT_CONFIGS: dict[str, AgentConfig] = {
    'claude': AgentConfig(
        name='claude',
        display_name='Claude Code',
        home_dir=Path.home() / '.claude' / 'skills',
        project_dir=Path('.claude') / 'skills',
        supports_home=True,
    ),
    'opencode': AgentConfig(
        name='opencode',
        display_name='OpenCode CLI',
        home_dir=Path.home() / '.config' / 'opencode' / 'skill',
        project_dir=Path('.opencode') / 'skill',
        supports_home=True,
    ),
    'codex': AgentConfig(
        name='codex',
        display_name='OpenAI Codex',
        home_dir=Path.home() / '.codex' / 'skills',
        project_dir=Path('.codex') / 'skills',
        supports_home=True,
    ),
    'gemini': AgentConfig(
        name='gemini',
        display_name='Gemini CLI',
        home_dir=Path.home() / '.gemini' / 'skills',
        project_dir=Path('.gemini') / 'skills',
        supports_home=True,
    ),
    'copilot': AgentConfig(
        name='copilot',
        display_name='GitHub Copilot',
        home_dir=Path.home() / '.copilot' / 'skills',
        project_dir=Path('.github') / 'skills',
        supports_home=True,
    ),
    'cursor': AgentConfig(
        name='cursor',
        display_name='Cursor',
        home_dir=Path.home() / '.cursor' / 'skills',
        project_dir=Path('.cursor') / 'skills',
        supports_home=True,
    ),
    'windsurf': AgentConfig(
        name='windsurf',
        display_name='Windsurf',
        home_dir=Path.home() / '.codeium' / 'windsurf' / 'skills',
        project_dir=Path('.windsurf') / 'skills',
        supports_home=True,
    ),
    'qwen': AgentConfig(
        name='qwen',
        display_name='Qwen Code',
        home_dir=Path.home() / '.qwen' / 'skills',
        project_dir=Path('.qwen') / 'skills',
        supports_home=True,
    ),
    'antigravity': AgentConfig(
        name='antigravity',
        display_name='Google Antigravity',
        home_dir=Path.home() / '.gemini' / 'antigravity' / 'skills',
        project_dir=Path('.agent') / 'skills',
        supports_home=True,
    ),
    'openhands': AgentConfig(
        name='openhands',
        display_name='OpenHands',
        home_dir=Path.home() / '.openhands' / 'skills',
        project_dir=Path('.openhands') / 'skills',
        supports_home=True,
    ),
    'cline': AgentConfig(
        name='cline',
        display_name='Cline',
        home_dir=Path.home() / '.cline' / 'skills',
        project_dir=Path('.cline') / 'skills',
        supports_home=True,
    ),
    'goose': AgentConfig(
        name='goose',
        display_name='Goose',
        home_dir=Path.home() / '.config' / 'goose' / 'skills',
        project_dir=Path('.goose') / 'skills',
        supports_home=True,
    ),
    'roo': AgentConfig(
        name='roo',
        display_name='Roo Code',
        home_dir=Path.home() / '.roo' / 'skills',
        project_dir=Path('.roo') / 'skills',
        supports_home=True,
    ),
    'kilo': AgentConfig(
        name='kilo',
        display_name='Kilo Code',
        home_dir=Path.home() / '.kilocode' / 'skills',
        project_dir=Path('.kilocode') / 'skills',
        supports_home=True,
    ),
    'trae': AgentConfig(
        name='trae',
        display_name='Trae',
        home_dir=Path.home() / '.trae' / 'skills',
        project_dir=Path('.trae') / 'skills',
        supports_home=True,
    ),
    'droid': AgentConfig(
        name='droid',
        display_name='Droid',
        home_dir=Path.home() / '.factory' / 'skills',
        project_dir=Path('.factory') / 'skills',
        supports_home=True,
    ),
    'clawdbot': AgentConfig(
        name='clawdbot',
        display_name='Clawdbot',
        home_dir=Path.home() / '.clawdbot' / 'skills',
        project_dir=Path('skills'),
        supports_home=True,
    ),
    'kiro-cli': AgentConfig(
        name='kiro-cli',
        display_name='Kiro CLI',
        home_dir=Path.home() / '.kiro' / 'skills',
        project_dir=Path('.kiro') / 'skills',
        supports_home=True,
    ),
    'pi': AgentConfig(
        name='pi',
        display_name='Pi',
        home_dir=Path.home() / '.pi' / 'agent' / 'skills',
        project_dir=Path('.pi') / 'skills',
        supports_home=True,
    ),
    'neovate': AgentConfig(
        name='neovate',
        display_name='Neovate',
        home_dir=Path.home() / '.neovate' / 'skills',
        project_dir=Path('.neovate') / 'skills',
        supports_home=True,
    ),
    'zencoder': AgentConfig(
        name='zencoder',
        display_name='Zencoder',
        home_dir=Path.home() / '.zencoder' / 'skills',
        project_dir=Path('.zencoder') / 'skills',
        supports_home=True,
    ),
    'amp': AgentConfig(
        name='amp',
        display_name='Amp',
        home_dir=Path.home() / '.config' / 'agents' / 'skills',
        project_dir=Path('.agents') / 'skills',
        supports_home=True,
    ),
    'qoder': AgentConfig(
        name='qoder',
        display_name='Qoder',
        home_dir=Path.home() / '.qoder' / 'skills',
        project_dir=Path('.qoder') / 'skills',
        supports_home=True,
    ),
}

def get_agent_config(agent: str) -> AgentConfig | None:
    return AGENT_CONFIGS.get(agent)

def get_all_agents() -> list[str]:
    return list(AGENT_CONFIGS.keys())
