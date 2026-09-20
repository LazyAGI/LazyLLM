from lazyllm.tools.agent.skill_manager import SkillManager


def test_prompt_catalog_does_not_limit_loading(tmp_path):
    for name in ('resident', 'on-demand'):
        folder = tmp_path / name
        folder.mkdir()
        (folder / 'SKILL.md').write_text(
            f'---\nname: {name}\ndescription: {name} workflow\ntags: [private-search-tag]\n---\nFull {name} body.\n'
        )
    manager = SkillManager(dir=str(tmp_path), sandbox=object())
    keys = manager._visible_skill_keys()
    resident = next(key for key in keys if key.endswith('resident'))
    on_demand = next(key for key in keys if key.endswith('on-demand'))
    manager.set_prompt_skills([resident])
    prompt = manager.build_prompt()
    assert f'- {resident}: resident workflow' in prompt
    assert 'on-demand workflow' not in prompt
    assert 'private-search-tag' not in prompt
    assert 'source:' not in prompt
    assert 'Full on-demand body.' in manager.get_skill(on_demand)['content']
    assert 'on-demand workflow' not in ''.join(part['content'] for part in manager.describe_prompt())
    manager.set_prompt_skills([])
    assert 'resident workflow' not in manager.build_prompt()
    assert 'Full on-demand body.' in manager.get_skill(on_demand)['content']


def test_default_prompt_catalog_preserves_existing_behavior(tmp_path):
    folder = tmp_path / 'demo'
    folder.mkdir()
    (folder / 'SKILL.md').write_text('---\nname: demo\ndescription: Demo workflow\n---\nDo it.\n')
    manager = SkillManager(dir=str(tmp_path), sandbox=object())
    assert 'Demo workflow' in manager.build_prompt()
    assert 'source:' in manager.build_prompt()
