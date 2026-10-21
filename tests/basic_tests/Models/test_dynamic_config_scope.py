from contextlib import nullcontext

import pytest

import lazyllm
from lazyllm.module.llms.onlinemodule.chat import dynamic_chat_config
from lazyllm.module.llms.onlinemodule.embedding import dynamic_embed_config
from lazyllm.module.llms.onlinemodule.multimodal import dynamic_multimodal_config
from lazyllm.module.llms.onlinemodule.dynamic_router import ConfigsDict


@pytest.fixture(params=[
    ('chat', dynamic_chat_config),
    ('embed', dynamic_embed_config),
    ('multimodal', dynamic_multimodal_config),
])
def context(request):
    previous = lazyllm.globals['config'].get('dynamic_model_configs')
    cfg = ConfigsDict()
    lazyllm.globals.config['dynamic_model_configs'] = cfg
    try:
        yield (*request.param, cfg)
    finally:
        lazyllm.globals.config['dynamic_model_configs'] = previous


@pytest.mark.parametrize('raises', [False, True])
def test_new_module_does_not_keep_scoped_config(context, raises):
    slot, manager, cfg = context
    with pytest.raises(RuntimeError, match='body') if raises else nullcontext():
        with manager('fresh', source='qwen', model='scoped'):
            assert cfg['fresh'][slot]['model'] == 'scoped'
            if raises:
                raise RuntimeError('body')
    assert 'fresh' not in cfg


def test_existing_slot_is_restored(context):
    slot, manager, cfg = context
    cfg['existing'] = {slot: {'source': 'qwen', 'model': 'original'}}
    with manager('existing', model='scoped'):
        assert cfg['existing'][slot] == {'model': 'scoped'}
    assert cfg['existing'] == {slot: {'source': 'qwen', 'model': 'original'}}


def test_absent_slot_stays_absent_without_losing_other_slots(context):
    slot, manager, cfg = context
    cfg['existing'] = {'other': {'model': 'keep'}}
    with manager('existing', model='scoped'):
        assert cfg['existing'][slot] == {'model': 'scoped'}
    assert cfg['existing'] == {'other': {'model': 'keep'}}


def test_nested_new_module_is_removed_only_after_outer_exit(context):
    slot, manager, cfg = context
    with manager('fresh', model='outer'):
        with manager('fresh', model='inner'):
            assert cfg['fresh'][slot] == {'model': 'inner'}
        assert cfg['fresh'][slot] == {'model': 'outer'}
    assert 'fresh' not in cfg


def test_other_slot_added_inside_scope_survives(context):
    slot, manager, cfg = context
    with manager('fresh', model='temporary'):
        cfg['fresh']['other'] = {'model': 'keep'}
    assert cfg['fresh'] == {'other': {'model': 'keep'}}


def test_repeated_module_id_cleanup(context):
    slot, manager, cfg = context
    with manager(['fresh', 'fresh'], model='temporary'):
        assert cfg['fresh'][slot] == {'model': 'temporary'}
    assert 'fresh' not in cfg
