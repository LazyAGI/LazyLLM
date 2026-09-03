import lazyllm
from lazyllm.tools.tool_config_inject import (
    TOOL_AUTH_REGISTRY,
    inject_tool_config,
    register_tool_auth,
)


_MAIL_PROVIDER_NAMES = {
    'gmailimap', 'qqmail', 'qqexmail', 'netease163', 'neteaseqiye',
}


def test_builtin_registry_has_no_mail_provider_names():
    assert _MAIL_PROVIDER_NAMES.isdisjoint(TOOL_AUTH_REGISTRY)


def test_register_tool_auth_maps_external_tool():
    lazyllm.globals._init_sid(sid='test-register-tool-auth')
    previous = TOOL_AUTH_REGISTRY.get('custom_auth_tool')
    try:
        register_tool_auth('custom_auth_tool', 'dynamic_tool_auth')
        assert TOOL_AUTH_REGISTRY['custom_auth_tool'] == 'dynamic_tool_auth'
        inject_tool_config({'custom_auth_tool': 'tok-1'})
        assert lazyllm.globals.config['dynamic_tool_auth']['custom_auth_tool'] == 'tok-1'
    finally:
        if previous is None:
            TOOL_AUTH_REGISTRY.pop('custom_auth_tool', None)
        else:
            TOOL_AUTH_REGISTRY['custom_auth_tool'] = previous
        lazyllm.globals.clear()


def test_register_tool_auth_rejects_unknown_bucket():
    try:
        register_tool_auth('custom_auth_tool', 'not_a_bucket')
        raise AssertionError('expected ValueError')
    except ValueError as orig:
        assert 'config_key' in str(orig)
