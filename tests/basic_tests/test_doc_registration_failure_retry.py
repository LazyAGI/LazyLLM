from types import ModuleType
import uuid

import pytest

import lazyllm
from lazyllm.docs.utils import add_doc, DuplicateDocError


def module_with_target():
    module = ModuleType('doc_retry_' + uuid.uuid4().hex)

    def target():
        pass

    module.target = target
    return module


@pytest.mark.parametrize('language', ['ENGLISH', 'CHINESE'])
def test_missing_target_can_be_registered_after_it_becomes_available(language):
    module = ModuleType('doc_retry_' + uuid.uuid4().hex)
    with lazyllm.config.temp('language', language):
        with pytest.raises(AttributeError):
            add_doc('target', 'initial', module)
        module.target = lambda: None
        add_doc('target', 'successful retry', module)
        assert module.target.__doc__ == 'successful retry'
        with pytest.raises(DuplicateDocError):
            add_doc('target', 'duplicate', module)


@pytest.mark.parametrize('raise_error', [False, True])
def test_failed_append_can_be_retried_without_losing_base_doc(raise_error):
    module = module_with_target()
    add_doc('target', 'base doc', module)
    with lazyllm.config.temp('raise_on_add_doc_error', raise_error):
        if raise_error:
            with pytest.raises(RuntimeError, match='Failed to add doc'):
                add_doc('target', ['one', 'two'], module, append=' examples ')
        else:
            add_doc('target', ['one', 'two'], module, append=' examples ')
        assert module.target.__doc__ == 'base doc'
        add_doc('target', 'corrected', module, append=' examples ')
    assert module.target.__doc__ == 'base doc examples corrected'
    with pytest.raises(DuplicateDocError):
        add_doc('target', 'duplicate', module, append=' examples ')


@pytest.mark.parametrize('raise_error', [False, True])
def test_read_only_python_target_does_not_reserve_failed_key(raise_error):
    class ConditionalDoc(type):
        def __setattr__(cls, name, value):
            if name == '__doc__' and cls.blocked:
                raise ValueError('doc assignment disabled')
            super().__setattr__(name, value)

    class Target(metaclass=ConditionalDoc):
        blocked = True

    module = ModuleType('doc_retry_' + uuid.uuid4().hex)
    module.Target = Target
    with lazyllm.config.temp('raise_on_add_doc_error', raise_error):
        if raise_error:
            with pytest.raises(RuntimeError, match='Failed to add doc'):
                add_doc('Target', 'initial', module)
        else:
            add_doc('Target', 'initial', module)
        Target.blocked = False
        add_doc('Target', 'successful retry', module)
    assert Target.__doc__ == 'successful retry'


def test_successful_language_and_append_sections_remain_independent():
    module = module_with_target()
    with lazyllm.config.temp('language', 'ENGLISH'):
        add_doc('target', 'English', module)
        add_doc('target', 'example', module, append=' examples ')
    with lazyllm.config.temp('language', 'CHINESE'):
        add_doc('target', 'Chinese', module)
    assert module.target.__doc__ == 'Chinese'
    with lazyllm.config.temp('language', 'ENGLISH'), pytest.raises(DuplicateDocError):
        add_doc('target', 'duplicate', module)
