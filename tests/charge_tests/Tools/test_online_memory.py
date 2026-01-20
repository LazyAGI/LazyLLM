import pytest
import time

@pytest.mark.skip(reason='key not ready')
class TestMemory(object):
    def test_memory(self):
        from lazyllm.tools.memory import Memory
        m = Memory()
        m.add('My order #1234 was for a \'Nova 2000\', but it arrived damaged. It was a gift for my sister.',
              user_id='test12')
        time.sleep(10)
        m.get(user_id='test12')
