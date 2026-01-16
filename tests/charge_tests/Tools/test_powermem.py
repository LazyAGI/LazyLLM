import time
import pytest


@pytest.mark.skip_on_win
@pytest.mark.skip_on_mac
class TestPowerMem(object):
    def test_powermem(self):
        from lazyllm.tools.memory import Memory

        m = Memory(source='powermem')

        user_id = 'test_powermem_user'
        content = 'My order #5678 was for a "Super GPU", but it arrived damaged. It was a gift for my brother.'

        m.add(content, user_id=user_id)

        time.sleep(5)

        res_query = m.get(query='What arrived damaged?', user_id=user_id)

        assert 'Super GPU' in str(res_query)
        assert 'damaged' in str(res_query)

        res_all = m.get(user_id=user_id)

        assert 'Super GPU' in str(res_all)
