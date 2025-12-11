import pytest
import time
import os


def check_env_exists():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
        env_path = os.path.join(project_root, '.env')
        return os.path.exists(env_path) or os.path.exists(os.path.join(os.getcwd(), '.env'))
    except Exception:
        return False


keys_ready = check_env_exists()


@pytest.mark.skipif(not keys_ready, reason='env config not ready')
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
