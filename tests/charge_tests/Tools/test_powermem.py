import pytest
import time
import os


# Helper function to check if .env exists in the project root
def check_env_exists():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
        env_path = os.path.join(project_root, '.env')
        return os.path.exists(env_path) or os.path.exists(os.path.join(os.getcwd(), '.env'))
    except Exception:
        return False


# Determine if keys are ready based on .env existence
keys_ready = check_env_exists()


@pytest.mark.skipif(not keys_ready, reason='env config not ready')
class TestPowerMem(object):
    def test_powermem(self):
        from lazyllm.tools.memory import Memory

        # Initialize PowerMem
        m = Memory(source='powermem')

        # Prepare test data
        user_id = 'test_powermem_user'
        content = 'My order #5678 was for a "Super GPU", but it arrived damaged. It was a gift for my brother.'

        print(f"\n[Test] Adding memory for user: {user_id}")
        m.add(content, user_id=user_id)

        # Wait for indexing (async processing simulation)
        time.sleep(5)

        # Test retrieval with query
        print("[Test] Searching with query...")
        res_query = m.get(query="What arrived damaged?", user_id=user_id)

        # Verify specific content retrieval
        assert "Super GPU" in str(res_query)
        assert "damaged" in str(res_query)

        # Test retrieval all memories (empty query)
        print("[Test] Getting all memories (empty query)...")
        res_all = m.get(user_id=user_id)

        # Verify the content exists in full history
        assert "Super GPU" in str(res_all)