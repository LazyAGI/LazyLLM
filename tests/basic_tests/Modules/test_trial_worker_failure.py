import subprocess
import sys

import pytest


@pytest.mark.parametrize('fail', [True, False])
def test_real_trial_evaluation_finishes_or_reports_worker_failure(fail):
    code = '''import lazyllm
def evaluate(value):
    if FAIL:
        raise ValueError('native trial evaluation failure')
    return value + 1
module = lazyllm.ActionModule(evaluate)
module.trial_choice = lazyllm.Option(1, 2)
module.evalset([1])
try:
    lazyllm.TrialModule(module).update()
except RuntimeError as error:
    assert FAIL and 'Trial workers failed with exit codes' in str(error)
else:
    assert not FAIL
print('trial finished', flush=True)
'''.replace('FAIL', repr(fail))
    child = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = child.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        child.terminate()
        child.communicate(timeout=10)
        pytest.fail('Trial did not finish; owned parent terminated and waited')
    assert child.returncode == 0, stdout + stderr
    assert 'trial finished' in stdout
