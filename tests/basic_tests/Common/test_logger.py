import json
import os
import zipfile

import pytest

from lazyllm.common.logger.logger import _Log


class TestLogger(object):

    @pytest.fixture()
    def log_dir(self, tmp_path):
        def record(timestamp, message, level_no=40):
            return json.dumps({
                'record': {
                    'time': {'timestamp': timestamp},
                    'level': {'no': level_no},
                    'message': message,
                }
            }, ensure_ascii=False)

        with open(os.path.join(str(tmp_path), 'a.json.log'), 'w') as f:
            f.write(record(1000, 'from-plain-log') + '\n')
            f.write(record(2000, 'from-plain-log-2') + '\n')
        with zipfile.ZipFile(os.path.join(str(tmp_path), 'a.json.log.zip'), 'w') as z:
            z.writestr('a.json.log', record(3000, 'from-zip-log') + '\n')
        return str(tmp_path)

    def test_read_aggregates_plain_and_zipped_logs(self, log_dir):
        log = _Log()
        log._log_dir_path = log_dir
        records = log.read(limit=10)
        messages = [r['message'] for r in records]
        assert messages == ['from-plain-log', 'from-plain-log-2', 'from-zip-log']

    def test_read_limit_returns_most_recent(self, log_dir):
        log = _Log()
        log._log_dir_path = log_dir
        messages = [r['message'] for r in log.read(limit=1)]
        assert messages == ['from-zip-log']
