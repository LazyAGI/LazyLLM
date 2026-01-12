import pytest

failed_test_classes = set()

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if 'TestDocListServer' in item.nodeid:
        if report.failed:
            failed_test_classes.add('TestDocListServer')


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    if failed_test_classes:
        last_failed = session.config.cache.get('cache/lastfailed', {})

        for item in session.items:
            if 'TestDocListServer' in item.nodeid:
                last_failed[item.nodeid] = True

        session.config.cache.set('cache/lastfailed', last_failed)
