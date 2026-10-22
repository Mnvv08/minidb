import resource

import pytest

from wal import WALPager


def pytest_configure(config):
    """Crash tests open many files; macOS defaults to only 256 per process, so raise it."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = 4096 if hard == resource.RLIM_INFINITY else min(4096, hard)
    if soft != resource.RLIM_INFINITY and soft < want:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
        except (ValueError, OSError):
            pass


@pytest.fixture(autouse=True)
def fast_wal(monkeypatch):
    """Tests do not need real fsyncs, so switch them off for speed."""
    monkeypatch.setattr(WALPager, "DEFAULT_SYNC", False)
