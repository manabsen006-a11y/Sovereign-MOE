"""Session set-up: the instances the data tests read are fetched on first
use (see :mod:`tests._data`), before collection, so a fresh clone runs
them rather than skipping them."""

from tests import _data


def pytest_sessionstart(session):
    _data.ensure()
