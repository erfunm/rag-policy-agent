from __future__ import annotations

import pytest

from policy_agent.config import get_settings


@pytest.fixture(autouse=True)
def _fresh_settings() -> None:
    get_settings.cache_clear()
