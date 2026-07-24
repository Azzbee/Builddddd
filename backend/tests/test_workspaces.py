from __future__ import annotations

import pytest
from fastapi import HTTPException
from lattice.api import deps
from lattice.api.deps import build_container, get_container, set_container
from lattice.config import get_settings


@pytest.fixture(autouse=True)
def _clean_registry():
    set_container(None)
    deps._registry.clear()
    yield
    set_container(None)
    deps._registry.clear()


def test_build_container_overrides_workspace() -> None:
    c = build_container(workspace_id="proj-x")
    assert c.settings.workspace_id == "proj-x"
    assert c.ingestion.settings.workspace_id == "proj-x"


def test_distinct_workspaces_get_isolated_containers() -> None:
    a = get_container("alpha")
    b = get_container("beta")
    assert a is not b
    assert a.settings.workspace_id == "alpha"
    assert b.settings.workspace_id == "beta"
    # Cached per workspace.
    assert get_container("alpha") is a


def test_override_takes_precedence() -> None:
    sentinel = build_container(workspace_id="sentinel")
    set_container(sentinel)
    assert get_container("anything") is sentinel
    set_container(None)
    # After clearing the override, the registry is used again.
    assert get_container("anything").settings.workspace_id == "anything"


def test_direct_call_without_header_uses_default() -> None:
    # Calling outside FastAPI (no Header injection) must not blow up.
    c = get_container()
    assert isinstance(c.settings.workspace_id, str)


def test_header_workspace_is_validated() -> None:
    with pytest.raises(HTTPException) as exc:
        get_container("../other-tenant")
    assert exc.value.status_code == 400


def test_workspace_registry_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LATTICE_MAX_WORKSPACES", "1")
    get_settings.cache_clear()
    try:
        get_container("first")
        with pytest.raises(HTTPException) as exc:
            get_container("second")
        assert exc.value.status_code == 429
    finally:
        get_settings.cache_clear()
