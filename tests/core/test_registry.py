import pytest
from pfnstudio_core.registry import (
    _clear_for_tests,
    get_block,
    get_prior,
    list_blocks,
    list_priors,
    register_block,
    register_prior,
)


def setup_function():
    _clear_for_tests()


def test_register_and_get_prior():
    @register_prior("foo")
    class FooPrior:
        pass

    assert get_prior("foo") is FooPrior
    assert "foo" in list_priors()


def test_register_and_get_block():
    @register_block("widget")
    class Widget:
        def __init__(self, n: int = 1):
            self.n = n

    cls = get_block("widget")
    assert cls(n=4).n == 4
    assert "widget" in list_blocks()


def test_duplicate_registration_raises():
    @register_prior("dup")
    class A:
        pass

    with pytest.raises(ValueError):

        @register_prior("dup")
        class B:
            pass


def test_unknown_lookup_raises():
    with pytest.raises(KeyError):
        get_prior("nope")


def test_same_adapter_file_imported_twice_reregisters(tmp_path):
    # The local daemon imports one adapter bundle once per model that shares it
    # (t0-alpha/-beta/-gamma); the second import must not be a clash.
    from pfnstudio_core.registry import discover_in_project, get_adapter

    (tmp_path / "adapters").mkdir()
    (tmp_path / "adapters" / "t0_adapter.py").write_text(
        "from pfnstudio_core.registry import register_adapter\n"
        "@register_adapter('t0')\n"
        "class T0Adapter:\n"
        "    pass\n"
    )
    discover_in_project(tmp_path)
    first = get_adapter("t0")
    discover_in_project(tmp_path)
    second = get_adapter("t0")
    assert second.__qualname__ == "T0Adapter" and second is not first


def test_other_file_under_a_taken_adapter_name_still_raises(tmp_path):
    from pfnstudio_core.registry import discover_in_project, register_adapter

    @register_adapter("t0")
    class Elsewhere:
        pass

    (tmp_path / "adapters").mkdir()
    (tmp_path / "adapters" / "t0_adapter.py").write_text(
        "from pfnstudio_core.registry import register_adapter\n"
        "@register_adapter('t0')\n"
        "class T0Adapter:\n"
        "    pass\n"
    )
    with pytest.raises(ValueError):
        discover_in_project(tmp_path)
