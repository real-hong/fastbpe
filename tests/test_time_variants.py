"""scripts/time_variants.py rejects variants whose methods master never calls."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from fastbpe import Tokenizer

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def tv() -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))  # for its `import adversarial_inputs`
    try:
        spec = importlib.util.spec_from_file_location("time_variants", SCRIPTS / "time_variants.py")
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPTS))
    return module


def test_the_scripts_variants_pass(tv: ModuleType) -> None:
    assert tv.unknown_overrides(tv.VARIANTS) == []


def test_a_method_tokenizer_lacks_is_reported(tv: ModuleType) -> None:
    class Stale(tv.MasterCopyTokenizer):  # type: ignore[name-defined, misc]
        def _encode_piece(self, piece: str) -> list[int]:  # what master used to call
            return []

    assert tv.unknown_overrides({"stale": Stale}) == ["stale: Stale._encode_piece"]


def test_methods_a_base_declares_in_adds_are_allowed(tv: ModuleType) -> None:
    class Base(Tokenizer):
        ADDS = frozenset({"_step"})

        def _step(self) -> None:
            pass

    class Child(Base):
        def _step(self) -> None:
            pass

        def _other(self) -> None:
            pass

    assert tv.unknown_overrides({"child": Child}) == ["child: Child._other"]


def test_main_exits_on_an_unknown_override(tv: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    class Stale(Tokenizer):
        def _gone(self) -> None:
            pass

    monkeypatch.setattr(tv, "VARIANTS", {"stale": Stale})
    # --reps 0: if the check were missing, nothing would be timed.
    argv = ["time_variants.py", "--reps", "0", "--variants", "stale"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match=r"stale: Stale\._gone"):
        tv.main()
