from pathlib import Path

from experiments.external_memory_next.prepare_train_fit_ranking_features_v1 import (
    frozen_frequency_rows,
    normalized_source_sha256,
)


def test_normalized_source_hash_is_line_ending_independent(tmp_path: Path) -> None:
    lf = tmp_path / "lf.py"
    crlf = tmp_path / "crlf.py"
    lf.write_bytes(b"a\nb\n")
    crlf.write_bytes(b"a\r\nb\r\n")
    assert normalized_source_sha256(lf) == normalized_source_sha256(crlf)


def test_empty_generic_surface_uses_frozen_conservative_noop() -> None:
    class Base:
        @staticmethod
        def frequency_rows(**_: object) -> list[dict[str, object]]:
            raise AssertionError("empty Generic surface must not reach normalization")

    assert frozen_frequency_rows(
        Base, query=object(), generic_candidates=[], history_rows=[]
    ) == []


def test_nonempty_generic_surface_delegates_unchanged() -> None:
    expected = [{"candidate": "北京"}]

    class Base:
        @staticmethod
        def frequency_rows(**kwargs: object) -> list[dict[str, object]]:
            assert kwargs["generic_candidates"] == ["candidate"]
            return expected

    assert frozen_frequency_rows(
        Base, query=object(), generic_candidates=["candidate"], history_rows=[]
    ) is expected
