from dataclasses import dataclass
from pathlib import Path

from src.personalisation.standardized_generic import generate_resumable, read_jsonl


@dataclass
class Candidate:
    text: str
    rank: int

    def to_dict(self):
        return {"text": self.text, "rank": self.rank, "log_probability": -float(self.rank)}


@dataclass
class Result:
    candidates: tuple[Candidate, ...]
    runtime_device: str = "cuda"


class Backend:
    def __init__(self):
        self.calls = []

    def truncate_context_for_generation(self, context, segments):
        return context, len(context), len(context), False

    def _prompt(self, context, segments):
        return [0] * (len(context) + len(segments)), []

    def generate_batch(self, requests, *, top_k, beam_size):
        shapes = {(len(context) + len(segments), len(segments)) for context, segments in requests}
        assert len(shapes) == 1
        self.calls.append(tuple(requests))
        return tuple(Result((Candidate(context, 1),)) for context, _ in requests)


def row(row_id, context, segments=("ni",)):
    return {"row_id": row_id, "author": "a", "work_id": "w", "context": context,
            "pinyin_segments": list(segments), "gold": context, "source_split": "history"}


def run(rows, backend, output):
    return generate_resumable(
        rows, backend, output, batch_size=2, checkpoint_revision="c", official_code_revision="o",
        backend_source_revision="s", backend_integration_revision="i", context_semantics="long",
    )


def test_mixed_shapes_are_bucketed_and_output_order_is_restored(tmp_path: Path):
    rows = [row("r1", "aa"), row("r2", "b"), row("r3", "cc"), row("r4", "d", ("ni", "hao"))]
    backend = Backend()
    result = run(rows, backend, tmp_path / "generic.jsonl")
    assert result["generated"] == 4
    assert [value["row_id"] for value in read_jsonl(tmp_path / "generic.jsonl")] == ["r1", "r2", "r3", "r4"]
    assert any(len(call) == 2 for call in backend.calls)


def test_complete_cache_resumes_without_backend_calls(tmp_path: Path):
    rows = [row("r1", "aa"), row("r2", "b")]
    first = Backend()
    run(rows, first, tmp_path / "generic.jsonl")
    second = Backend()
    result = run(rows, second, tmp_path / "generic.jsonl")
    assert result["reused"] == 2
    assert result["generated"] == 0
    assert second.calls == []


def test_test_dependency_is_rejected(tmp_path: Path):
    rows = [row("r1", "aa")]
    rows[0]["source_split"] = "test"
    try:
        run(rows, Backend(), tmp_path / "generic.jsonl")
    except RuntimeError as error:
        assert "Test" in str(error)
    else:
        raise AssertionError("Test input was accepted")
