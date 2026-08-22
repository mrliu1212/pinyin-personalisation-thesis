from pathlib import Path

from experiments.external_memory_next.finalize_train_fit_generic_v1 import parse_final_json


def test_parse_final_json_after_progress_lines(tmp_path: Path) -> None:
    path = tmp_path / "stdout.log"
    path.write_text('generic 1/2\ngeneric 2/2\n{"status":"complete","rows":2}\n', encoding="utf-8")
    assert parse_final_json(path) == {"status": "complete", "rows": 2}
