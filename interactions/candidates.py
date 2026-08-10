"""Candidate-generator boundary and persistent Rime CLI adapter."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Candidate:
    text: str
    base_rank: int
    base_score: None = None


class CandidateGenerator(Protocol):
    name: str
    version: str
    schema_id: str
    max_candidates: int

    def candidates(self, pinyin_input: str) -> list[Candidate]: ...


class RimeCliCandidateGenerator:
    name = "librime"

    def __init__(
        self,
        executable: Path,
        shared_data: Path,
        prebuilt_data: Path,
        *,
        version: str,
        schema_id: str = "luna_pinyin",
        max_candidates: int = 10,
    ) -> None:
        self.version = version
        self.schema_id = schema_id
        self.max_candidates = max_candidates
        self._temporary_user = tempfile.TemporaryDirectory(prefix="phase4b_rime_user_")
        self._process = subprocess.Popen(
            [
                str(executable),
                "--shared-data",
                str(shared_data),
                "--user-data",
                self._temporary_user.name,
                "--prebuilt-data",
                str(prebuilt_data),
                "--schema",
                schema_id,
                "--max-candidates",
                str(max_candidates),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._cache: dict[str, list[Candidate]] = {}

    def candidates(self, pinyin_input: str) -> list[Candidate]:
        if pinyin_input in self._cache:
            return list(self._cache[pinyin_input])
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("Rime adapter pipes are unavailable")
        self._process.stdin.write(pinyin_input + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if line == "" and self._process.poll() is not None:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise RuntimeError(f"Rime adapter exited: {stderr.strip()}")
        texts = [item for item in line.rstrip("\n").split("\t") if item]
        candidates = [Candidate(text, index) for index, text in enumerate(texts, 1)]
        self._cache[pinyin_input] = candidates
        return list(candidates)

    def close(self) -> None:
        if self._process.stdin:
            self._process.stdin.close()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=5)
        self._temporary_user.cleanup()

    def __enter__(self) -> "RimeCliCandidateGenerator":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

