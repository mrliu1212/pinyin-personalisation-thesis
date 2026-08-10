"""Install pinned official Rime schema data and deploy the Luna Pinyin schema."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def setup_rime(lock_path: Path, output_dir: Path, custom_config: Path) -> dict[str, object]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    shared_dir = output_dir / "shared"
    user_dir = output_dir / "user"
    build_dir = output_dir / "build"
    shared_dir.mkdir(parents=True, exist_ok=True)
    user_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="phase4b_rime_sources_") as temporary:
        temporary_root = Path(temporary)
        for source in lock["sources"]:
            repository = temporary_root / source["name"]
            run(["git", "init", "--quiet", str(repository)])
            run(["git", "remote", "add", "origin", source["url"]], cwd=repository)
            run(
                ["git", "fetch", "--quiet", "--depth", "1", "origin", source["commit"]],
                cwd=repository,
            )
            run(["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=repository)
            actual_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repository, text=True
            ).strip()
            if actual_commit != source["commit"]:
                raise RuntimeError(f"commit mismatch for {source['name']}")
            for filename in source["files"]:
                shutil.copy2(repository / filename, shared_dir / filename)

    shutil.copy2(custom_config, user_dir / "default.custom.yaml")
    run(
        [
            "rime_deployer",
            "--build",
            str(user_dir.resolve()),
            str(shared_dir.resolve()),
            str(build_dir.resolve()),
        ]
    )
    librime_version = subprocess.check_output(
        ["brew", "list", "--versions", "librime"], text=True
    ).strip()
    manifest: dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_id": lock["schema_id"],
        "librime": librime_version,
        "source_lock": lock,
        "shared_data_dir": str(shared_dir),
        "user_data_dir": str(user_dir),
        "prebuilt_data_dir": str(build_dir),
    }
    (output_dir / "setup_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock", type=Path, default=Path("config/rime/sources.json")
    )
    parser.add_argument(
        "--custom-config",
        type=Path,
        default=Path("config/rime/default.custom.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/rime"))
    args = parser.parse_args()
    manifest = setup_rime(args.lock, args.output_dir, args.custom_config)
    print(f"Deployed {manifest['schema_id']} using {manifest['librime']}.")
    print(f"Rime data: {args.output_dir}")


if __name__ == "__main__":
    main()

