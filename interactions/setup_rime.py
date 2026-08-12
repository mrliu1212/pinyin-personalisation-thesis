"""Install pinned official Rime schema data and deploy the Luna Pinyin schema."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


PINNED_LIBRIME_VERSION = "1.17.0"


def run(
    command: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def resolve_rime_deployer(
    explicit: Path | None = None,
    rime_prefix: Path | None = None,
) -> Path:
    if explicit is not None:
        candidates = (explicit,)
    elif rime_prefix is not None:
        candidates = (
            rime_prefix / "bin" / "rime_deployer.exe",
            rime_prefix / "bin" / "rime_deployer",
        )
    else:
        discovered = shutil.which("rime_deployer")
        candidates = (Path(discovered),) if discovered else ()
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "rime_deployer was not found; pass --rime-deployer or --rime-prefix"
    )


def resolve_librime_version(explicit: str | None = None) -> str:
    if explicit:
        version = explicit
    else:
        brew = shutil.which("brew")
        if not brew:
            raise RuntimeError(
                "librime version cannot be inferred without Homebrew; pass --librime-version"
            )
        version = subprocess.check_output(
            [brew, "list", "--versions", "librime"], text=True
        ).strip()
    if PINNED_LIBRIME_VERSION not in version:
        raise ValueError(
            f"Phase 4F.1 requires librime {PINNED_LIBRIME_VERSION}; got {version!r}"
        )
    return version


def install_opencc_data(rime_prefix: Path | None, shared_dir: Path) -> None:
    if rime_prefix is None:
        return
    candidates = (
        rime_prefix / "share" / "opencc",
        rime_prefix.parent / "share" / "opencc",
    )
    source = next((path for path in candidates if (path / "t2s.json").is_file()), None)
    if source is None:
        checked = ", ".join(str(path / "t2s.json") for path in candidates)
        raise FileNotFoundError(f"pinned OpenCC configuration not found; checked: {checked}")
    shutil.copytree(source, shared_dir / "opencc", dirs_exist_ok=True)


def rime_runtime_environment(rime_prefix: Path | None) -> dict[str, str] | None:
    if os.name != "nt" or rime_prefix is None:
        return None
    env = os.environ.copy()
    runtime_paths = (rime_prefix / "bin", rime_prefix / "lib")
    env["PATH"] = os.pathsep.join(
        [*(str(path.resolve()) for path in runtime_paths), env.get("PATH", "")]
    )
    return env


def setup_rime(
    lock_path: Path,
    output_dir: Path,
    custom_config: Path,
    *,
    rime_deployer: Path | None = None,
    rime_prefix: Path | None = None,
    librime_version: str | None = None,
) -> dict[str, object]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    deployer = resolve_rime_deployer(rime_deployer, rime_prefix)
    version = resolve_librime_version(librime_version)
    shared_dir = output_dir / "shared"
    user_dir = output_dir / "user"
    build_dir = output_dir / "build"
    shared_dir.mkdir(parents=True, exist_ok=True)
    user_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    install_opencc_data(rime_prefix, shared_dir)

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
            str(deployer),
            "--build",
            str(user_dir.resolve()),
            str(shared_dir.resolve()),
            str(build_dir.resolve()),
        ],
        env=rime_runtime_environment(rime_prefix),
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_id": lock["schema_id"],
        "librime": version,
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
    parser.add_argument(
        "--rime-prefix",
        type=Path,
        default=Path(os.environ["RIME_PREFIX"]) if "RIME_PREFIX" in os.environ else None,
    )
    parser.add_argument("--rime-deployer", type=Path)
    parser.add_argument("--librime-version")
    args = parser.parse_args()
    manifest = setup_rime(
        args.lock,
        args.output_dir,
        args.custom_config,
        rime_deployer=args.rime_deployer,
        rime_prefix=args.rime_prefix,
        librime_version=args.librime_version,
    )
    print(f"Deployed {manifest['schema_id']} using {manifest['librime']}.")
    print(f"Rime data: {args.output_dir}")


if __name__ == "__main__":
    main()

