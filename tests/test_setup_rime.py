import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from interactions import setup_rime


class SetupRimeWindowsCompatibilityTests(unittest.TestCase):
    def test_explicit_windows_deployer_and_version_avoid_homebrew(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = root / "librime"
            deployer = prefix / "bin" / "rime_deployer.exe"
            deployer.parent.mkdir(parents=True)
            deployer.touch()
            opencc = prefix / "share" / "opencc"
            opencc.mkdir(parents=True)
            (opencc / "t2s.json").write_text("{}", encoding="utf-8")
            lock = root / "sources.json"
            lock.write_text(
                json.dumps({"schema_id": "luna_pinyin", "sources": []}),
                encoding="utf-8",
            )
            custom = root / "default.custom.yaml"
            custom.write_text("patch: {}\n", encoding="utf-8")
            output = root / "data"

            with (
                patch.object(setup_rime, "run") as run,
                patch.object(setup_rime.shutil, "which") as which,
            ):
                manifest = setup_rime.setup_rime(
                    lock,
                    output,
                    custom,
                    rime_prefix=prefix,
                    librime_version="librime 1.17.0",
                )

            which.assert_not_called()
            expected_command = (
                [
                    str(deployer.resolve()),
                    "--build",
                    str((output / "user").resolve()),
                    str((output / "shared").resolve()),
                    str((output / "build").resolve()),
                ],
            )
            self.assertEqual(run.call_args.args, expected_command)
            runtime_env = run.call_args.kwargs["env"]
            if setup_rime.os.name == "nt":
                self.assertIn(str((prefix / "lib").resolve()), runtime_env["PATH"])
            else:
                self.assertIsNone(runtime_env)
            self.assertEqual(manifest["librime"], "librime 1.17.0")
            self.assertTrue((output / "shared" / "opencc" / "t2s.json").is_file())

    def test_non_pinned_librime_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires librime 1.17.0"):
            setup_rime.resolve_librime_version("librime 1.16.1")

    def test_opencc_data_is_found_next_to_dist_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = root / "dist"
            source = root / "share" / "opencc"
            source.mkdir(parents=True)
            (source / "t2s.json").write_text("{}", encoding="utf-8")
            shared = root / "rime-data"
            setup_rime.install_opencc_data(prefix, shared)
            self.assertTrue((shared / "opencc" / "t2s.json").is_file())

    def test_missing_windows_deployer_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "--rime-deployer"):
                setup_rime.resolve_rime_deployer(rime_prefix=Path(directory))


if __name__ == "__main__":
    unittest.main()
