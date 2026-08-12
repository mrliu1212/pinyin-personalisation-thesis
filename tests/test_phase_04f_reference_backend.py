import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from experiments import exp_phase_04f_reference_backend


class Phase04FReferenceBackendTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "Windows-specific executable name")
    def test_windows_rime_adapter_uses_exe_suffix(self):
        self.assertEqual(
            exp_phase_04f_reference_backend.RIME_EXECUTABLE.name,
            "rime_candidate_cli.exe",
        )

    @unittest.skipUnless(sys.platform == "win32", "Windows-specific smoke result name")
    def test_windows_smoke_result_does_not_overwrite_frozen_mac_artifact(self):
        self.assertEqual(
            exp_phase_04f_reference_backend.SMOKE_RESULT.name,
            "smoke_test_windows_cuda.json",
        )

    def test_valid_existing_model_assets_skip_apk_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = {}
            members = ("generation.gguf", "embedding.gguf")
            for member in members:
                content = member.encode()
                destination = root / member
                destination.write_bytes(content)
                assets[member] = (
                    destination,
                    hashlib.sha256(content).hexdigest(),
                    len(content),
                )
            with (
                patch.object(exp_phase_04f_reference_backend, "ASSETS", assets),
                patch.object(
                    exp_phase_04f_reference_backend,
                    "REQUIRED_MODEL_ASSET_MEMBERS",
                    members,
                ),
                patch.object(exp_phase_04f_reference_backend, "_download_apk") as download,
                patch.object(exp_phase_04f_reference_backend.zipfile, "ZipFile") as zip_file,
            ):
                exp_phase_04f_reference_backend._prepare_extracted_assets()
            download.assert_not_called()
            zip_file.assert_not_called()

    def test_missing_or_invalid_model_asset_uses_apk_fallback(self):
        for existing_content in (None, b"invalid"):
            with self.subTest(existing_content=existing_content):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    destination = root / "model.gguf"
                    expected_content = b"valid"
                    if existing_content is not None:
                        destination.write_bytes(existing_content)
                    assets = {
                        "model.gguf": (
                            destination,
                            hashlib.sha256(expected_content).hexdigest(),
                            len(expected_content),
                        )
                    }
                    archive = Mock()
                    archive.__enter__ = Mock(return_value=archive)
                    archive.__exit__ = Mock(return_value=False)
                    archive.open.return_value = io.BytesIO(expected_content)
                    with (
                        patch.object(exp_phase_04f_reference_backend, "ASSETS", assets),
                        patch.object(
                            exp_phase_04f_reference_backend,
                            "REQUIRED_MODEL_ASSET_MEMBERS",
                            ("model.gguf",),
                        ),
                        patch.object(exp_phase_04f_reference_backend, "MODEL_DIR", root),
                        patch.object(exp_phase_04f_reference_backend, "_download_apk") as download,
                        patch.object(
                            exp_phase_04f_reference_backend.zipfile,
                            "ZipFile",
                            return_value=archive,
                        ),
                    ):
                        if existing_content is None:
                            exp_phase_04f_reference_backend._prepare_extracted_assets()
                            self.assertEqual(destination.read_bytes(), expected_content)
                        else:
                            with self.assertRaisesRegex(ValueError, "failed verification"):
                                exp_phase_04f_reference_backend._prepare_extracted_assets()
                    download.assert_called_once_with()

    def test_peak_process_ram_uses_getrusage_when_available(self):
        fake_resource = Mock(RUSAGE_SELF=0)
        fake_resource.getrusage.return_value.ru_maxrss = 1234
        with patch.object(exp_phase_04f_reference_backend, "resource", fake_resource):
            self.assertEqual(exp_phase_04f_reference_backend._peak_process_ram_raw(), 1234)
        fake_resource.getrusage.assert_called_once_with(fake_resource.RUSAGE_SELF)

    def test_peak_process_ram_is_none_without_resource_module(self):
        with patch.object(exp_phase_04f_reference_backend, "resource", None):
            self.assertIsNone(exp_phase_04f_reference_backend._peak_process_ram_raw())


if __name__ == "__main__":
    unittest.main()
