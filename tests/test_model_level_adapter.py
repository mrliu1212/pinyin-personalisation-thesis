from __future__ import annotations

import unittest

from src.model_level.pinyin_modes import (
    choose_pinyin_representation,
    full_to_initial,
    select_mode,
)


class PinyinManifestTests(unittest.TestCase):
    def test_official_first_letter_semantics(self) -> None:
        self.assertEqual(
            full_to_initial(("shi", "zhong", "chi", "ke", "yi")),
            ("s", "z", "c", "k", "y"),
        )

    def test_mode_selection_is_deterministic(self) -> None:
        one = select_mode("row-1", 2, seed=20260822)
        two = select_mode("row-1", 2, seed=20260822)
        self.assertEqual(one, two)

    def test_multi_syllable_mixed_is_non_degenerate(self) -> None:
        for index in range(100):
            value = choose_pinyin_representation(
                ("ren", "shi", "dang", "an"),
                row_id=f"row-{index}",
                epoch=index % 3,
                seed=20260822,
                forced_mode="mixed",
            )
            self.assertEqual(value.effective_mode, "mixed")
            self.assertGreater(len(value.abbreviated_positions), 0)
            self.assertLess(len(value.abbreviated_positions), 4)

    def test_single_syllable_mixed_fallback_is_recorded(self) -> None:
        value = choose_pinyin_representation(
            ("ke",),
            row_id="single-row",
            epoch=0,
            seed=20260822,
            forced_mode="mixed",
        )
        self.assertEqual(value.requested_mode, "mixed")
        self.assertIn(value.effective_mode, {"full", "initial"})
        self.assertEqual(len(value.segmented_input), 1)

    def test_one_exposure_has_one_manifest(self) -> None:
        value = choose_pinyin_representation(
            ("shi", "yong"), row_id="row", epoch=1, seed=7
        )
        self.assertEqual(value.row_id, "row")
        self.assertEqual(value.epoch, 1)
        self.assertEqual(len(value.segmented_input), 2)


try:
    import torch
    from torch import nn

    from src.model_level.pinyingpt_adapter import (
        SerialAdapterConfig,
        adapter_state_dict,
        gradient_audit,
        install_adapters,
        load_adapter_state_dict,
        set_adapter_training_mode,
    )
    from src.model_level.concat_training import prepare_concat_example
except ImportError:  # pragma: no cover - only the delivery container lacks torch
    torch = None


if torch is not None:
    class AdapterWrapperTests(unittest.TestCase):
        class FakeBlock(nn.Module):
            def __init__(self, hidden_size: int) -> None:
                super().__init__()
                self.projection = nn.Linear(hidden_size, hidden_size)

            def forward(self, hidden_states, **kwargs):
                return (self.projection(hidden_states), "cache")

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = type("Config", (), {"n_embd": 8})()
                self.transformer = nn.Module()
                self.transformer.h = nn.ModuleList(
                    [AdapterWrapperTests.FakeBlock(8), AdapterWrapperTests.FakeBlock(8)]
                )

            def forward(self, hidden_states):
                value = hidden_states
                for block in self.transformer.h:
                    value = block(value)[0]
                return value

        def config(self):
            return SerialAdapterConfig(reduction_factor=4)

        def test_zero_init_identity_and_parameter_surface(self) -> None:
            torch.manual_seed(1)
            model = self.FakeModel().eval()
            inputs = torch.randn(2, 3, 8)
            baseline = model(inputs).detach()
            audit = install_adapters(model, self.config(), expected_layers=2)
            adapted = model(inputs).detach()
            self.assertTrue(torch.equal(baseline, adapted))
            self.assertEqual(audit["adapter_parameters"], 84)
            self.assertEqual(audit["trainable_parameters"], 84)

        def test_adapter_inherits_base_device_and_dtype(self) -> None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dtype = torch.float32 if device.type == "cuda" else torch.float64
            model = self.FakeModel().to(device=device, dtype=dtype)
            inputs = torch.randn(1, 2, 8, device=device, dtype=dtype)
            baseline = model(inputs).detach()
            install_adapters(model, self.config(), expected_layers=2)
            adapted = model(inputs).detach()
            adapter_parameter = next(model.transformer.h[0].adapter.parameters())
            self.assertEqual(adapter_parameter.device, device)
            self.assertEqual(adapter_parameter.dtype, dtype)
            self.assertTrue(torch.equal(baseline, adapted))

        def test_only_adapter_receives_gradient(self) -> None:
            torch.manual_seed(2)
            model = self.FakeModel()
            install_adapters(model, self.config(), expected_layers=2)
            set_adapter_training_mode(model, enabled=True)
            model(torch.randn(2, 3, 8)).sum().backward()
            audit = gradient_audit(model)
            self.assertTrue(audit["adapter_gradient_present"])
            self.assertTrue(audit["adapter_nonzero_gradient_present"])
            self.assertTrue(audit["base_gradient_absent"])

        def test_adapter_state_round_trip(self) -> None:
            torch.manual_seed(3)
            first = self.FakeModel()
            install_adapters(first, self.config(), expected_layers=2)
            with torch.no_grad():
                first.transformer.h[0].adapter.up.bias.fill_(0.25)
            state = adapter_state_dict(first)

            torch.manual_seed(3)
            second = self.FakeModel()
            install_adapters(second, self.config(), expected_layers=2)
            load_adapter_state_dict(second, state)
            inputs = torch.randn(1, 2, 8)
            self.assertTrue(torch.equal(first(inputs), second(inputs)))

    class ConcatGeometryTests(unittest.TestCase):
        class FakeTokenizer:
            pad_token_id = 0
            unk_token_id = 999
            cls_token_id = 101
            sep_token_id = 102

            def __init__(self) -> None:
                self.tokens = {
                    "甲": 201,
                    "乙": 202,
                    "[jia]": 301,
                    "[j]": 302,
                    "[yi]": 303,
                    "[y]": 304,
                }
                self.reverse = {value: key for key, value in self.tokens.items()}

            def encode(self, text, add_special_tokens=False):
                return [400 + index for index, _ in enumerate(text)]

            def convert_tokens_to_ids(self, values):
                if isinstance(values, str):
                    return self.tokens.get(values, self.unk_token_id)
                return [self.tokens.get(value, self.unk_token_id) for value in values]

            def convert_ids_to_tokens(self, value):
                return self.reverse.get(value, "[UNK]")

        class FakeBackend:
            def __init__(self) -> None:
                self.tokenizer = ConcatGeometryTests.FakeTokenizer()
                self.model = type("Model", (), {"config": type("Config", (), {"n_positions": 20})()})()
                self.allowed_token_ids = {
                    "jia": (201,),
                    "j": (201,),
                    "yi": (202,),
                    "y": (202,),
                }

            def segment_pinyin(self, values):
                segments = tuple(values)
                if any(value not in self.allowed_token_ids for value in segments):
                    raise ValueError("unknown segment")
                return segments

            def _prompt(self, context, pinyin):
                context_ids = self.tokenizer.encode(context, add_special_tokens=False)
                pinyin_ids = self.tokenizer.convert_tokens_to_ids([f"[{value}]" for value in pinyin])
                prompt = [101, *context_ids, 102, *pinyin_ids, 102]
                first_separator = 1 + len(context_ids)
                positions = list(range(first_separator + 1)) + list(
                    range(first_separator + 1, first_separator + len(pinyin) + 2)
                )
                return prompt, positions

        def test_teacher_input_omits_last_gold_and_uses_exact_loss_positions(self) -> None:
            backend = self.FakeBackend()
            row = {
                "row_id": "row",
                "author": "Agent Phage",
                "condition": "full_short",
                "target_type": "short",
                "pinyin_segments": ["jia", "yi"],
                "gold": "甲乙",
                "target": "甲乙",
                "context": "前文",
            }
            representation = choose_pinyin_representation(
                row["pinyin_segments"],
                row_id="row",
                epoch=0,
                seed=1,
                forced_mode="full",
            )
            example = prepare_concat_example(backend, row, representation)
            prompt, _ = backend._prompt(row["context"], ("jia", "yi"))
            self.assertEqual(len(example.input_ids), len(prompt) + 1)
            self.assertEqual(example.input_ids[-1], 201)
            self.assertEqual(example.loss_positions, (len(prompt) - 1, len(prompt)))
            self.assertEqual(example.target_token_ids, (201, 202))
else:
    @unittest.skip("PyTorch is unavailable")
    class AdapterWrapperTests(unittest.TestCase):
        def test_pytorch_required(self) -> None:
            pass


if __name__ == "__main__":
    unittest.main()
