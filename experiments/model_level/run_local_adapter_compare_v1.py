from pathlib import Path

from src.reference_backend_pinyingpt.backend import PinyinGPTConcatBackend
from src.model_level.pinyingpt_adapter import (
    SerialAdapterConfig,
    install_adapters,
    load_adapter_bundle,
    set_adapter_training_mode,
)

ROOT = Path(__file__).resolve().parents[2]

BASE = ROOT / "local_artifacts" / "base_model" / "pinyingpt2-concat"
ADAPTER = ROOT / "local_artifacts" / "adapters" / "static" / "agent_full55926.safetensors"

CONTEXT = ""
PINYIN = "shi"
TOP_K = 10

generic = PinyinGPTConcatBackend(checkpoint=BASE)

personalised = PinyinGPTConcatBackend(checkpoint=BASE)
install_adapters(personalised.model, SerialAdapterConfig())
load_adapter_bundle(personalised.model, ADAPTER)
set_adapter_training_mode(personalised.model, enabled=False)

g = generic.generate(
    context=CONTEXT,
    typed_pinyin=PINYIN,
    top_k=TOP_K,
)

p = personalised.generate(
    context=CONTEXT,
    typed_pinyin=PINYIN,
    top_k=TOP_K,
)

g_rank = {x.text: x.rank for x in g.candidates}
p_rank = {x.text: x.rank for x in p.candidates}

print(f"context = {CONTEXT!r}")
print(f"pinyin  = {PINYIN}")
print()

print("GENERIC")
for x in g.candidates:
    new_rank = p_rank.get(x.text)
    change = "-" if new_rank is None else f"{x.rank}->{new_rank}"
    print(f"{x.rank:2d}. {x.text:<4}  logP={x.log_probability:8.3f}  rank={change}")

print()
print("AGENT FULL55926")
for x in p.candidates:
    old_rank = g_rank.get(x.text)
    change = "-" if old_rank is None else f"{old_rank}->{x.rank}"
    print(f"{x.rank:2d}. {x.text:<4}  logP={x.log_probability:8.3f}  rank={change}")
