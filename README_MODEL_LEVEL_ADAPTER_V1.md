# Model-Level Adapter V1 delivery

Package revision V4 fixes Adapter installation after a frozen base has already
been moved to CUDA: every new Adapter inherits its wrapped GPT2Block's device
and dtype before the block is installed. It also writes a frozen Generic
platform reference for the later Windows-to-cluster equivalence gate.

Copy this package into the clean worktree:

```text
C:\Users\chiar\Desktop\LBH\thesis-model-level
```

The archive paths already begin with `src`, `experiments`, `tests`, and `docs`.
Extracting the archive into the worktree root will place every file correctly.

Validate source and pure/unit gates first:

```powershell
Set-Location 'C:\Users\chiar\Desktop\LBH\thesis-model-level'
$py = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'

& $py -m compileall `
    '.\src\model_level' `
    '.\experiments\model_level' `
    '.\tests\test_model_level_adapter.py'

& $py -m unittest discover `
    -s '.\tests' `
    -p 'test_model_level_adapter.py' `
    -v
```

Then run the real PinyinGPT hard gates:

```powershell
$checkpoint = 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat'
$fit = 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_v2\clean3_train_fit_v1.jsonl'

& $py -m experiments.model_level.run_adapter_hard_gates_v1 `
    --checkpoint $checkpoint `
    --fit $fit `
    --output-root '.\results\model_level\adapter_v1_hard_gates' `
    --author 'Agent Phage' `
    --device cuda
```

Do not stage generated `results/` or smoke Adapter weights. The runner does not
perform formal training and records `used_dev3000=false` and `used_test=false`.

The Windows run creates:

```text
results/model_level/adapter_v1_hard_gates/generic_platform_reference.json
```

Copy that small file to the cluster. The cluster invocation uses the same
command plus:

```text
--platform-reference <copied Windows generic_platform_reference.json>
```
