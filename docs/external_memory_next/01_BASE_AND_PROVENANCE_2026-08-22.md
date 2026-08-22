# 01 — Base and Provenance Record

Date: 2026-08-22

## 1. What was done

All local Git worktrees, branches, living-index files, and relevant histories
were inspected before the new research worktree was created.

The isolated worktree was created at:

```text
C:\Users\chiar\Desktop\LBH\thesis-external-memory-next
```

with branch:

```text
work/external-memory-next
```

from commit:

```text
fb09ca2fa50589a0fc72130552212c5b47ed4365
```

## 2. Why this base was selected

`fb09ca2` is the newest committed base that contains the consolidated EM1,
EM2, EM3, standardized Full+Short comparison, Full Initial-to-Full transfer,
and Full RetunedFinal follow-up records.

The Initial-Pinyin branch at `39c5ae1c0a425e8c08b1322a9837e03dc83d3eaa`
is not an ancestor of `fb09ca2`; their merge base is
`80b053764e70ee2f2886892ba516a6b9e2470e59`. Therefore it would have been
incorrect to assume that the latest context-comparison branch contained all
Initial-Pinyin files or to merge the branch wholesale merely to collect them.

## 3. Living-index discrepancy

The two supplied index snapshots were read completely before research code was
modified:

| Supplied file | SHA256 |
|---|---|
| `FILE_INDEX(6).md` | `738dc8064893c00d95def10dda42d61504ea318e8b1003af0f3e24c6dbba0ea3` |
| `REPRODUCIBILITY_INDEX(6).md` | `428a420cd56224843e62104cbb3c648df0948d07e478c6a93a8c34b3ee7bfa35` |

No Git commit contains either exact blob. Relative to `fb09ca2`, the supplied
indexes add the latest Initial-Pinyin navigation and reinforce the living-index
maintenance rule. They were installed as `docs/FILE_INDEX.md` and
`docs/REPRODUCIBILITY_INDEX.md` in this isolated worktree.

This is an index-level unification, not evidence that every referenced large
artifact exists in the new worktree. Sibling worktrees remain read-only
provenance sources, and only required source/document files may be imported
with their original commit/hash recorded.

## 4. Relevant source checkpoints

| Lineage | Commit | Role |
|---|---|---|
| Context comparison / Full RetunedFinal | `fb09ca2fa50589a0fc72130552212c5b47ed4365` | New branch base |
| Initial-Pinyin recovery/context | `39c5ae1c0a425e8c08b1322a9837e03dc83d3eaa` | Read-only source lineage |
| EM3 Dev audit | `5c0427cf1049f4ddda35402f351c6b8d7bd1c6e0` | Ancestor of base |
| EM1/PV same-surface audit | `e500417c3aca4353c7b29956e0836aea53e16e91` | Ancestor of base |
| EM2 closure | `c6ced1183e07b5ab070b8153707b38f51c18dd2f` | Ancestor of base |
| Context Strengthening | `a9a9351c85fe7f40f17c5232e5f77b6c84e7b35c` | Ancestor of base |

## 5. Commands

```powershell
git worktree list --porcelain
git branch --all --verbose --no-abbrev
git log --all -- docs/FILE_INDEX.md docs/REPRODUCIBILITY_INDEX.md
git merge-base <source-commit> fb09ca2
git worktree add -b work/external-memory-next `
  C:\Users\chiar\Desktop\LBH\thesis-external-memory-next `
  fb09ca2fa50589a0fc72130552212c5b47ed4365
```

## 6. Limitations and next decision

The supplied indexes describe local-only result trees across multiple sibling
worktrees. Phase 0 must verify every reused artifact directly, including its
row count, schema, and SHA256, before a method is implemented. Summary prose is
not sufficient evidence.

Current scientific status: **base established; evidence audit in progress;
Dev3000 and Test unused in this phase**.
