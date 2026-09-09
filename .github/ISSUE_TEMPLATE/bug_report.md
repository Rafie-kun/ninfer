---
name: Bug Report
description: File a report to help us improve
title: "[Bug] <short title>"
labels: ['bug', 'triage']
assignees: ''
---

**Please search for existing issues** first to avoid duplicates.

**GPU & Driver**
- GPU model: e.g., RTX 3060 12GB
- Driver version: e.g., 535.104.05
- CUDA toolkit version: e.g., 13.3

**NInfer version / commit**
- Version (if tagged) or commit hash:

**Steps to reproduce**
1. ...
2. ...

**Expected behaviour**
...

**Actual behaviour**
...

**Build configuration**
```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
```

**Additional context**
- Any `Ampere fork:` messages seen in the output.
- KV dtype used (`--kv-dtype bf16` or `int8`).
- Context sizes (`--max-context`, `--kv-capacity`).