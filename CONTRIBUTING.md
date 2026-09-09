# Contributing to NInfer (Ampere fork)

Thank you for considering contributing! This document explains how you can help make NInfer better on Ampere GPUs (RTX 3060/3060 Ti, `sm_86`).

## Table of Contents
1. [Reporting Issues](#reporting-issues)
2. [Contributing Code](#contributing-code)
3. [Building & Testing](#building--testing)
4. [Code Style](#code-style)
5. [License](#license)

## Reporting Issues
- Use the **[GitHub Issues](https://github.com/Rafie-kun/ninfer/issues)** tracker.
- Include:
  - GPU model and driver version.
  - CUDA toolkit version.
  - CMake/Ninja/build environment.
  - Steps to reproduce the problem.
  - Any relevant output/logs (especially `Ampere fork:` messages).
- **Before posting**, search existing issues – many problems (e.g. unsupported KV dtype, NVFP4 on sm_86) are already documented in [docs/ampere-fork.md](docs/ampere-fork.md).
- If you’re filing a new issue, please use the appropriate template under `.github/ISSUE_TEMPLATE/` (bug report or feature request) so maintainers have the right information up front.

## Contributing Code
1. **Fork the repo** and create a branch from `main`:
   ```bash
   git checkout -b feat/short-desc
   ```
2. **Commit messages** follow the conventional‑commit style (lower‑case verb, optional scope, concise description):
   ```
   fix(ampere): stub oversized w8 kernel for sm_86
   ```
3. **Keep changes focused** – avoid mixing unrelated fixes in one PR.
4. **Add or update tests** if you change behaviour; the project uses a minimal test suite (see `tests/README.md` in upstream).
5. **Run the build** with `-DCMAKE_CUDA_ARCHITECTURES=86` to verify the Ampere compile still passes.
6. **Open a Pull Request** targeting `main` with a clear title and description. Include:
   - What you changed and why.
   - Any relevant issue numbers.
   - Build commands you used to verify the change.

## Building & Testing
```bash
# Clone & bootstrap (Linux host with CUDA 13.1+)
git clone https://github.com/Rafie-kun/ninfer.git
cd ninfer
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j   # compiles only sm_86 kernels
```
- The default build disables NVFP4/FP8‑MMA and stubs oversized W8 kernels for `sm_86`.
- To test a change, rebuild and run the example:
  ```bash
  ./build/apps/ninfer models/qwen3_6_27b.ninfer \
    --prompt "Hello" --max-context 4096 --kv-capacity 4096 --max-new 64 \
    --kv-dtype int8 --max-concurrency 1
  ```
- If a change breaks the `sm_86` compile, the CI will reject the PR (the fork marks `sm_86` build verified at commit `bef2fa72`).

## Code Style
- C++ follows the project’s existing style (mostly C++20, no exceptions where avoided, use of `cstdint`, etc.).
- CMake lists and `target_compile_definitions` are auto‑ON for `sm_86|sm_89`; keep `NINFER_DISABLE_NVFP4` and `NINFER_DISABLE_FP8_MMA` toggles where appropriate.
- Keep `docs/ampere-fork.md` up‑to‑date if you add or remove a supported feature.

## License
By contributing, you agree that your contributions are licensed under the project’s [MIT licence](LICENSE) (or whatever licence this fork uses). See the root `LICENSE` file for details.

---
*Thanks for helping keep NInfer running well on Ampere GPUs!*