# Ampere fork (RTX 3060 / 3060 Ti, sm_86) — status and guide

Experimental fork of [Neroued/ninfer](https://github.com/Neroued/ninfer) for Ampere GPUs.
Upstream targets one RTX 5090 (`sm_120a`) only. This fork builds for `sm_86`
(RTX 3060 12GB / RTX 3060 Ti 8GB — both are `sm_86`, so the build is identical;
only usable VRAM differs: ~11 GiB vs ~7.5 GiB).

## Honest status: 27B does not run on 8–12GB yet

The smallest registered weights exceed both cards before any KV or workspace:

| Artifact | File tensor bytes |
|---|---|
| `qwen3_6_27b.ninfer` (groupwise-int) | ~16.3 GiB |
| `qwen3_6_27b_nvfp4.ninfer` | ~17.1 GiB (also `sm_120a`-only kernels) |
| `qwen3_8_27b.ninfer` (groupwise-int) | ~20.4 GiB |
| `qwen3.6-35B-A3B` (groupwise-int) | ~22.8 GiB |

Sources: `model-cards/*-NInfer/README.md` weight tables. Startup pins the full
resident set (one GPU, one resident model, no weight offload — deliberate upstream
boundary, see `docs/maintainer/engine-architecture.md`), so startup OOMs on
8–12GB regardless of context settings. **No release of this fork can run 27B
inference until weight residency is solved** (§Roadmap). The `sm_86` build itself
is also still unverified — this container has no CUDA/GPU; first compile must run
on an Ampere box with CUDA 13.1.

## Supported subset on Ampere

| Feature | Upstream (sm_120a) | This fork (sm_86) |
|---|---|---|
| Weights | groupwise-int + NVFP4 | groupwise-int only (`qwen3_6_27b.ninfer`); NVFP4 artifacts fail fast |
| KV dtype | bf16 / int8 / fp8 / nvfp4 / k8v4 | bf16 / int8 only |
| Speculative | none / mtp / dflash / dflash2 | none only (MTP needs stubbed W8 kernels) |
| Vision (`--vision`) | yes | rejected |
| Context | up to 240k+ | 2–4k (8GB) / 4–8k (12GB), C=1 for bring-up |

## How it works

Two CMake options (auto-ON for arch `86|89`, `CMakeLists.txt`) compile out
Blackwell-only code and define `NINFER_DISABLE_NVFP4=1` / `NINFER_DISABLE_FP8_MMA=1`:

- `NINFER_DISABLE_NVFP4`: drops the `ninfer_nvfp4_non_rdc` TMA archive and all
  NVFP4/K8V4 `.cu` (mxf4 MMA, TMA/`mbarrier`, e2m1 converts don't exist on sm_86).
- `NINFER_DISABLE_FP8_MMA`: drops FP8 MMA/A8 `.cu` (f8 MMA needs sm_90+).
  FP8 A16 SIMT decode stays.
- `NINFER_DISABLE_PDL`: replaces programmatic dependent launch (`griddepcontrol`,
  sm_90+) with plain same-stream launches plus no-op device hooks
  (`src/core/pdl.cuh`). Same ordering, less overlap; all PDL pairs share one stream.
- `NINFER_DISABLE_W8_BIG_SMEM`: drops W8 kernels over the sm_86 48 KiB static-shared
  cap (`w8_small_t` 0xc200–0xe200, splitk up to 0x16000, one r64_c96 tile at 0xc400)
  and stubs their launchers (`src/ops/linear/w8/ampere_w8_stubs.cpp`). Retuning
  those tiles for Ampere is open perf work; MTP stays rejected until then.

Kept host dispatchers that used to call those kernels now throw an `Ampere fork: …`
`invalid_argument` naming the supported alternative (`src/ops/**/…_plan.cpp`,
`…/nvfp4_dispatch.cpp`, `…/fp8_dispatch.cpp`, `kv_cache/append/*`,
`softmax_attention/…/small_t.cu` + `prompt.cu`, `linear_topk.cpp`,
`wrapper/gdn_input_proj.cpp`). Runtime rejects the rest at startup with the same
prefix: NVFP4 artifacts (`src/targets/qwen3_6_27b/impl/package.cpp`), non-bf16/int8
KV, vision, and DFlash backends (`src/targets/qwen3_6/impl/runtime/layouts_impl.h`,
CC 86/89/90/120 allowed). If you see `Ampere fork:`, the request is outside the
supported subset — not a bug.

## Build and run

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
hf download neroued/Qwen3.6-27B-NInfer qwen3_6_27b.ninfer --local-dir models
# 12GB card:
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --prompt "Reply with one short sentence." \
  --max-context 4096 --kv-capacity 4096 --max-new 256 \
  --kv-dtype int8 --max-concurrency 1
# 8GB card: halve to --max-context 2048 --kv-capacity 2048.
```

Rules of thumb: `--kv-dtype int8` halves KV vs bf16 (Qwen3.6-27B: ~33 vs ~64 KiB/token
— 16 full-attention layers × 4 KV heads × 256 dim; GDN layers hold fixed state).
`--kv-capacity auto` computes from post-weights free memory and will refuse to start
when weights already exceed VRAM. `--spec none`, no `--vision`, C=1.

## Roadmap to actually running on 8–12GB

In increasing effort order; each is a product change, none is implemented:

1. **Weight streaming/offload** — page layers from host on demand. Breaks the
   "one resident model, no offload" boundary; needs paging, prefetch, and
   residency redesign across runtime/targets.
2. **~2b/w registered quant format** — new `QType`, converter (`tools/artifact/`,
   `tools/convert/`), SIMT kernels, target bindings, and quality evidence per
   `docs/maintainer/tensor-formats.md` §11. Needs ≈2.2 avg bits/word to land 27B
   in ~7 GiB.
3. **Smaller registered model** — new target package (shapes, bindings, frontend)
   for a small Qwen checkpoint; same admission cost as (2) plus a new identity.

Also open: verifying the sm_86 compile, retuning sm_120a-tuned tiles (e.g. large
shared-memory W8/Q4 kernels) for Ampere occupancy, and measuring whatever fits.
