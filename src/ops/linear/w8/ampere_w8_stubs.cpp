// Ampere fork: throwing stubs for W8 launchers whose kernels exceed the sm_86
// 48 KiB static-shared limit (sm_120a-tuned tiles). Compiled only when
// NINFER_DISABLE_W8_BIG_SMEM is set. Signatures match the w8_*_kernels.h
// declarations verbatim; the dispatch tables link unchanged. The entry points
// are unreachable in the Ampere subset (groupwise-int text-only, --spec none,
// no vision: no W8 weights execute) and fail fast if reached.

#include "ops/linear/w8/w8_launch.h"

#include "ops/attn_input_proj/w8/w8_attn_input_kernels.h"
#include "ops/dynamic_grouped_conv/w8/w8_dynamic_grouped_conv_add_kernels.h"
#include "ops/gdn_input_proj/w8/w8_gdn_input_kernels.h"
#include "ops/linear_add/w8/w8_linear_add_kernels.h"
#include "ops/linear_pair/w8/w8_pair_kernels.h"
#include "ops/linear_pair/w8/w8_pair_plan.h"
#include "ops/linear_swiglu/w8/w8_linear_swiglu_kernels.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

namespace {

[[noreturn]] void throw_ampere_w8(const char* entry) {
    throw std::invalid_argument(std::string("Ampere fork: ") + entry +
                                " needs >48 KiB static shared memory (sm_120a-tuned tile); "
                                "W8/MTP paths are unavailable in this build");
}

} // namespace

void launch_w8_small_t(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    throw_ampere_w8("launch_w8_small_t");
}

void launch_w8_exact_t_splitk(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    throw_ampere_w8("launch_w8_exact_t_splitk");
}

void launch_w8_exact_t_composite(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    throw_ampere_w8("launch_w8_exact_t_composite");
}

void launch_w8_dflash_medium(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    throw_ampere_w8("launch_w8_dflash_medium");
}

void launch_w8_medium_splitk_c144(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    throw_ampere_w8("launch_w8_medium_splitk_c144");
}

void launch_w8_mma_r64_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    throw_ampere_w8("launch_w8_mma_r64_c96");
}

void launch_w8_exact_mma_r64_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    throw_ampere_w8("launch_w8_exact_mma_r64_c96");
}

// --- Splitk/small-T family (excluded .cu files below) ---

void w8_attn_input_splitk_mma_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&,
                                     Tensor&, cudaStream_t) {
    throw_ampere_w8("w8_attn_input_splitk_mma_launch");
}

void w8_attn_input_splitk_mma_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&,
                                     cudaStream_t) {
    throw_ampere_w8("w8_attn_input_splitk_mma_launch");
}

void w8_dflash2_attn_input_small_t_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&,
                                          cudaStream_t) {
    throw_ampere_w8("w8_dflash2_attn_input_small_t_launch");
}

void w8_dflash2_attn_input_mma_r32_c64_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                             Tensor&, cudaStream_t) {
    throw_ampere_w8("w8_dflash2_attn_input_mma_r32_c64_launch");
}

void w8_dflash2_attn_input_mma_r64_c128_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                              Tensor&, cudaStream_t) {
    throw_ampere_w8("w8_dflash2_attn_input_mma_r64_c128_launch");
}

void w8_dflash2_attn_input_mma_r16_c64_k128_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                                   Tensor&, cudaStream_t) {
    throw_ampere_w8("w8_dflash2_attn_input_mma_r16_c64_k128_launch");
}

void w8_dflash2_attn_input_mma_r32_c32_k128_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                                   Tensor&, cudaStream_t) {
    throw_ampere_w8("w8_dflash2_attn_input_mma_r32_c32_k128_launch");
}

void w8_dflash2_attn_input_mma_r32_c64_k128_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                                   Tensor&, cudaStream_t) {
    throw_ampere_w8("w8_dflash2_attn_input_mma_r32_c64_k128_launch");
}

void w8_gdn_input_splitk_mma_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                    cudaStream_t) {
    throw_ampere_w8("w8_gdn_input_splitk_mma_launch");
}

void w8_gdn_input_splitk_conv_snapshot_launch(
    const Tensor&, const Weight&, const Tensor&, Tensor&, const Tensor&, const Tensor&,
    const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&, cudaStream_t) {
    throw_ampere_w8("w8_gdn_input_splitk_conv_snapshot_launch");
}

void w8_gdn_input_splitk_conv_record_launch(
    const Tensor&, const Weight&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    Tensor&, Tensor&, Tensor&, Tensor&, Tensor&, cudaStream_t) {
    throw_ampere_w8("w8_gdn_input_splitk_conv_record_launch");
}

void w8_linear_add_splitk_mma_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    throw_ampere_w8("w8_linear_add_splitk_mma_launch");
}

void w8_linear_add_medium_splitk_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    throw_ampere_w8("w8_linear_add_medium_splitk_launch");
}

void w8_pair_splitk_exact_t_launch(const Tensor&, const Weight&, const Weight&, Tensor&, Tensor&,
                                   cudaStream_t) {
    throw_ampere_w8("w8_pair_splitk_exact_t_launch");
}

void w8_pair_splitk_medium_launch(W8PairScheduleId, const Tensor&, const Weight&, const Weight&,
                                  Tensor&, Tensor&, cudaStream_t) {
    throw_ampere_w8("w8_pair_splitk_medium_launch");
}

void w8_linear_swiglu_splitk_exact_t_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    throw_ampere_w8("w8_linear_swiglu_splitk_exact_t_launch");
}

void w8_dflash2_linear_swiglu_small_t_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    throw_ampere_w8("w8_dflash2_linear_swiglu_small_t_launch");
}

void w8_dflash2_linear_swiglu_mma_r64_c96_k128_launch(const Tensor&, const Weight&, Tensor&,
                                                      cudaStream_t) {
    throw_ampere_w8("w8_dflash2_linear_swiglu_mma_r64_c96_k128_launch");
}

void w8_dynamic_grouped_conv_add_materialized_launch(W8DynamicConvAddSchedule, const Tensor&,
                                                     const Weight&, const Tensor&, const Tensor&,
                                                     Tensor&, Tensor&, cudaStream_t) {
    throw_ampere_w8("w8_dynamic_grouped_conv_add_materialized_launch");
}

} // namespace ninfer::ops::detail
