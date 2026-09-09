// Ampere fork: throwing stubs for W8 launchers whose kernels exceed the sm_86
// 48 KiB static-shared limit (sm_120a-tuned tiles). Compiled only when
// NINFER_DISABLE_W8_BIG_SMEM is set. Every stub matches the W8Launch signature
// from ops/linear/w8/w8_launch.h, so the dispatch tables link unchanged; the
// entry points are unreachable in the Ampere subset (groupwise-int text-only,
// --spec none, no vision: no W8 weights execute) and fail fast if reached.

#include "ops/linear/w8/w8_launch.h"

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

} // namespace ninfer::ops::detail
