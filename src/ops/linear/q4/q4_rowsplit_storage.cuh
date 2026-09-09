#pragma once

#include "ops/common/math.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Q4RowSplitStorage {
    static constexpr int kGroupK             = 64;
    static constexpr int kCodeBytesPerGroup  = 32;
    static constexpr int kScaleBytesPerGroup = 2;
};

struct Q3RowSplitStorage {
    static constexpr int kGroupK             = 64;
    static constexpr int kCodeBytesPerGroup  = 24;  // 3 bits x 8 codes per byte
    static constexpr int kScaleBytesPerGroup = 2;
};

struct Q4SimtDecodeAtom {
    __device__ static __forceinline__ void
    decode_eight(std::uint32_t packed, std::uint16_t scale_bits, float (&weights)[8]) {
        const std::uint32_t word = packed ^ 0x88888888u;
        const float scale        = __half2float(__ushort_as_half(scale_bits));
        const __half2 bias       = __half2half2(__ushort_as_half(0x6408)); // 1032.0
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const std::uint32_t bits = ((word >> (4 * pair)) & 0x000f000fu) | 0x64006400u;
            const __half2 decoded    = __hsub2(half2_from_bits(bits), bias);
            const float2 values      = __half22float2(decoded);
            weights[pair]            = values.x * scale;
            weights[pair + 4]        = values.y * scale;
        }
    }

    __device__ static __forceinline__ void
    decode_pair(std::uint8_t packed, std::uint16_t scale_bits, float& w0, float& w1) {
        const float scale = __half2float(__ushort_as_half(scale_bits));
        const int q0      = (static_cast<int>(packed & 0x0fu) ^ 0x08) - 0x08;
        const int q1      = (static_cast<int>(packed >> 4) ^ 0x08) - 0x08;
        w0                = static_cast<float>(q0) * scale;
        w1                = static_cast<float>(q1) * scale;
    }
};

struct Q4MmaDecodeAtom {
    // Eight packed codes -> four bf16 pairs, in weight order; out[i] holds the pair for
    // lane = 4 * chunk + i. A nibble n decodes to the integer (n ^ 8) - 8, which lies in [-8, 7]
    // and which bf16 represents exactly, so forming 128 + (n ^ 8) and subtracting 136 lands on it.
    // The values are unscaled: callers of this overload fold the group scale into the
    // accumulation, unlike decode_pair below.
    static __device__ __forceinline__ void decode_eight(unsigned word, unsigned (&out)[4]) {
        const unsigned kBias  = 0x43084308u; // bf16 136.0 in both halves
        const unsigned kMagic = 0x43004300u; // bf16 128.0 in both halves
        word ^= 0x88888888u;
        const unsigned lo        = word & 0x0f0f0f0fu;
        const unsigned hi        = (word >> 4) & 0x0f0f0f0fu;
        const unsigned even      = __byte_perm(lo, hi, 0x5140);
        const unsigned odd       = __byte_perm(lo, hi, 0x7362);
        const unsigned biased[4] = {
            __byte_perm(even, kMagic, 0x7150), __byte_perm(even, kMagic, 0x7372),
            __byte_perm(odd, kMagic, 0x7150), __byte_perm(odd, kMagic, 0x7372)};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const __nv_bfloat162 value =
                __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&biased[i]),
                        *reinterpret_cast<const __nv_bfloat162*>(&kBias));
            out[i] = *reinterpret_cast<const unsigned*>(&value);
        }
    }

    static __device__ __forceinline__ __nv_bfloat162 decode_pair(const std::uint8_t* codes,
                                                                 const std::uint8_t* scale_ptr,
                                                                 std::int64_t group_index,
                                                                 int lane) {
        const float scale =
            __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scale_ptr)));
        const std::uint8_t packed =
            codes[group_index * Q4RowSplitStorage::kCodeBytesPerGroup + lane];
        const int q0 = (static_cast<int>(packed & 0x0fu) ^ 0x08) - 0x08;
        const int q1 = (static_cast<int>(packed >> 4) ^ 0x08) - 0x08;
        return __floats2bfloat162_rn(static_cast<float>(q0) * scale,
                                     static_cast<float>(q1) * scale);
    }
};

} // namespace ninfer::ops::detail

struct Q3SimtDecodeAtom {
    __device__ static __forceinline__ void
    decode_eight(std::uint32_t packed, std::uint16_t scale_bits, float (&weights)[8]) {
        const std::uint32_t word = packed ^ 0xCCCCCCCCu;
        const float scale        = __half2float(__ushort_as_half(scale_bits));
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            // 3-bit decode: each nibble holds 2 codes of 3 bits each
            const std::uint32_t bits = ((word >> (3 * pair)) & 0x000007Fu);
            const int q0 = (static_cast<int>(bits & 0x07)) - 4;
            const int q1 = (static_cast<int>((bits >> 3) & 0x07)) - 4;
            weights[pair] = static_cast<float>(q0) * scale;
            weights[pair + 4] = static_cast<float>(q1) * scale;
        }
    }

    __device__ static __forceinline__ void
    decode_pair(std::uint8_t packed, std::uint16_t scale_bits, float& w0, float& w1) {
        const float scale = __half2float(__ushort_as_half(scale_bits));
        // 3-bit: 8 codes pack into 2 bytes (6 bits used, 2 padding)
        const std::uint32_t w0_bits = ((std::uint32_t)packed & 0x07) - 4;
        const std::uint32_t w1_bits = (((std::uint32_t)packed >> 3) & 0x07) - 4;
        w0 = static_cast<float>(w0_bits) * scale;
        w1 = static_cast<float>(w1_bits) * scale;
    }
};

struct Q3MmaDecodeAtom {
    // Three packed codes -> two bf16 pairs. 3 bits per code, values in [-4, 3].
    // Decode: subtract 4 to get signed int, multiply by scale.
    static __device__ __forceinline__ void decode_triple(std::uint32_t packed, std::uint16_t scale_bits, float (&weights)[6]) {
        const float scale = __half2float(__ushort_as_half(scale_bits));
        // Each 3-bit code occupies 2 bits + 1 padding; pack 8 codes into 3 bytes (24 bits)
        // Codes at positions 0-2, 3-5, 6-7 (last has only 2 codes)
        for (int i = 0; i < 3; ++i) {
            const std::uint32_t code_i = (packed >> (3 * i)) & 0x07;
            weights[i] = static_cast<float>(code_i - 4) * scale;
        }
        // Last 'code' at position 6-7 is actually just one code
        const std::uint32_t code_6 = (packed >> 18) & 0x07;
        weights[5] = static_cast<float>(code_6 - 4) * scale;
    }

    static __device__ __forceinline__ __nv_bfloat162 decode_pair(const std::uint8_t* codes,
                                                                 const std::uint8_t* scale_ptr,
                                                                 std::int64_t group_index,
                                                                 int lane) {
        const float scale =
            __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scale_ptr)));
        // Q3: 3 bits per code, 8 codes per group = 24 bits = 3 bytes
        const std::uint8_t packed = codes[group_index * Q3RowSplitStorage::kCodeBytesPerGroup + lane];
        // Decode first two codes from first 6 bits
        const std::uint32_t code0 = (packed & 0x07) - 4;
        const std::uint32_t code1 = ((packed >> 3) & 0x07) - 4;
        return __floats2bfloat162_rn(static_cast<float>(code0) * scale,
                                     static_cast<float>(code1) * scale);
    }
};

} // namespace ninfer::ops::detail
