cmake_minimum_required(VERSION 3.28)

# Ampere/Blackwell fork: allow sm_86 (3060 Ti), sm_89, sm_90, sm_120a.
# Original upstream is sm_120a-only; this fork downgrades to Ampere text-only.
# Ampere (sm_86/sm_89) has no mxf4/f8 MMA, TMA, mbarrier, or e2m1 converts, so
# NVFP4 weights and FP8 MMA routes are compiled out. Supported Ampere subset:
# groupwise-int text-only, BF16/INT8 KV, --spec none (MTP optional), no vision.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to build")
endif()
set(NINFER_SUPPORTED_ARCHS "86;86a;89;90;120;120a")
if(NOT CMAKE_CUDA_ARCHITECTURES IN_LIST NINFER_SUPPORTED_ARCHS)
  message(WARNING
    "NInfer Ampere fork: CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} "
    "not in tested set ${NINFER_SUPPORTED_ARCHS}. Build may fail on NVFP4/FP8 paths.")
endif()
option(NINFER_DISABLE_NVFP4 "Disable NVFP4 tensor paths (required on Ampere)" OFF)
option(NINFER_DISABLE_FP8_MMA "Disable FP8 MMA tensor paths (required on Ampere)" OFF)
option(NINFER_DISABLE_PDL "Disable programmatic dependent launch (required pre-sm_90)" OFF)
option(NINFER_DISABLE_W8_BIG_SMEM "Stub W8 kernels over the 48 KiB static-shared cap" OFF)
if(CMAKE_CUDA_ARCHITECTURES MATCHES "86|89")
  set(NINFER_DISABLE_NVFP4 ON)
  set(NINFER_DISABLE_FP8_MMA ON)
  set(NINFER_DISABLE_PDL ON)
  set(NINFER_DISABLE_W8_BIG_SMEM ON)
endif()
if(NINFER_DISABLE_PDL)
  message(STATUS "Ampere fork: programmatic dependent launch disabled (griddepcontrol needs sm_90+)")
  add_compile_definitions(NINFER_DISABLE_PDL=1)
endif()
if(NINFER_DISABLE_W8_BIG_SMEM)
  message(STATUS "Ampere fork: oversized W8 kernels stubbed (48 KiB static-shared cap)")
  add_compile_definitions(NINFER_DISABLE_W8_BIG_SMEM=1)
endif()
if(NINFER_DISABLE_NVFP4)
  message(STATUS "Ampere fork: NVFP4 tensor paths disabled")
  add_compile_definitions(NINFER_DISABLE_NVFP4=1)
endif()
if(NINFER_DISABLE_FP8_MMA)
  message(STATUS "Ampere fork: FP8 MMA tensor paths disabled (FP8 A16 decode stays available)")
  add_compile_definitions(NINFER_DISABLE_FP8_MMA=1)
endif()

project(ninfer LANGUAGES C CXX CUDA)

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI and server" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS cuda_link=1)
  set(CMAKE_JOB_POOL_LINK cuda_link)
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.1)
  message(FATAL_ERROR
    "NInfer requires CUDA 13.1 or newer; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()

set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_MEDIA_ACQUIRE ON)
endif()

set(NINFER_BUILD_PROMPT_INPUT OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_PROMPT_INPUT ON)
endif()

set(NINFER_BUILD_SERVE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_SERVE ON)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
  libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
if(NINFER_BUILD_MEDIA_ACQUIRE)
  pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
endif()
find_package(Threads REQUIRED)

if(NINFER_BUILD_APPS OR BUILD_TESTING)
  # Product operational logging is an offline, repository-pinned dependency. Core-only builds do
  # not configure or link it.
  add_subdirectory(third_party/spdlog)
endif()

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
