// Minimal CUDA driver API declarations for the NVENC encoders on Windows.
//
// Linux builds use the CUDA toolkit's real cuda.h and lazy-load libcuda.so
// through generated stubs. On Windows this header and cuda_loader.cpp stand in
// for both: the build needs no CUDA toolkit, the binary does not import
// nvcuda.dll, and machines without an NVIDIA driver still start (every call
// then returns CUDA_ERROR_NOT_INITIALIZED and NVENC reports itself
// unavailable).
//
// Only the subset the encoder sources use is declared. Struct layouts, enum
// values and the exported (_v2) symbol names follow the CUDA 12 driver API
// ABI for 64-bit Windows.

#ifndef WEBRTC_SYS_NVIDIA_CUDA_SHIM_CUDA_H
#define WEBRTC_SYS_NVIDIA_CUDA_SHIM_CUDA_H

// The real cuda.h's guard; NvCodecUtils.h keys its CUDA helpers on it.
#define __cuda_cuda_h__

#include <stddef.h>

// Selects the pre-CUDA-13 cuCtxCreate signature in the encoder sources.
#define CUDA_VERSION 12000

#if defined(_WIN32)
#define CUDAAPI __stdcall
#else
#define CUDAAPI
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cudaError_enum {
  CUDA_SUCCESS = 0,
  CUDA_ERROR_INVALID_VALUE = 1,
  CUDA_ERROR_OUT_OF_MEMORY = 2,
  CUDA_ERROR_NOT_INITIALIZED = 3,
  CUDA_ERROR_NO_DEVICE = 100,
  CUDA_ERROR_INVALID_DEVICE = 101,
  CUDA_ERROR_INVALID_CONTEXT = 201,
  CUDA_ERROR_NOT_FOUND = 500,
  CUDA_ERROR_NOT_SUPPORTED = 801,
  CUDA_ERROR_UNKNOWN = 999
} CUresult;

typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st* CUcontext;
typedef struct CUarray_st* CUarray;
typedef struct CUstream_st* CUstream;

typedef enum CUmemorytype_enum {
  CU_MEMORYTYPE_HOST = 1,
  CU_MEMORYTYPE_DEVICE = 2,
  CU_MEMORYTYPE_ARRAY = 3,
  CU_MEMORYTYPE_UNIFIED = 4
} CUmemorytype;

typedef struct CUDA_MEMCPY2D_st {
  size_t srcXInBytes;
  size_t srcY;
  CUmemorytype srcMemoryType;
  const void* srcHost;
  CUdeviceptr srcDevice;
  CUarray srcArray;
  size_t srcPitch;

  size_t dstXInBytes;
  size_t dstY;
  CUmemorytype dstMemoryType;
  void* dstHost;
  CUdeviceptr dstDevice;
  CUarray dstArray;
  size_t dstPitch;

  size_t WidthInBytes;
  size_t Height;
} CUDA_MEMCPY2D;

CUresult CUDAAPI cuInit(unsigned int flags);
CUresult CUDAAPI cuDriverGetVersion(int* driver_version);
CUresult CUDAAPI cuGetErrorName(CUresult error, const char** name);

CUresult CUDAAPI cuDeviceGetCount(int* count);
CUresult CUDAAPI cuDeviceGet(CUdevice* device, int ordinal);
CUresult CUDAAPI cuDeviceGetName(char* name, int length, CUdevice device);

CUresult CUDAAPI cuCtxCreate(CUcontext* context,
                             unsigned int flags,
                             CUdevice device);
CUresult CUDAAPI cuCtxDestroy(CUcontext context);
CUresult CUDAAPI cuCtxPushCurrent(CUcontext context);
CUresult CUDAAPI cuCtxPopCurrent(CUcontext* context);
CUresult CUDAAPI cuCtxSetCurrent(CUcontext context);
CUresult CUDAAPI cuCtxGetCurrent(CUcontext* context);

CUresult CUDAAPI cuMemAllocPitch(CUdeviceptr* pointer,
                                 size_t* pitch,
                                 size_t width_in_bytes,
                                 size_t height,
                                 unsigned int element_size_bytes);
CUresult CUDAAPI cuMemFree(CUdeviceptr pointer);
CUresult CUDAAPI cuMemcpy2D(const CUDA_MEMCPY2D* copy);
CUresult CUDAAPI cuMemcpy2DUnaligned(const CUDA_MEMCPY2D* copy);
CUresult CUDAAPI cuMemcpy2DAsync(const CUDA_MEMCPY2D* copy, CUstream stream);
CUresult CUDAAPI cuArrayDestroy(CUarray array);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // WEBRTC_SYS_NVIDIA_CUDA_SHIM_CUDA_H
