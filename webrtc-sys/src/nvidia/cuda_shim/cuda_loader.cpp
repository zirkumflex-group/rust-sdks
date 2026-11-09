// Resolves the CUDA driver API declared in cuda_shim/cuda.h from nvcuda.dll
// on first use. See that header for why this replaces linking nvcuda.lib.

#include "cuda.h"

#include <windows.h>

#include <mutex>

namespace {

struct CudaApi {
  bool loaded = false;
  CUresult(CUDAAPI* init)(unsigned int) = nullptr;
  CUresult(CUDAAPI* driver_get_version)(int*) = nullptr;
  CUresult(CUDAAPI* get_error_name)(CUresult, const char**) = nullptr;
  CUresult(CUDAAPI* device_get_count)(int*) = nullptr;
  CUresult(CUDAAPI* device_get)(CUdevice*, int) = nullptr;
  CUresult(CUDAAPI* device_get_name)(char*, int, CUdevice) = nullptr;
  CUresult(CUDAAPI* ctx_create)(CUcontext*, unsigned int, CUdevice) = nullptr;
  CUresult(CUDAAPI* ctx_destroy)(CUcontext) = nullptr;
  CUresult(CUDAAPI* ctx_push_current)(CUcontext) = nullptr;
  CUresult(CUDAAPI* ctx_pop_current)(CUcontext*) = nullptr;
  CUresult(CUDAAPI* ctx_set_current)(CUcontext) = nullptr;
  CUresult(CUDAAPI* ctx_get_current)(CUcontext*) = nullptr;
  CUresult(CUDAAPI* mem_alloc_pitch)(CUdeviceptr*,
                                     size_t*,
                                     size_t,
                                     size_t,
                                     unsigned int) = nullptr;
  CUresult(CUDAAPI* mem_free)(CUdeviceptr) = nullptr;
  CUresult(CUDAAPI* memcpy_2d)(const CUDA_MEMCPY2D*) = nullptr;
  CUresult(CUDAAPI* memcpy_2d_unaligned)(const CUDA_MEMCPY2D*) = nullptr;
  CUresult(CUDAAPI* memcpy_2d_async)(const CUDA_MEMCPY2D*, CUstream) = nullptr;
  CUresult(CUDAAPI* array_destroy)(CUarray) = nullptr;
};

template <typename T>
bool Resolve(HMODULE module, const char* name, T& slot) {
  slot = reinterpret_cast<T>(GetProcAddress(module, name));
  return slot != nullptr;
}

// Loaded once and kept for the life of the process: the function pointers
// below must stay valid however callers load or free nvcuda.dll themselves.
const CudaApi& Api() {
  static CudaApi api;
  static std::once_flag once;
  std::call_once(once, [] {
    HMODULE module = LoadLibraryW(L"nvcuda.dll");
    if (!module) {
      return;
    }
    // The _v2 names are the 64-bit entry points real cuda.h maps these to.
    bool ok = Resolve(module, "cuInit", api.init) &&
              Resolve(module, "cuDriverGetVersion", api.driver_get_version) &&
              Resolve(module, "cuGetErrorName", api.get_error_name) &&
              Resolve(module, "cuDeviceGetCount", api.device_get_count) &&
              Resolve(module, "cuDeviceGet", api.device_get) &&
              Resolve(module, "cuDeviceGetName", api.device_get_name) &&
              Resolve(module, "cuCtxCreate_v2", api.ctx_create) &&
              Resolve(module, "cuCtxDestroy_v2", api.ctx_destroy) &&
              Resolve(module, "cuCtxPushCurrent_v2", api.ctx_push_current) &&
              Resolve(module, "cuCtxPopCurrent_v2", api.ctx_pop_current) &&
              Resolve(module, "cuCtxSetCurrent", api.ctx_set_current) &&
              Resolve(module, "cuCtxGetCurrent", api.ctx_get_current) &&
              Resolve(module, "cuMemAllocPitch_v2", api.mem_alloc_pitch) &&
              Resolve(module, "cuMemFree_v2", api.mem_free) &&
              Resolve(module, "cuMemcpy2D_v2", api.memcpy_2d) &&
              Resolve(module, "cuMemcpy2DUnaligned_v2",
                      api.memcpy_2d_unaligned) &&
              Resolve(module, "cuMemcpy2DAsync_v2", api.memcpy_2d_async) &&
              Resolve(module, "cuArrayDestroy", api.array_destroy);
    if (!ok) {
      api = CudaApi{};
      FreeLibrary(module);
      return;
    }
    api.loaded = true;
  });
  return api;
}

}  // namespace

#define CUDA_SHIM_CALL(field, ...)                                   \
  do {                                                               \
    const CudaApi& api = Api();                                      \
    return api.loaded ? api.field(__VA_ARGS__)                       \
                      : CUDA_ERROR_NOT_INITIALIZED;                  \
  } while (0)

extern "C" {

CUresult CUDAAPI cuInit(unsigned int flags) {
  CUDA_SHIM_CALL(init, flags);
}

CUresult CUDAAPI cuDriverGetVersion(int* driver_version) {
  CUDA_SHIM_CALL(driver_get_version, driver_version);
}

CUresult CUDAAPI cuGetErrorName(CUresult error, const char** name) {
  const CudaApi& api = Api();
  if (api.loaded) {
    return api.get_error_name(error, name);
  }
  if (name) {
    *name = "CUDA_ERROR_NOT_INITIALIZED (nvcuda.dll unavailable)";
  }
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetCount(int* count) {
  CUDA_SHIM_CALL(device_get_count, count);
}

CUresult CUDAAPI cuDeviceGet(CUdevice* device, int ordinal) {
  CUDA_SHIM_CALL(device_get, device, ordinal);
}

CUresult CUDAAPI cuDeviceGetName(char* name, int length, CUdevice device) {
  CUDA_SHIM_CALL(device_get_name, name, length, device);
}

CUresult CUDAAPI cuCtxCreate(CUcontext* context,
                             unsigned int flags,
                             CUdevice device) {
  CUDA_SHIM_CALL(ctx_create, context, flags, device);
}

CUresult CUDAAPI cuCtxDestroy(CUcontext context) {
  CUDA_SHIM_CALL(ctx_destroy, context);
}

CUresult CUDAAPI cuCtxPushCurrent(CUcontext context) {
  CUDA_SHIM_CALL(ctx_push_current, context);
}

CUresult CUDAAPI cuCtxPopCurrent(CUcontext* context) {
  CUDA_SHIM_CALL(ctx_pop_current, context);
}

CUresult CUDAAPI cuCtxSetCurrent(CUcontext context) {
  CUDA_SHIM_CALL(ctx_set_current, context);
}

CUresult CUDAAPI cuCtxGetCurrent(CUcontext* context) {
  CUDA_SHIM_CALL(ctx_get_current, context);
}

CUresult CUDAAPI cuMemAllocPitch(CUdeviceptr* pointer,
                                 size_t* pitch,
                                 size_t width_in_bytes,
                                 size_t height,
                                 unsigned int element_size_bytes) {
  CUDA_SHIM_CALL(mem_alloc_pitch, pointer, pitch, width_in_bytes, height,
                 element_size_bytes);
}

CUresult CUDAAPI cuMemFree(CUdeviceptr pointer) {
  CUDA_SHIM_CALL(mem_free, pointer);
}

CUresult CUDAAPI cuMemcpy2D(const CUDA_MEMCPY2D* copy) {
  CUDA_SHIM_CALL(memcpy_2d, copy);
}

CUresult CUDAAPI cuMemcpy2DUnaligned(const CUDA_MEMCPY2D* copy) {
  CUDA_SHIM_CALL(memcpy_2d_unaligned, copy);
}

CUresult CUDAAPI cuMemcpy2DAsync(const CUDA_MEMCPY2D* copy, CUstream stream) {
  CUDA_SHIM_CALL(memcpy_2d_async, copy, stream);
}

CUresult CUDAAPI cuArrayDestroy(CUarray array) {
  CUDA_SHIM_CALL(array_destroy, array);
}

}  // extern "C"
