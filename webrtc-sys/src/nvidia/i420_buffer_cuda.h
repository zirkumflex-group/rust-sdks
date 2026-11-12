#ifndef WEBRTC_NVIDIA_I420_BUFFER_CUDA_H_
#define WEBRTC_NVIDIA_I420_BUFFER_CUDA_H_

#include <cuda.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "NvEncoder/NvEncoderCuda.h"
#include "api/video/i420_buffer.h"
#include "api/video/video_frame_buffer.h"
#include "third_party/libyuv/include/libyuv/planar_functions.h"

namespace webrtc {

inline void CopyHostPlaneToDevice(CUdeviceptr dst,
                                  uint32_t dst_pitch,
                                  const uint8_t* src,
                                  uint32_t src_pitch,
                                  uint32_t width_bytes,
                                  uint32_t height) {
  CUDA_MEMCPY2D copy = {0};
  copy.srcMemoryType = CU_MEMORYTYPE_HOST;
  copy.srcHost = src;
  copy.srcPitch = src_pitch;
  copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  copy.dstDevice = dst;
  copy.dstPitch = dst_pitch;
  copy.WidthInBytes = width_bytes;
  copy.Height = height;
  CUDA_DRVAPI_CALL(cuMemcpy2D(&copy));
}

inline void CopyI420BufferToDeviceFrame(CUcontext context,
                                        const I420BufferInterface& buffer,
                                        const NvEncInputFrame& input_frame) {
  if (input_frame.bufferFormat != NV_ENC_BUFFER_FORMAT_IYUV ||
      input_frame.numChromaPlanes != 2) {
    NVENC_THROW_ERROR("NVENC I420 upload requires an IYUV input frame",
                      NV_ENC_ERR_INVALID_PARAM);
  }

  CUDA_DRVAPI_CALL(cuCtxPushCurrent(context));

  const CUdeviceptr dst_y = reinterpret_cast<CUdeviceptr>(input_frame.inputPtr);
  const CUdeviceptr dst_u = dst_y + input_frame.chromaOffsets[0];
  const CUdeviceptr dst_v = dst_y + input_frame.chromaOffsets[1];

  CopyHostPlaneToDevice(dst_y, input_frame.pitch, buffer.DataY(),
                        buffer.StrideY(), buffer.width(), buffer.height());
  CopyHostPlaneToDevice(dst_u, input_frame.chromaPitch, buffer.DataU(),
                        buffer.StrideU(), buffer.ChromaWidth(),
                        buffer.ChromaHeight());
  CopyHostPlaneToDevice(dst_v, input_frame.chromaPitch, buffer.DataV(),
                        buffer.StrideV(), buffer.ChromaWidth(),
                        buffer.ChromaHeight());

  CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));
}

// Uploads `buffer` to an NVENC input frame of either layout; false when it can't be read.
//
// An NV12 input frame takes an NV12 buffer as it is: two plane copies to the device, with
// nothing converted on the CPU. Any other buffer goes through I420, its chroma interleaved
// into `scratch` (kept by the caller, so it is allocated once). An IYUV input frame takes
// I420 as before.
inline bool CopyFrameBufferToDeviceFrame(CUcontext context,
                                         VideoFrameBuffer& buffer,
                                         const NvEncInputFrame& input_frame,
                                         std::vector<uint8_t>& scratch) {
  if (input_frame.bufferFormat == NV_ENC_BUFFER_FORMAT_IYUV) {
    webrtc::scoped_refptr<I420BufferInterface> i420 = buffer.ToI420();
    if (!i420) {
      return false;
    }
    CopyI420BufferToDeviceFrame(context, *i420, input_frame);
    return true;
  }
  if (input_frame.bufferFormat != NV_ENC_BUFFER_FORMAT_NV12 ||
      input_frame.numChromaPlanes != 1) {
    NVENC_THROW_ERROR("NVENC upload requires an NV12 or IYUV input frame",
                      NV_ENC_ERR_INVALID_PARAM);
  }

  const uint32_t width = static_cast<uint32_t>(buffer.width());
  const uint32_t height = static_cast<uint32_t>(buffer.height());
  const uint32_t chroma_width = (width + 1) / 2;
  const uint32_t chroma_height = (height + 1) / 2;
  // A row of interleaved chroma: U and V for each pair of columns.
  const uint32_t chroma_row = chroma_width * 2;

  const uint8_t* y = nullptr;
  const uint8_t* uv = nullptr;
  uint32_t stride_y = 0;
  uint32_t stride_uv = 0;
  // Keeps a converted frame alive until it is uploaded.
  webrtc::scoped_refptr<I420BufferInterface> i420;
  if (buffer.type() == VideoFrameBuffer::Type::kNV12) {
    const NV12BufferInterface* nv12 = buffer.GetNV12();
    y = nv12->DataY();
    stride_y = nv12->StrideY();
    uv = nv12->DataUV();
    stride_uv = nv12->StrideUV();
  } else {
    i420 = buffer.ToI420();
    if (!i420) {
      return false;
    }
    scratch.resize(static_cast<size_t>(chroma_row) * chroma_height);
    libyuv::MergeUVPlane(i420->DataU(), i420->StrideU(), i420->DataV(),
                         i420->StrideV(), scratch.data(), chroma_row,
                         chroma_width, chroma_height);
    y = i420->DataY();
    stride_y = i420->StrideY();
    uv = scratch.data();
    stride_uv = chroma_row;
  }

  CUDA_DRVAPI_CALL(cuCtxPushCurrent(context));

  const CUdeviceptr dst_y = reinterpret_cast<CUdeviceptr>(input_frame.inputPtr);
  const CUdeviceptr dst_uv = dst_y + input_frame.chromaOffsets[0];
  CopyHostPlaneToDevice(dst_y, input_frame.pitch, y, stride_y,
                        std::min(width, input_frame.pitch), height);
  CopyHostPlaneToDevice(dst_uv, input_frame.chromaPitch, uv, stride_uv,
                        std::min(chroma_row, input_frame.chromaPitch),
                        chroma_height);

  CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));
  return true;
}

}  // namespace webrtc

#endif  // WEBRTC_NVIDIA_I420_BUFFER_CUDA_H_
