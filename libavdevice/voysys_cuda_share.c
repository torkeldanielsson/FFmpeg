/*
 * Copyright (c) 2020 Voysys AB / Torkel Danielsson <torkel@voysys.com>
 * 
 * This file is MIT-licensed, to allow for modification and integration.
 * Note: the MIT license applies to this specific file ONLY.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/**
 * @file
 * Input device for Voysys Cuda Share
 * 
 * Inspiration from ffmpeg source code and various online posts:
 * @see https://stackoverflow.com/questions/49862610/opengl-to-ffmpeg-encode
 */

#include <time.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
#include <time.h>

#include "libavutil/internal.h"
#include "libavutil/hwcontext_cuda.h"
#include "libavutil/log.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/time.h"
#include "libavutil/parseutils.h"
#include "libavutil/pixdesc.h"
#include "libavformat/internal.h"
#include "avdevice.h"

#define VOYSYS_CUDA_OUTPUT_DATA_NAME ("/voysys_cuda_output_data")
#define VOYSYS_CUDA_OUTPUT_MUTEX_NAME ("/voysys_cuda_output_mutex")
#define VOYSYS_CUDA_OUTPUT_CONDVAR_NAME ("/voysys_cuda_output_condition_variable")

typedef struct VoysysCudaOutput {
    uint8_t cuda_ipc_mem_handle[64];
    int32_t has_new_data;
    int32_t width;
    int32_t height;
    int32_t framerate_num;
    int32_t framerate_den;
} VoysysCudaOutput;

typedef struct VoysysContext {
    AVClass * class;          // "class for private options" - is this required to be here?

    VoysysCudaOutput * voysys_cuda_output_shm;
    int voysys_cuda_output_shm_fd;

    pthread_mutex_t * voysys_cuda_output_mutex;
    int voysys_cuda_output_mutex_fd;

    pthread_cond_t * new_frame_condition_variable;
    int new_frame_condition_variable_fd;

    AVBufferRef * cuda_av_hardware_device;
    CUcontext cuda_context;
    AVBufferRef * cuda_hw_av_frame_ctx;
} VoysysContext;

static av_cold int voysys_read_header(AVFormatContext * avctx) 
{
    VoysysContext * p = avctx->priv_data;
    AVStream * st = NULL;
    int ret = -1;
    int width = 0;
    int height = 0;
    AVRational framerate = { 0 };

    p->voysys_cuda_output_shm_fd = shm_open(VOYSYS_CUDA_OUTPUT_DATA_NAME, O_RDWR, 0660);
    if (p->voysys_cuda_output_shm_fd == -1) {
        ret = AVERROR(errno);
        av_log(avctx, AV_LOG_ERROR, "Voysys output not available, start Voysys first (failed to open %s: %s)\n", VOYSYS_CUDA_OUTPUT_DATA_NAME, av_err2str(ret));
        return ret;
    }

    p->voysys_cuda_output_shm = (VoysysCudaOutput *)(mmap(NULL, sizeof(VoysysCudaOutput), PROT_READ, MAP_SHARED, p->voysys_cuda_output_shm_fd, 0));
    if (p->voysys_cuda_output_shm == NULL || p->voysys_cuda_output_shm == MAP_FAILED) {
        ret = AVERROR(errno);
        av_log(avctx, AV_LOG_ERROR, "Failed to map shmem %s (%s)\n", VOYSYS_CUDA_OUTPUT_DATA_NAME, av_err2str(ret));
        return ret;
    }

    p->voysys_cuda_output_mutex_fd = shm_open(VOYSYS_CUDA_OUTPUT_MUTEX_NAME, O_RDWR, 0660);
    if (p->voysys_cuda_output_mutex_fd == -1) {
        ret = AVERROR(errno);
        av_log(avctx, AV_LOG_ERROR, "Failed to open shmem %s (%s)\n", VOYSYS_CUDA_OUTPUT_MUTEX_NAME, av_err2str(ret));
        return ret;
    }

    p->voysys_cuda_output_mutex = (pthread_mutex_t *)(mmap(NULL, sizeof(pthread_mutex_t), PROT_READ, MAP_SHARED, p->voysys_cuda_output_mutex_fd, 0));
    if (p->voysys_cuda_output_mutex == NULL || p->voysys_cuda_output_mutex == MAP_FAILED) {
        ret = AVERROR(errno);
        av_log(avctx, AV_LOG_ERROR, "Failed to map shmem %s (%s)\n", VOYSYS_CUDA_OUTPUT_MUTEX_NAME, av_err2str(ret));
        return ret;
    }


    p->new_frame_condition_variable_fd = shm_open(VOYSYS_CUDA_OUTPUT_CONDVAR_NAME, O_RDWR, 0660);
    if (p->new_frame_condition_variable_fd == -1) {
        ret = AVERROR(errno);
        av_log(avctx, AV_LOG_ERROR, "Failed to open shmem %s (%s)\n", VOYSYS_CUDA_OUTPUT_CONDVAR_NAME, av_err2str(ret));
        return ret;
    }

    p->new_frame_condition_variable = (pthread_cond_t *)(mmap(NULL, sizeof(pthread_cond_t), PROT_READ, MAP_SHARED, p->new_frame_condition_variable_fd, 0));
    if (p->new_frame_condition_variable == NULL || p->new_frame_condition_variable == MAP_FAILED) {
        ret = AVERROR(errno);
        av_log(avctx, AV_LOG_ERROR, "Failed to map shmem %s (%s)\n", VOYSYS_CUDA_OUTPUT_CONDVAR_NAME, av_err2str(ret));
        return ret;
    }

    {
        pthread_mutex_lock(p->voysys_cuda_output_mutex);
        width = p->voysys_cuda_output_shm->width;
        height = p->voysys_cuda_output_shm->height;
        framerate.num = p->voysys_cuda_output_shm->framerate_num;
        framerate.den = p->voysys_cuda_output_shm->framerate_den;
        pthread_mutex_unlock(p->voysys_cuda_output_mutex);
    }

    if (width <= 0 || height <= 0) {
        av_log(avctx, AV_LOG_ERROR, "Invalid resolution - is Voysys started correctly?\n");
        return AVERROR_EXTERNAL;
    }

    // use ffmpeg to allocate a cuda context
    ret = av_hwdevice_ctx_create(&p->cuda_av_hardware_device, AV_HWDEVICE_TYPE_CUDA, NULL, NULL, NULL);
    if (ret < 0) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create cuda context (%s)\n", av_err2str(ret));
        return ret;
    } 

    {
        // extract the cuda device
        AVHWDeviceContext * hwDevContext = (AVHWDeviceContext*)(p->cuda_av_hardware_device->data);
        AVCUDADeviceContext * cudaDevCtx = (AVCUDADeviceContext*)(hwDevContext->hwctx);
        p->cuda_context = &cudaDevCtx->cuda_ctx;
    }

    // allocate a cuda hw frame context (we will copy from the cuda ipc shared memory to this)
    p->cuda_hw_av_frame_ctx = av_hwframe_ctx_alloc(p->cuda_context);

    // configure and initialize the cuda hw frame context
    {
        AVHWFramesContext * frameCtxPtr = (AVHWFramesContext*)(p->cuda_hw_av_frame_ctx->data);
        frameCtxPtr->width = width;
        frameCtxPtr->height = height;
        frameCtxPtr->sw_format = AV_PIX_FMT_YUV420P; // There are only certain supported types here, we need to conform to these types
        frameCtxPtr->format = AV_PIX_FMT_CUDA;
        frameCtxPtr->device_ref = p->cuda_av_hardware_device;
        frameCtxPtr->device_ctx = (AVHWDeviceContext*)(p->cuda_av_hardware_device->data);

        ret = av_hwframe_ctx_init(p->cuda_hw_av_frame_ctx);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "Failed to initialize the cuda hw frame context (%s)\n", av_err2str(ret));
            return ret;
        }
    }

    st->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    st->codecpar->codec_id   = AV_CODEC_ID_RAWVIDEO;
    st->codecpar->width      = width;
    st->codecpar->height     = height;
    st->codecpar->format     = AV_PIX_FMT_CUDA;
    st->avg_frame_rate       = framerate;
    st->codecpar->bit_rate   = width * height * 3 * av_q2d(framerate) * 8;

    av_log(avctx, AV_LOG_INFO,
           "w:%d h:%d fps:%d/%d bit_rate:%"PRId64"\n",
           width, height,
           framerate.num, framerate.den,
           st->codecpar->bit_rate);
    return 0;
}

static int voysys_read_packet(AVFormatContext *avctx, AVPacket *pkt)
{
    VoysysContext * p = avctx->priv_data;
    int timeout = 0;
    int got_frame = 0;
    CUresult cuRes = CUDA_ERROR_UNKNOWN;
    CUcontext oldCudaCtx = NULL;
    CUdeviceptr cudaDevicePtr = NULL;
    CUDA_MEMCPY2D cudaMemCpyStruct = { 0 };
    int avRes = -1;
    int pthreadRes = -1;
    CUipcMemHandle ipcHandle = { 0 };
    AVFrame frame = { 0 };

    {
        pthread_mutex_lock(p->voysys_cuda_output_mutex);

        struct timespec ts = { 0 };
        {
            clock_gettime(CLOCK_REALTIME, &ts);

            ts.tv_nsec += 100'000'000;

            if (ts.tv_nsec >= 1'000'000'000) {
                ts.tv_sec++;
                ts.tv_nsec -= 1'000'000'000;
            }
        }

        do {
            pthreadRes = pthread_cond_timedwait(p->new_frame_condition_variable, p->voysys_cuda_output_mutex, &ts);

            if (pthreadRes != 0 && pthreadRes != ETIMEDOUT) {
                av_log(avctx, AV_LOG_ERROR, "pthread_cond_timedwait error (%d)\n", pthreadRes);
                pthread_mutex_unlock(p->voysys_cuda_output_mutex);
                return AVERROR_EXTERNAL;
            }

            if (pthreadRes == ETIMEDOUT) {
                timeout = 1;
            }

            if (pthreadRes == 0) {
                got_frame = p->voysys_cuda_output_shm->has_new_data;
            }

        } while (got_frame == 0 && timeout == 0);

        if (got_frame == 1) {
            p->voysys_cuda_output_shm->has_new_data = 0;

            assert(sizeof(CUipcMemHandle) == sizeof(p->voysys_cuda_output_shm->cuda_ipc_mem_handle));
            memcpy(&ipcHandle, &p->voysys_cuda_output_shm->cuda_ipc_mem_handle, sizeof(CUipcMemHandle));

            cuRes = cuIpcOpenMemHandle(&cudaDevicePtr, ipcHandle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
            if (cuRes != CUDA_SUCCESS) {
                av_log(avctx, AV_LOG_ERROR, "cuIpcOpenMemHandle error (%d)\n", cuRes);
                pthread_mutex_unlock(p->voysys_cuda_output_mutex);
                return AVERROR_EXTERNAL;
            }

            // copy the mapped memory to our local memory (which has been allocated using ffmpeg)

            // it's ok for this call to fail
            cuRes = cuCtxPopCurrent(&oldCudaCtx);

            cuRes = cuCtxPushCurrent(p->cuda_context);
            if (cuRes != CUDA_SUCCESS) {
                av_log(avctx, AV_LOG_ERROR, "failed to push cuda context (%d)\n", cuRes);
                pthread_mutex_unlock(p->voysys_cuda_output_mutex);
                return AVERROR_EXTERNAL;
            }

            avRes = av_hwframe_get_buffer(p->cuda_hw_av_frame_ctx, &frame, 0); 
            if (avRes < 0) {
                av_log(avctx, AV_LOG_ERROR, "Failed to allocate hw frame (%s)\n", av_err2str(avRes));
                pthread_mutex_unlock(p->voysys_cuda_output_mutex);
                return avRes;
            }

            //Setup for memcopy
            cudaMemCpyStruct.srcXInBytes = 0;
            cudaMemCpyStruct.srcY = 0;
            cudaMemCpyStruct.srcMemoryType = CU_MEMORYTYPE_ARRAY;
            cudaMemCpyStruct.srcArray = cudaDevicePtr;

            cudaMemCpyStruct.dstXInBytes = 0;
            cudaMemCpyStruct.dstY = 0;
            cudaMemCpyStruct.dstMemoryType = CU_MEMORYTYPE_DEVICE;

            cudaMemCpyStruct.dstDevice = cudaDevicePtr;
            cudaMemCpyStruct.dstPitch = frame.linesize[0];
            cudaMemCpyStruct.WidthInBytes = frame.width * 4; // bytes per pixel == 4?
            cudaMemCpyStruct.Height = frame.height;

            cuRes = cuMemcpy2D(&cudaMemCpyStruct); 
            if (cuRes != CUDA_SUCCESS) {
                av_log(avctx, AV_LOG_ERROR, "failed to copy from shared to av cuda memory (%d)\n", cuRes);
                pthread_mutex_unlock(p->voysys_cuda_output_mutex);
                return AVERROR_EXTERNAL;
            }

            // it's ok for this call to fail
            cuRes = cuCtxPopCurrent(&oldCudaCtx);

            cuRes = cuIpcCloseMemHandle(&cudaDevicePtr);
            if (cuRes != CUDA_SUCCESS) {
                av_log(avctx, AV_LOG_ERROR, "cuIpcCloseMemHandle error (%d)\n", cuRes);
                pthread_mutex_unlock(p->voysys_cuda_output_mutex);
                return AVERROR_EXTERNAL;
            }
        }

        pthread_mutex_unlock(p->voysys_cuda_output_mutex);
    }

    // At this point we should have a valid cuda frame that is fully owned by us,
    // but how can we pass this along into the rest of ffmpeg?
    // It seems that the "AVPacket" abstraction is only for cpu-side data?
    // I would like to do something like this:
    // avRes = avcodec_send_frame(avCodecContext, &frame); 

    av_frame_unref(&frame);

    return 0;
}

static av_cold int voysys_read_close(AVFormatContext * avctx)
{
    VoysysContext * p = avctx->priv_data;

    munmap(p->voysys_cuda_output_shm, sizeof(VoysysCudaOutput));
    close(p->voysys_cuda_output_shm_fd);

    munmap(p->voysys_cuda_output_mutex, sizeof(pthread_mutex_t));
    close(p->voysys_cuda_output_mutex_fd);

    munmap(p->new_frame_condition_variable, sizeof(pthread_cond_t));
    close(p->new_frame_condition_variable_fd);

    return 0;
}

static const AVOption options[] = {
    { NULL },
};

static const AVClass voysys_class = {
    .class_name = "voysys indev",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
    .category   = AV_CLASS_CATEGORY_DEVICE_VIDEO_INPUT,
};

AVInputFormat ff_voysys_cuda_share_demuxer = {
    .name           = "voysys",
    .long_name      = NULL_IF_CONFIG_SMALL("Voysys cuda share"),
    .priv_data_size = sizeof(VoysysContext),
    .read_header    = voysys_read_header,
    .read_packet    = voysys_read_packet,
    .read_close     = voysys_read_close,
    .flags          = AVFMT_NOFILE,
    .priv_class     = &voysys_class,
};
