/**
 *****************************************************************************
 * Copyright (C) 2021 Kalray
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * @file host.c
 *
 * Main program on Host
 *
 * @author Minh Quan HO <mqho@kalrayinc.com>
 *
 ******************************************************************************
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>
#include <time.h>
#include <CL/cl.h>

#include "ocl_utils.h"

int main(int argc, char* argv[])
{
    int ret = 0;
    // kernel information
    const int kernel_width = 3;
    const int kernel_height = 3;

    // Channels information
    const int channels_out = 32;
    const int channels_in = 32;

    // image information
    const int    image_width  = 112;
    const int    image_height = 112;
    const size_t image_size   = image_width * image_height * sizeof(unsigned char);

    // ===================================================================
    // OpenCL stuffs
    // ===================================================================
    cl_int err = CL_SUCCESS;

    cl_platform_id   platform;         // OpenCL platform
    cl_device_id     device_id;        // device ID
    cl_context       context;          // context
    cl_command_queue queue;            // command queue
    cl_program       program;          // program
    cl_mem           ocl_image_input;  // Device buffer for input image
    size_t           max_workgroup_size; // CL_DEVICE_MAX_WORK_GROUP_SIZE
    cl_uint          max_compute_units;  // CL_DEVICE_MAX_COMPUTE_UNITS

    // ===================================================================
    // Device detection
    // ===================================================================
    err = clGetPlatformIDs(1, &platform, NULL);
    OCL_CHECK_ERROR_QUIT(err, "Failed to clGetPlatformIDs");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 1, &device_id, NULL);
    OCL_CHECK_ERROR_QUIT(err, "Failed to clGetDeviceIDs");

    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS,
                          sizeof(max_compute_units), &max_compute_units, NULL);
    OCL_CHECK_ERROR_QUIT(err, "Failed to clGetDeviceInfo(CL_DEVICE_MAX_COMPUTE_UNITS)");

    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                          sizeof(max_workgroup_size), &max_workgroup_size, NULL);
    OCL_CHECK_ERROR_QUIT(err, "Failed to clGetDeviceInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE)");

    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    assert(context);

    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    assert(queue);

    // ===================================================================
    // List of kernels
    // ===================================================================
    struct kernel_desc_s {
        const char    *name;
        size_t        globalSize[2];
        size_t        localSize[2];
        // additional fields to handle multi-kernels
        cl_kernel     ocl_kernel;
        cl_mem        ocl_image_output;
        unsigned char *host_image_output;
        cl_event      ocl_event[2];
        float         host_elapsed_ms[2];
        float         device_elapsed_ms[2];
        bool          ocl_have_native_kernel;
    } kernel_desc[] = {
        {
            .name       = "convolution",
            .globalSize = {, image_height},// TODO
            .localSize  = {max_workgroup_size, 1},//TODO
        }
    };

    const int nb_kernels = sizeof(kernel_desc) / sizeof(kernel_desc[0]);
    assert(nb_kernels > 0 && "No kernel available");

    // ===================================================================
    // Buffer creation, Program creation & Kernel arguments
    // ===================================================================
    // Create READ-ONLY input image
    ocl_image_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     image_size, img.row_pointers[0], &err);
    OCL_CHECK_ERROR_QUIT(err, "Failed to clCreateBuffer for input image");

    // free allocated buffer for image reading (img.row_pointers[])
    free_img_row_pointers(&img);

    // Create program
    program = ocl_CreateProgramFromBinary(context, device_id, "output/opencl_kernels/convolution.cl.pocl");
    assert(program);

    // From program, create all kernels
    for (int i = 0; i < nb_kernels; i++)
    {
        // create output buffer for each kernel, on Host and Device
        kernel_desc[i].host_image_output = (unsigned char *)malloc(image_size * sizeof(unsigned char));
        assert(kernel_desc[i].host_image_output && "Failed to allocate host_image_output");

        kernel_desc[i].ocl_image_output  = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                                          image_size, NULL, &err);
        OCL_CHECK_ERROR_QUIT(err, "Failed to clCreateBuffer for %d-th output image", i);

        // create kernel
        kernel_desc[i].ocl_kernel = clCreateKernel(program, kernel_desc[i].name, &err);
        OCL_CHECK_ERROR_QUIT(err, "Failed to clCreateKernel %s", kernel_desc[i].name);

        // set arguments
        cl_uint nb_arguments = 0;
        err  = clSetKernelArg(kernel_desc[i].ocl_kernel, nb_arguments++,
                sizeof(cl_mem), &kernel_desc[i].compute);
        err |= clSetKernelArg(kernel_desc[i].ocl_kernel, nb_arguments++,
                sizeof(cl_mem), &kernel_desc[i].A);
        err |= clSetKernelArg(kernel_desc[i].ocl_kernel, nb_arguments++,
                sizeof(cl_mem), &kernel_desc[i].W);
        OCL_CHECK_ERROR_QUIT(err, "Failed to clSetKernelArg %s", kernel_desc[i].name);
    }

    // ===================================================================
    // Run all kernels
    // ===================================================================
    for (int i = 0; i < nb_kernels; i++)
    {
        for (int hot = 0; hot < 2; hot++)
        {
            // time_spent on host
            struct timespec host_start, host_end;
            clock_gettime(CLOCK_MONOTONIC, &host_start);

            err = clEnqueueNDRangeKernel(queue, kernel_desc[i].ocl_kernel, 2, NULL,
                                         kernel_desc[i].globalSize, kernel_desc[i].localSize,
                                         0, NULL, &kernel_desc[i].ocl_event[hot]);
            OCL_CHECK_ERROR_QUIT(err, "Failed to clEnqueueNDRangeKernel %s", kernel_desc[i].name);

            cl_event event_read;
            err = clEnqueueReadBuffer(queue, kernel_desc[i].ocl_image_output, CL_FALSE, 0,
                                      image_size, kernel_desc[i].host_image_output,
                                      1, &kernel_desc[i].ocl_event[hot], &event_read);
            OCL_CHECK_ERROR_QUIT(err, "Failed to clEnqueueReadBuffer %s", kernel_desc[i].name);

            err = clWaitForEvents(1, &event_read);
            OCL_CHECK_ERROR_QUIT(err, "Failed to clWaitForEvents kernel %s", kernel_desc[i].name);

            clock_gettime(CLOCK_MONOTONIC, &host_end);

            err = clReleaseEvent(event_read);
            OCL_CHECK_ERROR_QUIT(err, "Failed to clReleaseEvent kernel %s", kernel_desc[i].name);

            // ----------------------------------------------------------------
            // get profiling info
            // ----------------------------------------------------------------
            // time_spent on Host
            const double host_elapsed_ns = (host_end.tv_sec - host_start.tv_sec) * 1E9 +
                                           (host_end.tv_nsec - host_start.tv_nsec);
            kernel_desc[i].host_elapsed_ms[hot] = (float)(host_elapsed_ns * 1E-6);

            // time_spent on Device
            cl_ulong start = 0;
            cl_ulong end   = 0;

            err = clGetEventProfilingInfo(kernel_desc[i].ocl_event[hot],
                                          CL_PROFILING_COMMAND_START,
                                          sizeof(cl_ulong), &start, NULL);
            OCL_CHECK_ERROR_QUIT(err, "Failed to get CL_PROFILING_COMMAND_START of kernel %s",
                                 kernel_desc[i].name);

            err = clGetEventProfilingInfo(kernel_desc[i].ocl_event[hot],
                                          CL_PROFILING_COMMAND_END,
                                          sizeof(cl_ulong), &end, NULL);
            OCL_CHECK_ERROR_QUIT(err, "Failed to get CL_PROFILING_COMMAND_END of kernel %s",
                                 kernel_desc[i].name);

            kernel_desc[i].device_elapsed_ms[hot] = (double)(end - start) * 1E-06;
        }
    }

    for (int i = 0; i < nb_kernels; i++)
    {
        // TODO correctness check
        if (kernel_desc[i].ocl_have_native_kernel) {
            printf("[HOST] Kernel %19s(): Host cold %6.3f ms hot %6.3f ms"
                   " - Device cold %6.3f ms hot %6.3f ms"
                   " - Speedup vs. Step-0 %5.2f  %s (HAVE_FAST_MATH = %d)\n",
                   kernel_desc[i].name,
                   kernel_desc[i].host_elapsed_ms[0], kernel_desc[i].host_elapsed_ms[1],
                   kernel_desc[i].device_elapsed_ms[0], kernel_desc[i].device_elapsed_ms[1],
                   (kernel_desc[0].device_elapsed_ms[1] / kernel_desc[i].device_elapsed_ms[1]),
                   passed ? "[PASSED]" : "[FAILED]", HAVE_FAST_MATH);
        } else {
            printf("[HOST] Kernel %19s(): Host cold %6.3f ms hot %6.3f ms"
                   " - Device cold %6.3f ms hot %6.3f ms"
                   " - Speedup vs. Step-0 %5.2f  %s\n",
                   kernel_desc[i].name,
                   kernel_desc[i].host_elapsed_ms[0], kernel_desc[i].host_elapsed_ms[1],
                   kernel_desc[i].device_elapsed_ms[0], kernel_desc[i].device_elapsed_ms[1],
                   (kernel_desc[0].device_elapsed_ms[1] / kernel_desc[i].device_elapsed_ms[1]),
                   passed ? "[PASSED]" : "[FAILED]");
        }

        ret |= !passed;
    }

    // ===================================================================
    // Cleanup
    // ===================================================================
    clReleaseMemObject(ocl_image_input);

    for (int i = 0; i < nb_kernels; i++)
    {
        clReleaseMemObject(kernel_desc[i].ocl_image_output);
        clReleaseKernel(kernel_desc[i].ocl_kernel);
        clReleaseEvent(kernel_desc[i].ocl_event[0]);
        clReleaseEvent(kernel_desc[i].ocl_event[1]);
        free(kernel_desc[i].host_image_output);
    }

    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseDevice(device_id);

quit:
    return ret;
}
