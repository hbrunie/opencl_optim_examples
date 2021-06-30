/* Convolution kernel obtained with TVM on x,y,f,c,w,h 112,112,32,32,3,3
 * thanks to func.imported_modules[0].save("kernel_mppa.cl") call in python
 * See conv_mppa_fast.py in mppa branch, tvm_ttile
 */
__kernel void convolution_kernel(__global float* restrict compute, __global float* restrict A, __global float* restrict W) {
  for (int out_channels_outer = 0; out_channels_outer < 4; ++out_channels_outer) {
      vstore8(((float8)(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f)), 0, compute + (((((((int)get_group_id(0)) * 14336) + (((int)get_local_id(0)) * 3584)) + (((int)get_group_id(1)) * 128)) + (((int)get_local_id(1)) * 32)) + (out_channels_outer * 8)));
    for (int axe_in_channels = 0; axe_in_channels < 32; ++axe_in_channels) {
      for (int axe_kernel_h = 0; axe_kernel_h < 3; ++axe_kernel_h) {
        for (int axe_kernel_w = 0; axe_kernel_w < 3; ++axe_kernel_w) {
            vstore8((vload8(0, compute + (((((((int)get_group_id(0)) * 14336) + (((int)get_local_id(0)) * 3584)) + (((int)get_group_id(1)) * 128)) + (((int)get_local_id(1)) * 32)) + (out_channels_outer * 8))) + (((float8)(A[((((((((((int)get_group_id(0)) * 14592) + (((int)get_local_id(0)) * 3648)) + (axe_kernel_w * 3648)) + (((int)get_group_id(1)) * 128)) + (((int)get_local_id(1)) * 32)) + (axe_kernel_h * 32)) + axe_in_channels))], A[((((((((((int)get_group_id(0)) * 14592) + (((int)get_local_id(0)) * 3648)) + (axe_kernel_w * 3648)) + (((int)get_group_id(1)) * 128)) + (((int)get_local_id(1)) * 32)) + (axe_kernel_h * 32)) + axe_in_channels))], A[((((((((((int)get_group_id(0)) * 14592) + (((int)get_local_id(0)) * 3648)) + (axe_kernel_w * 3648)) + (((int)get_group_id(1)) * 128)) + (((int)get_local_id(1)) * 32)) + (axe_kernel_h * 32)) + axe_in_channels))], A[((((((((((int)get_group_id(0)) * 14592) + (((int)get_local_id(0)) * 3648)) + (axe_kernel_w * 3648)) + (((int)get_group_id(1)) * 128)) + (((int)get_local_id(1)) * 32)) + (axe_kernel_h * 32)) + axe_in_channels))], A[((((((((((int)get_group_id(0)) * 14592) + (((int)get_local_id(0)) * 3648)) + (axe_kernel_w * 3648)) + (((int)get_group_id(1)) * 128)) + (((int)get_local_id(1)) * 32)) + (axe_kernel_h * 32)) + axe_in_channels))], A[((((((((((int)get_group_id(0)) * 14592) + (((int)get_local_id(0)) * 3648)) + (axe_kernel_w * 3648)) + (((int)get_group_id(1)) * 128)) + (((int)get_local_id(1)) * 32)) + (axe_kernel_h * 32)) + axe_in_channels))], A[((((((((((int)get_group_id(0)) * 14592) + (((int)get_local_id(0)) * 3648)) + (axe_kernel_w * 3648)) + (((int)get_group_id(1)) * 128)) + (((int)get_local_id(1)) * 32)) + (axe_kernel_h * 32)) + axe_in_channels))], A[((((((((((int)get_group_id(0)) * 14592) + (((int)get_local_id(0)) * 3648)) + (axe_kernel_w * 3648)) + (((int)get_group_id(1)) * 128)) + (((int)get_local_id(1)) * 32)) + (axe_kernel_h * 32)) + axe_in_channels))])) * vload8(0, W + ((((axe_kernel_w * 3072) + (axe_kernel_h * 1024)) + (axe_in_channels * 32)) + (out_channels_outer * 8))))), 0, compute + (((((((int)get_group_id(0)) * 14336) + (((int)get_local_id(0)) * 3584)) + (((int)get_group_id(1)) * 128)) + (((int)get_local_id(1)) * 32)) + (out_channels_outer * 8)));
        }
      }
    }
  }
}
