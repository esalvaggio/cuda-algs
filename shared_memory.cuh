#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_



#include <mxnet/base.h>

namespace mxnet
{
namespace op
{



#define TILE_WIDTH 16



// k = W ???
__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int n, m, h0, w0, h_base, w_base, h, w;
    int X_tile_width = TILE_WIDTH + K - 1;
    //extern __shared__ float shmem[2 * TILE_WIDTH * TILE_WIDTH];
    extern __shared__ float shmem[];
    float * X_shared = &shmem[0];
    float * W_shared = &shmem[X_tile_width * X_tile_width];
    n = blockIdx.x;
    m = blockIdx.y;
    h0 = threadIdx.y;
    w0 = threadIdx.x;
    int W_grid = ceil(W_out / (TILE_WIDTH * 1.0));
    //h_base = (blockIdx.z/W_grid) * TILE_WIDTH;
    //w_base = (blockIdx.z % W_grid) * TILE_WIDTH;
    h_base = (blockIdx.z/W_grid) * TILE_WIDTH;
    w_base = (blockIdx.z % W_grid) * TILE_WIDTH;
    h = h_base + h0;
    w = w_base + w0;


    float acc = 0.0;
    for(int c = 0; c < C; c++)
    {
      if((h0 < K) && (w0 < K) && (m < M))
      {
        W_shared[h0 * K + w0] = k4d(m,c,h0,w0);
      }
      __syncthreads();

      //printf("Before loop \n");
      for(int i = h; i < h_base + X_tile_width; i += TILE_WIDTH)
      {
        for(int j = w; j < w_base + X_tile_width; j += TILE_WIDTH)
        {
          //if((h < H_out) && (w < W_out) && (n < B) && (m < M)) {

          if(i < H && j < W)
          {
            X_shared[(i - h_base)* X_tile_width + (j - w_base)] = x4d(n,c,i,j);
          }

        //  }
        }
      }
      __syncthreads();
      for(int p = 0; p < K; p++)
      {
        for(int q = 0; q < K; q++)
        {
          if( ((h0 + p)< X_tile_width) && ( (w0 +p) < X_tile_width)) {
            acc = acc + X_shared[(h0 + p) * X_tile_width + (w0 + q)] * W_shared[p * K + q];
          }
        }
      }

      //printf("End of last loop");
      __syncthreads();

    }

    //printf("Successfully finished calculations\n");
    if((n < B) && (m < M) && (h < H_out) && (w < W_out)) {
      y4d(n,m,h,w) = acc;
    }


  //  printf("All done \n");


    // int W_grid = ceil(W_out * 1.0 / TILE_WIDTH);
    // int n, m , h, w, c, p, q;
    // n = blockIdx.x;
    // m = blockIdx.y;
    // h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
    // w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;
    // float acc = 0.0;
    // for(c = 0; c < C; c++)
    // {
    //   for(p = 0; p < K; p++)
    //   {
    //     for(q = 0; q < K; q++)
    //     {
    //       if((h < H_out) && (w < W_out) && (n < B) && (m < M))
    //       {
    //       acc = acc + x4d(n,c,h+p, w+q) * k4d(m,c,p,q);
    //       }
    //     }
    //   }
    // }
    // if((h < H_out) && (w < W_out) && (n < B) && (m < M)) {
    //   y4d(n,m,h,w) = acc;
    //}



    #undef y4d
    #undef x4d
    #undef k4d
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
//    CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...

    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int W_grid = ceil(W_out / (TILE_WIDTH * 1.0));
    int H_grid = ceil(H_out / (TILE_WIDTH * 1.0));
    int Z = W_grid * H_grid;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, Z);

    size_t shmem_size = sizeof(float) * ( (TILE_WIDTH + K-1)*(TILE_WIDTH + K-1) + K*K );

    forward_kernel<<<gridDim, blockDim, shmem_size>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);



    // Set the kernel dimensions
    // dim3 gridDim(0);
    // dim3 blockDim(0);

    // Call the kernel
    // forward_kernel<<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}




/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
