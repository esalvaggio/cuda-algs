#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_



#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

#define TILE_WIDTH 16
// k = W ???

__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
    __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];

    int blockX = blockIdx.x;
    int blockY = blockIdx.y;
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;
    int row = blockY * blockDim.y + threadY;
    int col = blockX * blockDim.x + threadX;
    subTileA[threadY][threadX] = 0.0;
    subTileB[threadY][threadX] = 0.0;

    float result = 0;
    int numTiles = ((numAColumns -1)/TILE_WIDTH) + 1;
    int a;
    for(a = 0; a < numTiles; a++)
    {
        if((row < numARows) && ((a * TILE_WIDTH + threadX) < numAColumns))
        {
            subTileA[threadY][threadX] = A[(row * numAColumns) + (a * TILE_WIDTH) +threadX];
        }
        else
        {
            subTileA[threadY][threadX] = 0.0;
        }

        if(((a * TILE_WIDTH + threadY) < numBRows) && (col < numBColumns))
        {
            subTileB[threadY][threadX] = B[(a * TILE_WIDTH + threadY) * numBColumns + col];
        }
        else
        {
            subTileB[threadY][threadX] = 0.0;
        }
        __syncthreads();
        if(row < numCRows && col < numCColumns)
        {
            for(int b = 0; b < TILE_WIDTH; b++)
            {
              result = result + subTileA[threadY][b] * subTileB[b][threadX];
            }
        }
      __syncthreads();
    }

    if(row < numCRows && col < numCColumns)
    {
        C[row * numCColumns + col] = result;
    }
}

__global__ void unroll_Kernel(int n, int C, int H, int W, int K, float* X, float* X_unroll) {
  #define x4d(i3, i2, i1, i0) X[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  int c, s, h_out, w_out, h_unroll, w_unroll, w_base, p, q;
  int t = blockIdx.x * 1024 + threadIdx.x;
  int H_out = H - K + 1;
  int W_out = W - K + 1;
  int W_unroll = H_out * W_out;

  if(t < (C * W_unroll)) {
    c = t / W_unroll;
    s = t % W_unroll;
    h_out = s / W_out;
    w_out = s % W_out;
    h_unroll = h_out * W_out + w_out;
    w_base = c * K * K;

    for(p = 0; p < K; p++) {
      for(q = 0; q < K; q++) {
        w_unroll = w_base + p * K + q;
        X_unroll[w_unroll * W_unroll + h_unroll] = x4d(n,c, h_out + p, w_out + q);
      }
    }
   }
    #undef x4d
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w){

    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    float * X_ptr = x.dptr_;
    float * W_ptr = w.dptr_;

    int W_unroll = H_out * W_out; 
    int H_unroll = C * K * K;

    int num_threads = C * H_out * W_out;
    int num_blocks = ceil(1.0 * num_threads / 1024);

    dim3 dimGrid(ceil((1.0 * W_unroll) / TILE_WIDTH), ceil((1.0 * M) / TILE_WIDTH), 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    float *X_unroll;
    cudaMalloc((void**) &X_unroll, W_unroll * H_unroll * sizeof(float));
    for(int b = 0; b < B; b++) {
        unroll_Kernel<<<num_blocks, 1024>>>(b, C, H, W, K, X_ptr, X_unroll);
        matrixMultiplyShared<<<dimGrid, dimBlock>>>(W_ptr, X_unroll, &y.dptr_[b*M*H_out*W_out], M, H_unroll, H_unroll, W_unroll, M, W_unroll);
    }
    cudaFree(X_unroll);

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
