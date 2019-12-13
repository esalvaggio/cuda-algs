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
__global__ void unroll_Kernel(int B, int M, int C, int H, int W, int K, float* X, float* X_unroll, float * W_ptr, float * Y_ptr)
{
#define y4d(i3, i2, i1, i0) Y_ptr[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) X[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) W_ptr[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]


__shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
__shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];

const int H_out = H - K + 1;
const int W_out = W - K + 1;
int blockX = blockIdx.x;
int blockY = blockIdx.y;
int blockZ = blockIdx.z;
int threadX = threadIdx.x;
int threadY = threadIdx.y;
int row = blockY * TILE_WIDTH + threadY;
int col = blockX * TILE_WIDTH + threadX;
int numAColumns = C * K * K;

float acc = 0.0;

int iterations = ceil(numAColumns * 1.0 / (TILE_WIDTH));
int i;

for(i = 0; i < iterations; i++)
{
  int temp_col = i * TILE_WIDTH + threadX;
  int temp_row = i * TILE_WIDTH + threadY;
  subTileA[threadY][threadX] = 0.0;
  subTileB[threadY][threadX] = 0.0;

  int weight_m = row;
  int weight_c = temp_col/(K*K);
  int weight_h = (temp_col % (K*K) )/ K;
  int weight_w = (temp_col % (K*K)) % K;

  if(temp_col < numAColumns && row < M)
  {
    subTileA[threadY][threadX] = k4d(weight_m, weight_c, weight_h, weight_w);
  }
  else
  {
    subTileA[threadY][threadX] = 0.0;
  }

  int input_b = blockZ;
  int input_c = temp_row/(K*K);
  int input_p = (temp_row % (K*K)) / K;
  int input_h = col / W_out;
  int input_q = (temp_row % (K*K)) % K;
  int input_w = (col % W_out);


  if(temp_row < numAColumns && col < H_out * W_out)
  {
    subTileB[threadY][threadX] = x4d(input_b, input_c, input_h + input_p, input_w + input_q);
  }
  else
  {
    subTileB[threadY][threadX] = 0.0;
  }

  __syncthreads();


  for(int j = 0; j < TILE_WIDTH; j++)
  {
    acc = acc + subTileA[threadY][j] * subTileB[j][threadX];
  }
  __syncthreads();

}

int output_b = blockZ;
int output_m = row;
int output_h = col / W_out;
int output_w = col % W_out;

if(row < M && col < W_out * H_out)
{
  y4d(output_b, output_m, output_h, output_w) = acc;
}



//}
    #undef x4d
    #undef y4d
    #undef k4d
}

///afasdfkjasdnfkjansjdfnadsk

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
    dim3 dimGrid(ceil((1.0 * W_unroll) / TILE_WIDTH), ceil((1.0 * M) / TILE_WIDTH), B);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    float *X_unroll;
    cudaMalloc((void**) &X_unroll, W_unroll * H_unroll * sizeof(float));
    //for(int b = 0; b < B; b++) {
    unroll_Kernel<<<dimGrid, dimBlock>>>(B, M, C, H, W, K, X_ptr, X_unroll, W_ptr, y.dptr_);


        //matrixMultiplyShared<<<dimGrid, dimBlock>>>(W_ptr, X_unroll, &y.dptr_[b*M*H_out*W_out], M, H_unroll, H_unroll, W_unroll, M, W_unroll);
    //}
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
