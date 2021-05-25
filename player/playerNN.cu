//
// Created by benjamin on 22/05/2021.
//

#include "playerNN.cuh"



/*
__global__ void linearLayerForward( float* W, float* A, float* Z, float* b, int Z_x_dim, int Z_y_dim, int inner_dim) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float Z_value = 0;

    if (row < Z_y_dim && col < Z_x_dim) {
        for (int i = 0; i < inner_dim; i++) {
            Z_value += W[row * inner_dim + i] * A[i * Z_x_dim + col];
        }
        Z[row * Z_x_dim + col] = Z_value + b[row];
    }
}

__global__ void reluActivationForward(float* Z, int Z_x_dim) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Z_x_dim) {
        Z[index] = fmaxf(Z[index], 0);
    }
}

__device__ void estimate(float* feature, float* score, float* W, float* b, int *dims, int Nl){
    dim3 block_size(8, 8);
    dim3 relu_block_size(256);

    int M_offset = 0;
    int v_offset = 0;

    float * A = new float[FEATURE_COUNT];
    memcpy(A,feature, FEATURE_COUNT*sizeof(float));
    float * Z;

    for (int i = 1; i < Nl; i++) {
        dim3 num_of_blocks(	(dims[i-1] + block_size.x - 1) / block_size.x,
                               (dims[i] + block_size.y - 1) / block_size.y);
        Z = new float[dims[i]];

        linearLayerForward<<<num_of_blocks, block_size>>>( W + M_offset, A, Z, b + v_offset, 1, dims[i], dims[i-1]);

        if(i == Nl - 1) continue;
        dim3 relu_num_of_blocks(( dims[i] + relu_block_size.x - 1) / relu_block_size.x);
        reluActivationForward <<< relu_num_of_blocks, relu_block_size>>>(Z, dims[i]);

        free(A);
        A = Z;
    }
    score[0] = Z[0];

    free(A);
    free(Z);
}*/


__global__ void make_decision_NN(int *P, int *S, int *d, float* F, int *m, int* w, float* par, float* W, float* b , int* dims, int Nl, int M_size, int b_size) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= GAMES_COUNT) {
        m[id] = -1;
        return;
    }

    if(w[id] >= 0) {
        m[id] = -1;
        return;
    }
    int ai_id = id/AI_COUNT;
    float* params = par + (ai_id * GAMES_PER_AI);

    int* status = S + (id*4);

    if (count_vm(status) <= 1){
        m[id] = first_vm(status);
        return;
    }

    float score[PIECES_COUNT];

    for (int i = 0; i < 4; i++){
        switch(status[i]){
            case INVALID:
                score[i] = - 256;
            case OUT_OF_START:
                score[i] = params[0]; break;
            case INTO_GOAL_LANE:
                score[i] = params[1]; break;
            case INTO_GOAL:
                score[i] = params[2];break;
            case IN_GOAL_LANE:
                score[i] = params[3];break;
            case SAFE:
                float * f = F + id * PIECES_PER_PLAYER * FEATURE_COUNT + i * FEATURE_COUNT;
                //estimate(f,&score[i], W + ai_id * M_size, b + ai_id * b_size, dims, Nl);
                break;
        }
    }
}

PlayerNN::PlayerNN(std::vector<int> &hidden_layers) {

    // Extract dimensions
    N_layers = hidden_layers.size() + 2;
    int* dim_host = new int[N_layers];
    dim_host[0] = FEATURE_COUNT;
    for (int i = 0; i < hidden_layers.size(); i++)
        dim_host[i+1] = hidden_layers[i];
    dim_host [hidden_layers.size()+1] = 1;

    // Allocate dim mem
    int* dim_memory = nullptr;
    cudaMalloc(&dim_memory, N_layers * sizeof(int));
    CUDAException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory.");
    dim = std::shared_ptr<int>(dim_memory, [&](int* ptr){ cudaFree(ptr); });

    // Copy dim to device
    cudaMemcpy(dim.get(),dim_host, N_layers * sizeof(int), cudaMemcpyHostToDevice);

    //Calculate Matrix total size
    M_size = 0;
    for (int i = 1; i < N_layers; i++)
        M_size += dim_host[i-1]*dim_host[i];

    float* w_memory = nullptr;
    cudaMalloc(&w_memory, AI_COUNT * M_size * sizeof(float));
    CUDAException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory.");
    W = std::shared_ptr<float>(w_memory, [&](float* ptr){ cudaFree(ptr); });

    //Calculate vector total size
    b_size = 0;
    for (int i = 1; i < N_layers; i++)
        b_size += dim_host[i];

    float* b_memory = nullptr;
    cudaMalloc(&b_memory, AI_COUNT * b_size * sizeof(float));
    CUDAException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory.");
    b = std::shared_ptr<float>(b_memory, [&](float* ptr){ cudaFree(ptr); });

    float* p_memory = nullptr;
    cudaMalloc(&p_memory, PARAM_COUNT * sizeof(float));
    CUDAException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory.");
    p = std::shared_ptr<float>(p_memory, [&](float* ptr){ cudaFree(ptr); });

}


void PlayerNN::make_decision_gpu(int *P, int *S, int *d, float *F, int *M, int *w, int Ngames) {

}


