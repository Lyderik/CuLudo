//
// Created by benjamin on 22/05/2021.
//

#ifndef LUDOAI_PLAYERNN_CUH
#define LUDOAI_PLAYERNN_CUH

#include "iPlayer.cuh"
#include "player_logic.cuh"
#include <memory>
#include <vector>
#include <algorithm>
#include "../util/status.h"
#include "../util/cuda-exception.cuh"
#include <curand.h>
#include <random>

__global__ void init_rand(unsigned int seed, curandState_t* states)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= GAMES_COUNT) return;
    curand_init(seed, /* the seed can be the same for each thread, here we pass the time from CPU */
                id,   /* the sequence number should be different for each core */
                0,    /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &states[id]);
}

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

__device__ float sigmoid(float x) {
    return 1.0f / (1 + exp(-x));
}

__device__ void estimate(float* feature, float& score, float* W, float* b, int *dims, int Nl){
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

        delete[] A;
        A = Z;
    }
    score = sigmoid(Z[0]);

    delete[] A;
    delete[] Z;
}



__global__ void make_decision_NN(int *P, int *S, int *d, float* F, int *m, int* w, float* par, float* Ws, float* b , int* dims, int Nl, int Ms, int vs) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= GAMES_COUNT) {
        m[id] = -1;
        return;
    }
    int ai_id = id / AI_COUNT;

    float* params = par + (ai_id * PARAM_COUNT);
    if(w[id] >= 0) {
        m[id] = -1;
        //printf("Params: %f | %f | %f | %f \n", params[0], params[1], params[2], params[3]);
        return;
    }


    int* status = S + (id*4);

    if (count_vm(status) <= 1){
        m[id] = first_vm(status);
        return;
    }

    float score[PIECES_COUNT];

    for (int i = 0; i < 4; i++){
        switch(status[i]){
            case INVALID:
                score[i] = params[4];break;
            case OUT_OF_START:  // Special case
                score[i] = params[0]; break;
            case INTO_GOAL_LANE: // Special case
                score[i] = params[1]; break;
            case INTO_GOAL: // Special case
                if(id == 1) printf("Params: %f | %f | %f | %f \n", params[0], params[1], params[2], params[3]);
                score[i] = params[2];break;
            case IN_GOAL_LANE: // Special case
                score[i] = params[3];break;
            case SAFE: // Use network to estimate score
                estimate( &F[id * PIECES_PER_PLAYER * FEATURE_COUNT + i * FEATURE_COUNT], score[i], &Ws [ai_id * Ms], &b[ai_id * vs], dims, Nl);
                break;
            default:
                score[i] = -2.0;
        }
    }
    int bm = -1;
    float bs = -1.0;
    for (int i = 0; i < 4; i++){
        if (score[i] > bs){
            bm = i;
            bs = score[i];
        }
    }
    if(bm >= 0){
        m[id] = bm;
    }else{
        m[id] = best_to_suicide(status, P + (16 * id));
    }
}
__global__ void mutate(curandState_t* states, int rid, float mr, float ms, float* m, int size){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= size) return;

    if(curand_uniform(&states[rid]) > mr) return;
    m[id] += curand_normal(&states[rid]) * ms;

}
__global__ void handle_mutate(curandState_t* states, float mr, float ms, int* best, float* W, float* b, float* par, int Ms, int vs){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= AI_COUNT) return;
    if(best[id]) return; // dont mutate the best

    mutate<<<Ms,1>>>(states, id, mr, ms, W, Ms);
    mutate<<<vs,1>>>(states, id, mr, ms, b, vs);
    mutate<<<AI_COUNT*PARAM_COUNT,1>>>(states, id, mr, ms, par, AI_COUNT*PARAM_COUNT);
}

class PlayerNN : public iPlayer {
public:
    PlayerNN(std::vector<int> &hidden_layers);
    void reset();
    void selectNmute(int* scores);
    void setMutateParams(float mr, float ms);

private:
    void make_decision_gpu(int* P, int* S, int* d, float* F, int* M, int*w, int Ngames);

    curandState_t* states; // TODO MAKE SHARED

    int N_layers = 0;
    int M_size = 0;
    int v_size = 0;

    std::shared_ptr<int> dim;
    std::shared_ptr<float> Ws; // Weight
    std::shared_ptr<float> b; // Bias
    std::shared_ptr<float> p; // Params

    float mut_rate = MUTATION_RATE;
    float mut_str  = MUTATION_STRENGTH;

};


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
    Ws = std::shared_ptr<float>(w_memory, [&](float* ptr){ cudaFree(ptr); });

    //Calculate vector total size
    v_size = 0;
    for (int i = 1; i < N_layers; i++)
        v_size += dim_host[i];

    float* b_memory = nullptr;
    cudaMalloc(&b_memory, AI_COUNT * v_size * sizeof(float));
    CUDAException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory.");
    b = std::shared_ptr<float>(b_memory, [&](float* ptr){ cudaFree(ptr); });

    float* p_memory = nullptr;
    cudaMalloc(&p_memory, AI_COUNT * PARAM_COUNT * sizeof(float));
    CUDAException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory.");
    p = std::shared_ptr<float>(p_memory, [&](float* ptr){ cudaFree(ptr); });

    // Init random states
    cudaMalloc((void**)&states, AI_COUNT * sizeof(curandState_t));
    init_rand<<<AI_COUNT, 1 >>>(time(0), states);

    reset();
}


void PlayerNN::make_decision_gpu(int *P, int *S, int *d, float *F, int *M, int *w, int Ngames) {
    //(int *P, int *S, int *d, float* F, int *m, int* w, float* par, float* Ws, float* b , int* dims, int Nl, int Ms, int vs)
    make_decision_NN<<<Ngames,1>>>(P, S, d, F, M, w, p.get(), Ws.get(), b.get(), dim.get(), N_layers, M_size, v_size);
}

void PlayerNN::reset() {
    curandGenerator_t gen;

    /* Create pseudo-random number generator */
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    /* Set seed */
    curandSetPseudoRandomGeneratorSeed(gen, time(0));


    curandGenerateNormal(gen, Ws.get(), AI_COUNT * M_size, 0, 1);
    curandGenerateNormal(gen, b.get(), AI_COUNT * v_size, 0, 1);
    curandGenerateNormal(gen, p.get(), AI_COUNT * PARAM_COUNT, 0, 1);
}

void PlayerNN::selectNmute(int* scores){
    std::vector<std::pair<int,int>> sc;
    for (int ai = 0; ai < AI_COUNT; ai++)
        sc.push_back(std::pair<int,int>(scores[ai], ai));

    std::sort(sc.begin(), sc.end(), [](std::pair<int,int> a, std::pair<int,int> b) {
        return a.first > b.first;
    });

    int top_host[AI_COUNT] = {0};
    for (int i = 0; i < AI_KEEP; i++)
        top_host[sc[i].second] = 1;

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,24);

    for (int ai = 0; ai < AI_COUNT; ai++){
        if(!top_host[ai]){
            int parent = sc[distribution(generator)].second;
            cudaMemcpy( Ws.get()+M_size*ai,    Ws.get()+M_size*parent,      M_size * sizeof(float), cudaMemcpyDeviceToDevice );
            CUDAException::throwIfDeviceErrorsOccurred("Cannot copy CUDA memory of W.");
            cudaMemcpy( b.get()+v_size*ai,      b.get()+v_size*parent,      v_size * sizeof(float), cudaMemcpyDeviceToDevice );
            CUDAException::throwIfDeviceErrorsOccurred("Cannot copy CUDA memory of b.");
            cudaMemcpy( p.get()+PARAM_COUNT*ai, p.get()+PARAM_COUNT*parent, PARAM_COUNT * sizeof(float), cudaMemcpyDeviceToDevice );
            CUDAException::throwIfDeviceErrorsOccurred("Cannot copy CUDA memory of p.");
        }
    }


    int* top_dev = nullptr;
    cudaMalloc(&top_dev, AI_COUNT * sizeof(int));
    CUDAException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory.");

    cudaMemcpy(top_dev,top_host, AI_COUNT * sizeof(int), cudaMemcpyHostToDevice);

    handle_mutate<<<AI_COUNT,1>>>(states, mut_rate, mut_str, top_dev, Ws.get(), b.get(), p.get(), M_size, v_size);

    cudaFree(top_dev);
}

void PlayerNN::setMutateParams(float mr, float ms) {
    mut_rate = mr;
    mut_str = ms;
}


#endif //LUDOAI_PLAYERNN_CUH
