//
// Created by benjamin on 25/05/2021.
//

#ifndef LUDOAI_PLAYER_RANDOM_CUH
#define LUDOAI_PLAYER_RANDOM_CUH

//
// Created by benjamin on 20/05/2021.
//


#include "../util/status.h"
#include "iPlayer.cuh"
#include "player_logic.cuh"
#include <curand.h>

__global__ void init_rand_player(unsigned int seed, curandState_t* states)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= GAMES_COUNT) return;
    curand_init(seed, /* the seed can be the same for each thread, here we pass the time from CPU */
                id,   /* the sequence number should be different for each core */
                0,    /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &states[id]);
}



__global__ void make_decision_random(int *S, int *m, int* w, curandState_t* states) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= GAMES_COUNT) return;


    if(w[id] >= 0) {
        m[id] = -1;
        return;
    }

    int* status = S + (id*4);

    if (count_vm(status) <= 1){
        m[id] = first_vm(status);
        return;
    }

    int vm[PIECES_PER_PLAYER]={0};
    int vmi = 0;
    for (int i = 0; i < PIECES_PER_PLAYER; i++){
        if(status[i] != INVALID)
            vm[vmi++] = i;
    }
    int r = ceilf(curand_uniform(&states[id]) * vmi);
    m[id] = vm[r - 1];
}


class PlayerRandom : public iPlayer {


public:
    PlayerRandom();

private:
    curandState_t* statess; // TODO MAKE SHARED

    void make_decision_gpu(int* P, int* S, int* d, float* F, int* M, int*w, int Ngames);

};

PlayerRandom::PlayerRandom() {
    cudaMalloc((void**)&statess, GAMES_COUNT * sizeof(curandState_t));
    CUDAException::throwIfDeviceErrorsOccurred("Cannot copy CUDA memory for states.");
    init_rand_player<<<GAMES_COUNT, 1 >>>(time(0), statess);
    CUDAException::throwIfDeviceErrorsOccurred("Cannot init CUDA memory of stats");
}

void PlayerRandom::make_decision_gpu(int *P, int *S, int *d, float* F, int *M, int*w, int Ngames) {
    make_decision_random<<<Ngames,1>>>(S, M, w, statess);
    CUDAException::throwIfDeviceErrorsOccurred("Cannot make random decisions.");
}







#endif //LUDOAI_PLAYER_RANDOM_CUH
