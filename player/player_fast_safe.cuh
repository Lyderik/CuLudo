//
// Created by benjamin on 20/05/2021.
//

#ifndef LUDOAI_PLAYER_FAST_SAFE_CUH
#define LUDOAI_PLAYER_FAST_SAFE_CUH

#include "../util/status.h"
#include "iPlayer.cuh"
#include "player_logic.cuh"

__global__ void make_decision_fast(int *P, int *S, int *d, float* F, int *m, int* w) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= GAMES_COUNT) {
        m[id] = -1;
        return;
    }

    if(w[id] >= 0) {
        m[id] = -1;
        return;
    }

    int* status = S + (id*4);

    if (count_vm(status) <= 1){
        m[id] = first_vm(status);
        return;
    }

    int into_goal = -1;
    int out_of_start = -1;
    int p = -1;
    int s = -1;

    for (int i = 0; i < 4; i++){
        switch(status[i]){
            case INTO_GOAL_LANE:
                m[id] = i;
                return;
            case INTO_GOAL:
                into_goal = i;
                break;
            case OUT_OF_START:
                out_of_start = i;
                break;
            case SAFE:
                if(P[i + (id * 16)] > s){
                    p = i;
                    s = P[i + (id * 16)];
                }
                break;
        }
    }
    if(into_goal >= 0){
        m[id] = into_goal;
        return;
    }
    if(out_of_start >= 0){
        m[id] = out_of_start;
        return;
    }
    if(p >= 0){
        m[id] = p;
        return;
    }
    m[id] = first_vm(status);
    //m[id] = best_to_suicide(status, P + (16 * id));


}


class PlayerFastSafe : public iPlayer {


public:
    PlayerFastSafe();

private:
     void make_decision_gpu(int* P, int* S, int* d, float* F, int* M, int*w, int Ngames);

};

PlayerFastSafe::PlayerFastSafe() {

}

void PlayerFastSafe::make_decision_gpu(int *P, int *S, int *d, float* F, int *M, int*w, int Ngames) {
    make_decision_fast<<<Ngames,1>>>(P, S, d, F, M, w);
}


#endif //LUDOAI_PLAYER_FAST_SAFE_CUH


