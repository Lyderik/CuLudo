//
// Created by benjamin on 21/05/2021.
//

#ifndef LUDOAI_PLAUER_FAST_AGRO_CUH
#define LUDOAI_PLAUER_FAST_AGRO_CUH


#include "../util/status.h"
#include "iPlayer.cuh"
#include "player_logic.cuh"

__global__ void make_decision_agro(int *P, int *S, int *d, float* F, int *m, int* w) {
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
    // add the agro
    float * f = F + id * PIECES_PER_PLAYER * FEATURE_COUNT;

    float dmg = 0;
    int p = -1;
    for (int i = 0; i < 4; i++) {
        if (f[i*FEATURE_COUNT] > dmg){
            p = i;
            dmg = f[i*FEATURE_COUNT];
        }
    }
    if (p >= 0){
        //printf("Jeg dræbte nogen! >:D\n");
        printf("Kill:\n[ %d, %d, %d, %d, %d, %d, %d, %d,%d, %d, %d, %d,%d, %d, %d, %d ]\n[%d]\n[%f, %f, %f, %f]\n", P[id*16], P[id*16+1], P[id*16+2], P[id*16+3], P[id*16+4], P[id*16+5], P[id*16+6], P[id*16+7], P[id*16+8], P[id*16+9], P[id*16+10], P[id*16+11], P[id*16+12], P[id*16+13], P[id*16+14], P[id*16+15],d[id],f[0],f[4],f[8],f[12]);
        m[id] = p;
        return;
    }

    int into_goal = -1;
    int out_of_start = -1;
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
    //m[id] = first_vm(status);
    m[id] = best_to_suicide(status, P + (16 * id));
}


class PlayerFastAgro : public iPlayer {


public:
    PlayerFastAgro();

private:
    void make_decision_gpu(int* P, int* S, int* d, float* F, int* M, int*w, int Ngames);

};

PlayerFastAgro::PlayerFastAgro() {

}

void PlayerFastAgro::make_decision_gpu(int *P, int *S, int *d, float* F, int *M, int* w, int Ngames) {
    make_decision_agro<<<Ngames,1>>>(P, S, d, F, M, w);
}


#endif //LUDOAI_PLAUER_FAST_AGRO_CUH
