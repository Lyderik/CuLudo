//
// Created by benjamin on 21/05/2021.
//

#ifndef LUDOAI_PLAYER_LOGIC_CUH
#define LUDOAI_PLAYER_LOGIC_CUH
#include "../util/status.h"

__device__ int count_vm(int *s){
    int n_vm = 0;
    for (int i = 0; i < 4; i++)
        if (s[i] != INVALID)
            n_vm++;
    return n_vm;
}
__device__ int first_vm(int *s){
    for (int i = 0; i < 4; i++)
        if (s[i] != INVALID)
            return i;
    return -1;
}
__device__ int best_to_suicide(int *s, int* p){
    int pb = -1;
    int ib = -1;
    for (int i = 0; i < 4; i++){
        if (s[i] != INVALID && p[i] > pb){
            ib = i;
            pb = p[i];
        }
    }
    return ib;
}

#endif //LUDOAI_PLAYER_LOGIC_CUH
