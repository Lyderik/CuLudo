//
// Created by benjamin on 21/05/2021.
//

#ifndef LUDOAI_IPLAYER_CUH
#define LUDOAI_IPLAYER_CUH

#include "../util/settings.cuh"

class iPlayer {
public:
    void make_decision(int* P, int* S, int* d, float* F, int* M, int* w, int Ngames){
        make_decision_gpu(P, S, d, F, M, w, Ngames);
    }
protected:
    virtual void make_decision_gpu(int* P, int* S, int* d, float* F, int* M, int* w, int Ngames) = 0;
};


#endif //LUDOAI_IPLAYER_CUH
