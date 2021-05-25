//
// Created by benjamin on 13/04/2021.
//

#ifndef LUDOAI_GAME_CUH
#define LUDOAI_GAME_CUH

#include <vector>
#include <memory>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "settings.cuh"
#include "../player/iPlayer.cuh"

class Game {
public:
    Game(iPlayer* p0, iPlayer* p1, iPlayer* p2, iPlayer* p3);

    void printData(std::ostream& ios);
    void printStats(std::ostream& ios);

    void play();

    void getScores(int* scores, int player);


private:
    curandState_t* states; // TODO MAKE SHARED

    std::shared_ptr<int> dice;      // Vector of dices
    std::shared_ptr<int> pieces;
    std::shared_ptr<int> status;    // piece move status
    std::shared_ptr<int> won;
    std::shared_ptr<float> features;
    std::shared_ptr<int> moves;

    int current_player = 0;
    int n_turn = 0;

    curandGenerator_t gen;

    iPlayer* players[PLAYER_COUNT];

    bool next_turn();
    void reset();



};


#endif //LUDOAI_GAME_CUH
