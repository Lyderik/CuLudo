//
// Created by benjamin on 13/04/2021.
//

#include "game.cuh"

#include "status.h"

#include "cuda-exception.cuh"

#include <iomanip>



__global__ void init(unsigned int seed, curandState_t* states)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= GAMES_COUNT) return;
    curand_init(seed, /* the seed can be the same for each thread, here we pass the time from CPU */
                id,   /* the sequence number should be different for each core */
                0,    /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &states[id]);
}
__device__ int extra_rolls(curandState_t* states, int id){ // estra rolls only care about sixes
    for (int i = 0; i <2; i++)
        if(ceilf(curand_uniform(&states[id]) * 6) == 6) return true;
    return false;
}

__global__ void mem_set(int* mem, int value){
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    mem[id] = value;
}

__global__ void mem_set(float* mem, float value){
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    mem[id] = value;
}

__global__ void roll(curandState_t* states, int* numbers)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    numbers[id] = ceilf(curand_uniform(&states[id]) * 6);
}

__device__ int is_star(int sq)
{
    switch(sq)
    {
        case 5:  return 6;
        case 18: return 6;
        case 31: return 6;
        case 44: return 6;

        case 11: return 7;
        case 24: return 7;
        case 37: return 7;
        case 50: return 7; //jump to goal

        default: return 0;
    }
}
__device__ int is_star_behind(int sq)
{
    switch(sq)
    {
        case 5:  return 45; // ene
        case 18: return -7;
        case 31: return -7;
        case 44: return -7;

        case 11: return -6;
        case 24: return -6;
        case 37: return -6;
        case 50: return -6;

        default: return 0;
    }
}

__device__ int is_globe(int t){
    return (t % 13) == 0 || (t % 13) == 8;
}

__device__ void is_globe(int* t, int* g){
    for (int i = 0; i < PIECES_PER_PLAYER; i++) {
        g[i] = is_globe(t[i]);
    }
}

__device__ bool is_immune_t(int* ps, int pi, int t){
    if (is_globe(ps[pi])) return true;
    for (int i = 0; i < PIECES_PER_PLAYER; i++){
        if(pi == i) continue;
        if(ps[i] == t) return true;
    }
    return false;
}
__device__ int n_friends(int* ps, int pi, int t){
    int friends = 0;
    for (int i = 0; i < PIECES_PER_PLAYER; i++){
        if(pi == i) continue;
        if(ps[i] == t) friends++;
    }
    return friends;
}

__device__ int calc_target(int ps, int d)
{
    if(ps < 0)
        return (d == 6)?0:-1;

    if(ps == 99)
        return -1;

    int sq = ps + d;
    sq += is_star(sq);

    if(sq == 57) return 99;
    if(sq > 57) return 114 - sq;
    if(ps < 51 && sq >= 51) sq++; // add one when moving from outside to inside goal lane
    return sq;
}

__device__ void calc_targets (int* ps, int d, int* t)
{
    for (int i = 0; i < PIECES_PER_PLAYER; i++){
        if(ps[i] < 0){
            t[i] = (d == 6)?0:-1;
            continue;
        }
        if(ps[i] == 99) {
            t[i] = 99;
            continue;
        }
        int sq = ps[i] + d;
        sq += is_star(sq);
        if(sq > 56) sq = 114 - sq;
        t[i] = sq;
    }
}

__device__ void opponent_count(int* ps, int* t, int* op)
{
    for (int i = 4; i < PIECES_PER_PLAYER; i++) {
        int sq = t[i];
        if (sq < 0 || sq > 50){
            op[i] = 0;
        }else {
            int count = 0;
            for (int j = 4; j < 16; j++)
                if (ps[j] == sq)
                    count++;
            op[i] = count;
        }
    }
}

__device__ int opponent_count(int* ps, int sq)
{
    if (sq < 0 || sq > 50){
        return 0;
    }

    int count = 0;
    for (int j = 4; j < 16; j++)
        if (ps[j] == sq)
            count++;
    return count;
}

__device__ int oponent_on_sq(int* ps, int sq){
    for (int j = 4; j < 16; j++)
        if (ps[j] == sq)
            return j;
    return -1;
}

__device__ int op_behind(int* ps, int sq){
    int sqs[2];
    sqs[0] = sq;
    sqs[1] = sq + is_star_behind(sq);
    int l = 1 + (sqs[0] != sqs[1]);
    int ranges[8];
    int ridx = 0;
    for (int i = 0; i < l; i++){
        int bl = sqs[i]-6; // bottom limit
        int ul = bl + 5;
        if(bl < 0){ // need wrap and two ranges
            ranges[ridx++] = bl+52;
            ranges[ridx++] = 51;
            ranges[ridx++] = 0;
            ranges[ridx++] = ul;
        }else{ // no wrap one range
            ranges[ridx++] = bl;
            ranges[ridx++] = ul;
        }
    }
    int count = 0;
    for (int i = 0; i < ridx; i+=2){
        for (int j = 4; j < 16; j++) {
            if (ranges[i] <= ps[j] && ps[j] <= ranges[i + 1]){
                count++;
            }
        }
    }
    return count;
}


__device__ void is_valid_move(int* ps, int d, int* vm) {
    for (int i = 0; i < PIECES_PER_PLAYER; i++) {

        if (ps[i] >= 0 && ps[i] < 99){
            vm[i] = 1;
            continue;
        }

        if (ps[i] < 0 && d == 6){
            vm[i] = 1;
            continue;
        }
        vm[i] = 0;
    }
}
__device__ void is_safe_move(int* vm, int* t, int* op, int* g, int* sm, int* dmg){
    for (int i = 0; i < PIECES_PER_PLAYER; i++) {
        if (!vm[i]){
            sm[i] = 0;
            continue;
        }else if(t[i] == 0 || t[i] > 50){
            sm[i] = 1;
            continue;
        }else if(op[i] == 0){
            sm[i] = 1;
            continue;
        }else if(op[i] == 1 && !g[i]){
            sm[i] = 1;
            if(t[i] <= 50 && t[i] >=0)
                dmg[i] = t[i];
            continue;
        }
        sm[i] = 0;
    }
}



__global__ void pre_ana(int* D, int* P, int* S, int* w, float* F, curandState_t* states)
{

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= GAMES_COUNT) return;

    //printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);

    if(w[col] >= 0) return;

    int* ps = P+(col*PIECES_COUNT); // current game pieces
    int* s  = S+(col*PIECES_PER_PLAYER);
    int d = D[col];  // current game dice roll

    int vm[PIECES_PER_PLAYER]; // valid moves
    is_valid_move(ps, d, vm);
    // If no valid moves, add to extra dice rolls for a 6
    bool no_valid = true;

    for (int i = 0; i < PIECES_PER_PLAYER; i++)
        if(vm[i]) no_valid = false;

    if(no_valid){
        if (extra_rolls(states, col)){
            //printf("Hit 6 one the extra rolls: %d\n", col);
            d = 6;
            D[col] = 6;
            for (int i = 0; i < PIECES_PER_PLAYER; i++)
                if (ps[i] == -1) vm[i] = 1;
        }
    }


    int t[PIECES_PER_PLAYER]; // targets
    calc_targets(ps, d, t);

    int op[PIECES_PER_PLAYER]; // opponent count
    opponent_count(ps, t, op);

    int g[PIECES_PER_PLAYER]; // is target globe
    is_globe(t, g);

    int sm[PIECES_PER_PLAYER]; // is move safe
    int dmg[PIECES_PER_PLAYER]= {0}; // dmg if non zero if it can send someone home
    is_safe_move(vm, t, op, g, sm, dmg);


    for (int i = 0; i < PIECES_PER_PLAYER; i++){
        if (!vm[i]) { // If not valid move
            s[i] = INVALID; continue;
        }
        if (!sm[i]){ // If not safe move
            s[i] = VALID; continue;
        }
        if (ps[i] < 0){ // If can move out of start
            s[i] = OUT_OF_START; continue;
        }
        if (t[i] > 50){ // Landing in goal lane
            if (t[i] == 99) { // Piece into goal
                s[i] = INTO_GOAL; continue;
            }
            if (ps[i] < 50) { // Move from outside into goal lane
                s[i] = INTO_GOAL_LANE; continue;
            } // else move inside goal lane
            s[i] = IN_GOAL_LANE; continue;
        }
        // Else safe move one the main lane
        s[i] = SAFE;
        // TODO  populate feature vector

    }
    for (int i = 0; i < PIECES_PER_PLAYER; i++) {
        float *f = F + col * PIECES_PER_PLAYER * FEATURE_COUNT + i * FEATURE_COUNT;
        if (s[i] == SAFE) {
            int imun_p = is_immune_t(ps, i, ps[i]);
            int imun_t = is_immune_t(ps, i, t[i]);
            int f_vuln = (n_friends(ps, i, ps[i]) == 1 && !g[i]);
            int f_safe = (n_friends(ps, i, t[i]) == 1 && !is_globe(t[i]));

            // the normalized dmg
            f[0] = (float) (dmg[i]) / 52.0;
            // get killed potential currently
            f[1] = (float) (imun_p)? 0 : op_behind(ps, ps[i]) / 6.0;
            // get killed potential if move is made
            f[2] = (float) (imun_t)? 0 : op_behind(ps,  t[i]) / 6.0;
            // normalized progress
            f[3] = (float) (t[i]-ps[i]) / 52.0;
            // normalized
            f[4] = (float) (ps[i]) / 52.0;
            // imunity change
            f[5] = (float) (imun_t - imun_p + f_safe - f_vuln);
        }else{
            for (int j = 0; j < FEATURE_COUNT; j++)
                f[j] = 0;
        }
    }
}


__global__ void update_pos(int* P){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= GAMES_COUNT) return;

    int* ps = P+(col*PIECES_COUNT);

    int tp[16];

    for (int i = 0; i < PIECES_PER_PLAYER; i++){
        for (int j = 0; j < PIECES_PER_PLAYER; j++){
            int idx = i + 4*((j+3)%4);
            tp[idx] = ps[i + 4*j];

            if(tp[idx] >= 0 && tp[idx] < 52) {
                tp[idx] -= 13;
                if (tp[idx] < 0) tp[idx] += 52;
            }
        }
    }
    for (int i = 0; i < PIECES_COUNT; i++)
        ps[i] = tp[i];
}


__global__ void play_moves(int* D, int* P, int* S, int*M, int* w, int cp){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= GAMES_COUNT) return;

    if(w[col] >= 0) return;

    int m = M[col];
    int d = D[col];
    int s = S[(col*4)+m];
    int* p = P +(col*16);

    if (m >= 0){
        if(s != INVALID){
            if(s == OUT_OF_START){
                for (int i = 4; i < 16; i++)
                    if(p[i] == 0)
                        p[i] = -1; // kill all oponents
                p[m] = 0;
                return;
            }

            int t = calc_target(p[m], d);
            if (t == 99){
                p[m] = t;
                bool winner = true;
                for (int i = 0; i < PIECES_PER_PLAYER; i++)
                    if(p[i] != 99)
                        winner = false;
                if (winner) w[col] = cp;
            }else {
                int op_at_t = opponent_count(p,t);
                if (op_at_t == 0){
                    p[m] = t;
                } else if (op_at_t == 1) {
                    if(is_globe(t)){
                        p[m] = -1;
                    } else {
                        p[m] = t;
                        p[oponent_on_sq(p, t)] = -1; // send op home
                    }
                }else{
                    p[m] = -1;
                }
            }
        }else{
            // error
        }
    }



}

Game::Game(iPlayer* p0, iPlayer* p1, iPlayer* p2, iPlayer* p3) {

    players[0] = p0;
    players[1] = p1;
    players[2] = p2;
    players[3] = p3;

    // Allocate memory for the pieces
    int* pieces_memory = nullptr;
    cudaMalloc(&pieces_memory, PIECES_COUNT * GAMES_COUNT * sizeof(int));
    CUDAException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory for Tensor3D.");
    pieces = std::shared_ptr<int>(pieces_memory, [&](int* ptr){ cudaFree(ptr); });

    // Allocate memory for the dices
    int* dice_memory = nullptr;
    cudaMalloc(&dice_memory, GAMES_COUNT * sizeof(int));
    CUDAException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory for Tensor3D.");
    dice = std::shared_ptr<int>(dice_memory, [&](int* ptr){ cudaFree(ptr); });

    // Init random states
    cudaMalloc((void**)&states, GAMES_COUNT * sizeof(curandState_t));
    init<<<GAMES_COUNT, 1 >>>(time(0), states);
    //init<<<GAMES_COUNT, 1 >>>(1337UL, states);

    // Allocate status register
    int* status_memory = nullptr;
    cudaMalloc(&status_memory, PIECES_PER_PLAYER * GAMES_COUNT * sizeof(int));
    CUDAException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory for Tensor3D.");
    status = std::shared_ptr<int>(status_memory, [&](int* ptr){ cudaFree(ptr); });

    // Allocate won register
    int* won_memory = nullptr;
    cudaMalloc(&won_memory, GAMES_COUNT * sizeof(int));
    CUDAException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory for Tensor3D.");
    won = std::shared_ptr<int>(won_memory, [&](int* ptr){ cudaFree(ptr); });

    // Allocate feature register
    float* feature_memory = nullptr;
    cudaMalloc(&feature_memory, FEATURE_COUNT * PIECES_PER_PLAYER * GAMES_COUNT * sizeof(float));
    CUDAException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory for Tensor3D.");
    features = std::shared_ptr<float>(feature_memory, [&](float* ptr){ cudaFree(ptr); });

    // Allocate moves register
    int* moves_memory = nullptr;
    cudaMalloc(&moves_memory, GAMES_COUNT * sizeof(int));
    CUDAException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory for Tensor3D.");
    moves = std::shared_ptr<int>(moves_memory, [&](int* ptr){ cudaFree(ptr); });

    reset();
}

void Game::printData(std::ostream& ios = std::cout) {
    ios << "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=:|";
    ios << "  " << current_player << "  ";
    ios << "|:=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=" << std::endl;
    // Print pieces
    ios << "Pieces Matrix" << std::endl;
    int* memP =  (int *)malloc (PIECES_COUNT * GAMES_COUNT * sizeof(int));
    cudaMemcpy(memP, pieces.get(), PIECES_COUNT * GAMES_COUNT * sizeof(int), cudaMemcpyDeviceToHost);

    for(int r = 0; r < PIECES_COUNT; r++){
        for(int c = 0; c < GAMES_COUNT; c++)
            ios << std::setw(4) << memP[c*PIECES_COUNT + r];
        ios << std::endl;
        if (r%4 == 3 && r != 0) ios << std::endl;
    }
    free(memP);
    //
    ios << "Dice Matrix" << std::endl;
    int* memD =  (int *)calloc(GAMES_COUNT, sizeof(int));
    cudaMemcpy(memD, dice.get(), GAMES_COUNT * sizeof(int), cudaMemcpyDeviceToHost);

    for(int c = 0; c < GAMES_COUNT; c++)
        ios << std::setw(4)  << memD[c];
    ios << std::endl;
    free(memD);
    //
    ios << "Status register" << std::endl;
    int* memS =  (int *)calloc(PIECES_PER_PLAYER * GAMES_COUNT, sizeof(int));
    cudaMemcpy(memS, status.get(), PIECES_PER_PLAYER * GAMES_COUNT * sizeof(int), cudaMemcpyDeviceToHost);

    for(int r = 0; r < PIECES_PER_PLAYER; r++){
        for(int c = 0; c < GAMES_COUNT; c++)
            ios << std::setw(4) << memS[c*PIECES_PER_PLAYER + r];
        ios << std::endl;
    }
    free(memS);
    //
    ios << "Moves" << std::endl;
    int* memM =  (int *)calloc(GAMES_COUNT, sizeof(int));
    cudaMemcpy(memM, moves.get(), GAMES_COUNT * sizeof(int), cudaMemcpyDeviceToHost);

    for(int c = 0; c < GAMES_COUNT; c++)
        ios << std::setw(4) << memM[c];
    ios << std::endl;
    free(memM);
    //
    ios << "Won" << std::endl;
    int* memW =  (int *)calloc(GAMES_COUNT, sizeof(int));
    cudaMemcpy(memW, won.get(), GAMES_COUNT * sizeof(int), cudaMemcpyDeviceToHost);

    for(int c = 0; c < GAMES_COUNT; c++)
        ios << std::setw(4) << memW[c];
    ios << std::endl;
    free(memW);



}
void Game::printStats(std::ostream &ios) {
    ios << "TOTAL GAMES: " << GAMES_COUNT << std::endl;
    ios << "TOTAL TURNS: " << n_turn << std::endl;

    int *memW = (int *) calloc(GAMES_COUNT, sizeof(int));
    cudaMemcpy(memW, won.get(), GAMES_COUNT * sizeof(int), cudaMemcpyDeviceToHost);

    ios << std::fixed << std::setprecision(2);
    ios << "WIN DISTRIBUTION:" << std::endl;
    ios << "   P0    P1    P2    P3" << std::endl;

    for (int ai = 0; ai < AI_COUNT; ai++) {
        int p_stats[4] = {0};
        for (int i = 0; i < GAMES_PER_AI; i++)
            p_stats[memW[AI_COUNT*ai + i]]++;
        for (int i = 0; i < PLAYER_COUNT; i++)
            ios << std::setw(6) << (float) p_stats[i] / (float) GAMES_PER_AI;
        ios << std::endl;
    }
    ios << std::endl;

    free(memW);
}

void Game::getScores(int* scores, int player){

    int *memW = (int *) calloc(GAMES_COUNT, sizeof(int));
    cudaMemcpy(memW, won.get(), GAMES_COUNT * sizeof(int), cudaMemcpyDeviceToHost);

    for (int ai = 0; ai < AI_COUNT; ai++) {
        for (int i = 0; i < GAMES_PER_AI; i++) {
            int winner = memW[AI_COUNT * ai + i];
            if (winner == player) scores[ai]++;
        }
    }

    free(memW);
}


bool Game::next_turn() {
    //std::cout << won.get() << " ";
    current_player = (current_player + 1) % 4;
    n_turn++;

    update_pos<<<GAMES_COUNT,1>>>(pieces.get());
    roll<<<GAMES_COUNT, 1 >>>(states, dice.get());
    cudaDeviceSynchronize();
    pre_ana<<<GAMES_COUNT,1>>>(dice.get(), pieces.get(), status.get(), won.get(), features.get(), states);
    cudaDeviceSynchronize();
    players[current_player]->make_decision(pieces.get(), status.get(), dice.get(), features.get(), moves.get(), won.get(), GAMES_COUNT);
    cudaDeviceSynchronize();
    play_moves<<<GAMES_COUNT,1>>>( dice.get(), pieces.get(), status.get(), moves.get(), won.get(), current_player);
    cudaDeviceSynchronize();
    int* memW =  (int *)calloc(GAMES_COUNT, sizeof(int));
    cudaMemcpy(memW, won.get(), GAMES_COUNT * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < GAMES_COUNT; i++) {
        if (memW[i] < 0) {
            free(memW);
            return false;
        }
    }
    free(memW);
    return true;
}

void Game::reset() {
    mem_set<<<PIECES_COUNT * GAMES_COUNT,1>>>(pieces.get(), -1);
    mem_set<<<PIECES_PER_PLAYER * GAMES_COUNT,1>>>(status.get(), INVALID);
    mem_set<<< GAMES_COUNT, 1 >>>(won.get(), -1);
    mem_set<<<FEATURE_COUNT*GAMES_COUNT,1>>>(features.get(), 0);
    mem_set<<<GAMES_COUNT,1>>>(moves.get(), 0);
    init<<<GAMES_COUNT, 1 >>>(time(0), states);
    n_turn = 0;
}

void Game::play() {
    reset();
    while (!next_turn()){
        //if(n_turn < 1000 && n_turn % 4 == 3 ) printData(std::cout);
    };
}