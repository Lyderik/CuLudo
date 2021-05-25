#include <iostream>
#include <chrono>


#include "util/game.cuh"
#include "player/playerNN.cuh"
#include "player/plauer_fast_agro.cuh"
#include "player/player_fast_safe.cuh"
#include "player/player_random.cuh"

#include <vector>
#include <iomanip>
#include <iostream>
#include <fstream>


int main() {



    std::vector<int> hidden_layers;

    /*hidden_layers.push_back(6);
    hidden_layers.push_back(7);
    hidden_layers.push_back(4);*/
    //hidden_layers.push_back(32); // only when even bigger
    hidden_layers.push_back(20);
    hidden_layers.push_back(16);
    hidden_layers.push_back(12);
    hidden_layers.push_back(4);


    PlayerFastAgro* player_agro = new PlayerFastAgro();
    PlayerFastSafe* player_safe = new PlayerFastSafe();
    PlayerRandom* player_rando = new PlayerRandom();
    PlayerNN* player_nn = new PlayerNN(hidden_layers);

    Game games(player_safe, player_safe, player_safe, player_nn);

    std::vector<float> mrv;
    std::vector<float> msv;

    /*for (float mrt = 0.025; mrt < 0.5; mrt *= 2 )
        mrv.push_back(mrt);
    for (float mst = 0.1; mst < 0.5; mst *= 2)
        msv.push_back(mst);*/


    mrv.push_back(0.1);
    msv.push_back(1);

    int epoc = 50;

    auto prefix  = "JUST_TEST_3_bad_names";

    for (auto mr : mrv ) {
        for (auto ms : msv) {
            // Prepare files for data logging
            std::ostringstream stringStreamavg;
            stringStreamavg << prefix << "_DATA_AVG_MR_" << std::fixed << std::setprecision(3) << std::setw(5)<<mr<<"_MS_"<<std::setw(5)<<ms<<".csv";
            std::string fileNameavg = stringStreamavg.str();
            std::cout << fileNameavg << std::endl;
            std::ofstream davgout;
            davgout.open (fileNameavg);

            std::ostringstream stringStream;
            stringStream << prefix << "_DATA_RAW_MR_" << std::fixed << std::setprecision(3) << std::setw(5)<<mr<<"_MS_"<<std::setw(5)<<ms<<".csv";
            std::string fileName = stringStream.str();
            std::ofstream dout;
            dout.open (fileName);

            player_nn->setMutateParams(mr, ms);

            for (int i = 0; i < epoc; i++) {

                games.play();
                int scores[AI_COUNT] = {0};
                int scores_as[AI_COUNT] = {0};
                games.getScores(scores, 3);
                games.getScores(scores_as, 2);
                player_nn->selectNmute(scores_as);


                int total = scores[0];
                dout << scores[0];
                for (int j = 0; j < AI_COUNT; j++) {
                    total += scores[j];
                    dout << ", "<< scores[j] ;
                }
                dout << std::endl;
                std::cout << std::setw(4) << i << " | avg score: " << std::setw(5) << total << " -> "
                          << (float) (total) / (float) (GAMES_COUNT) << std::endl;
                davgout << i << ", " << (float) (total) / (float) (GAMES_COUNT) << std::endl;
            }
            dout.close();
            davgout.close();
            player_nn->reset();
        }
    }
    return 0;
}
