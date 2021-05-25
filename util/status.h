//
// Created by benjamin on 20/05/2021.
//

#ifndef LUDOAI_STATUS_H
#define LUDOAI_STATUS_H

enum Piece_status {
    INVALID = 0,
    VALID = 1,
    SAFE = 2,
    OUT_OF_START = 3,
    INTO_GOAL_LANE = 4,
    IN_GOAL_LANE = 5,
    INTO_GOAL = 6
};

#endif //LUDOAI_STATUS_H
