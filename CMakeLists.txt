cmake_minimum_required(VERSION 3.20)


set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUTENSOR_ROOT /opt/nvidia/libcutensor)

project(ludoai CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 14)

#add_library(cutensor SHARED IMPORTED )
#set_target_properties(cutensor PROPERTIES
#        IMPORTED_LOCATION "${CUTENSOR_ROOT}/lib/11/libcutensor.so"
#        INTERFACE_INCLUDE_DIRECTORIES "${CUTENSOR_ROOT}/include")

include_directories(/usr/local/cuda/include)

add_executable(ludoNN main.cu util/cuda-exception.cuh util/game.cu util/game.cuh player/player_fast_safe.cuh util/status.h player/iPlayer.cuh player/plauer_fast_agro.cuh player/player_logic.cuh util/settings.cuh player/playerNN.cuh player/player_random.cuh)


set_target_properties(
        ludoNN
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 61
)

target_link_libraries(ludoNN CUDA::curand)

