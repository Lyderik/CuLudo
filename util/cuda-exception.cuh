//
// Created by benjamin on 12/04/2021.
//

#ifndef LUDOAI_CUDA_EXCEPTION_CUH
#define LUDOAI_CUDA_EXCEPTION_CUH

#include <exception>
#include <iostream>


#include <cuda_runtime.h>

class CUDAException : std::exception {
private:
    const char* exception_message;

public:
    CUDAException(const char* exception_message) :
            exception_message(exception_message)
    { }

    virtual const char* what() const throw()
    {
        return exception_message;
    }

    static void throwIfDeviceErrorsOccurred(const char* exception_message) {
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << error << ": " << exception_message;
            throw CUDAException(exception_message);
        }
    }
};


#endif //LUDOAI_CUDA_EXCEPTION_CUH
