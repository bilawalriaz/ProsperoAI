#pragma once
#include <CL/cl.h>
#include <cstdint>
#include <string>
#include <vector>
class OpenCLContext;
class Weights
{
  public:
    cl_mem get_buffer(const std::string &, bool optional = false) __asm__(
        "kokoro__ZN7Weights10get_bufferERKNSt3__112basic_"
        "stringIcNS0_11char_traitsIcEENS0_9allocatorIcEEEEb");
    std::vector<int> get_shape(const std::string &) const
        __asm__("kokoro__ZNK7Weights9get_shapeERKNSt3__112basic_stringIcNS0_11char_traitsIcEENS0_"
                "9allocatorIcEEEE");
};
float nnopt_f16_to_f32(uint16_t) __asm__("kokoro__Z16nnopt_f16_to_f32t");
