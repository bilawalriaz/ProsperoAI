#include "../src/kokoro_host_generator.cpp"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Unused model/runtime dependencies: this fixture exercises the work splitter.
cl_mem Weights::get_buffer(const std::string &, bool)
{
    return nullptr;
}
std::vector<int> Weights::get_shape(const std::string &) const
{
    return {};
}
float nnopt_f16_to_f32(uint16_t)
{
    return 0;
}
extern "C" cl_int clEnqueueReadBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *,
                                      cl_uint, const cl_event *, cl_event *)
{
    return -1;
}
extern "C" int op_decoder_gpu_fp32(OpenCLContext &, Weights &, cl_command_queue, cl_mem, cl_mem,
                                   cl_mem, cl_mem, int, int, std::vector<int16_t> &)
{
    return -1;
}

// The pool hands each slice to a worker and counts exactly the workers that
// took one. Two defects lived here: workers without a slice decremented the
// completion count, which let the calling thread return while a worker was
// still inside the body (reading a dead context), and the worker re-entered
// pthread_cond_wait without holding the mutex, which libpthread aborts on.
// Repeated dispatches with a changing thread count and a changing item count
// are the cheapest way to catch either one.
int main()
{
    char settings[8];
    for (int iteration = 0; iteration < 4000; ++iteration)
    {
        const int items = 1 + (iteration % 9);
        const int threads = (iteration % 97 == 0) ? 1 : 4;
        std::snprintf(settings, sizeof settings, "%d", threads);
        setenv("NNOPT_HOSTGEN_THREADS", settings, 1);
        std::vector<double> out(static_cast<size_t>(items), -1.0);
        parallel_range(items,
                       [&](int first, int last)
                       {
                           for (int i = first; i < last; ++i)
                               out[static_cast<size_t>(i)] = static_cast<double>(i);
                       });
        for (int i = 0; i < items; ++i)
        {
            if (out[static_cast<size_t>(i)] != static_cast<double>(i))
            {
                std::fprintf(stderr, "iteration %d items %d index %d: %g\n", iteration, items, i,
                             out[static_cast<size_t>(i)]);
                return 1;
            }
        }
    }
    // Every worker must be done before the call returns, so a second dispatch
    // that overwrites the same slice cannot lose a write.
    std::vector<double> shared(64, 0.0);
    for (int iteration = 0; iteration < 500; ++iteration)
    {
        const double value = static_cast<double>(iteration);
        parallel_range(static_cast<int>(shared.size()),
                       [&](int first, int last)
                       {
                           for (int i = first; i < last; ++i)
                               shared[static_cast<size_t>(i)] = value;
                       });
        for (size_t i = 0; i < shared.size(); ++i)
            assert(shared[i] == value);
    }
    std::puts("Kokoro work splitter: 4000 mixed dispatches and 500 repeated writes clean");
    return 0;
}
