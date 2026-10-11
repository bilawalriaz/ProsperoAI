#include "../src/kokoro_host_generator.cpp"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>

// Unused model/runtime dependencies: this fixture exercises pure convolution.
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
// Reached only through NNOPT_HOSTGEN_MODE=gpu_fp32, which this fixture never sets.
extern "C" int op_decoder_gpu_fp32(OpenCLContext &, Weights &, cl_command_queue, cl_mem, cl_mem,
                                   cl_mem, cl_mem, int, int, std::vector<int16_t> &)
{
    return -1;
}

// Splitting a stage by output channel, row or frame must not change one output
// element's arithmetic, so the threaded path has to match the serial one
// exactly. Every case runs both ways and compares bit-for-bit.
int main()
{
    std::mt19937 rng(273);
    std::uniform_real_distribution<double> value(-1, 1);
    char settings[8];
    for (int threads : {1, 4})
    {
        std::snprintf(settings, sizeof settings, "%d", threads);
        setenv("NNOPT_HOSTGEN_THREADS", settings, 1);
        for (int length : {1, 17, 32, 37, 79, 257})
            for (int kernel : {1, 3, 7})
                for (int dilation : {1, 3, 5})
                    for (int stride : {1, 2, 6})
                        for (bool has_bias : {false, true})
                        {
                            const int channels = 5, outputs = 3;
                            const int padding = dilation * (kernel - 1) / 2;
                            vec64 input(channels * length), weights(outputs * channels * kernel);
                            vec64 bias(has_bias ? outputs : 0);
                            for (auto &x : input)
                                x = value(rng);
                            for (auto &x : weights)
                                x = value(rng);
                            for (auto &x : bias)
                                x = value(rng);
                            vec64 actual;
                            int out_length;
                            conv1d(input, channels, length, weights, outputs, kernel, bias, stride,
                                   padding, dilation, actual, out_length);
                            for (int oc = 0; oc < outputs; ++oc)
                                for (int ol = 0; ol < out_length; ++ol)
                                {
                                    double expected = 0;
                                    for (int tap = 0; tap < kernel; ++tap)
                                    {
                                        int il = ol * stride - padding + tap * dilation;
                                        if (il < 0 || il >= length)
                                            continue;
                                        for (int ic = 0; ic < channels; ++ic)
                                            expected +=
                                                input[ic * length + il] *
                                                weights[oc * channels * kernel + ic * kernel + tap];
                                    }
                                    if (has_bias)
                                        expected += bias[oc];
                                    assert(actual[oc * out_length + ol] == expected);
                                }
                        }
    }

    // Strided instance norm, snake activation and frame-split STFT with four
    // workers against the single-threaded results.
    std::mt19937 rng2(511);
    const int channels = 9, length = 901, frames_in = 777, n_fft = 20, hop = 5;
    vec64 x(channels * length), gamma(channels), beta(channels), alpha(channels);
    for (auto &v : x)
        v = value(rng2);
    for (int c = 0; c < channels; ++c)
    {
        gamma[c] = value(rng2);
        beta[c] = value(rng2);
        alpha[c] = value(rng2);
    }
    vec64 signal(frames_in * hop);
    for (auto &v : signal)
        v = value(rng2);
    vec64 window(n_fft);
    for (int i = 0; i < n_fft; ++i)
        window[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / n_fft));

    std::snprintf(settings, sizeof settings, "%d", 1);
    setenv("NNOPT_HOSTGEN_THREADS", settings, 1);
    const vec64 norm_serial = adain1d(x, channels, length, gamma, beta);
    vec64 snake_serial = x;
    snake_inplace(snake_serial, channels, length, alpha);
    vec64 mag_serial, phase_serial;
    int frames = 0;
    stft(signal, static_cast<int>(signal.size()), n_fft, hop, window, mag_serial, phase_serial,
         frames);

    std::snprintf(settings, sizeof settings, "%d", 4);
    setenv("NNOPT_HOSTGEN_THREADS", settings, 1);
    const vec64 norm_parallel = adain1d(x, channels, length, gamma, beta);
    vec64 snake_parallel = x;
    snake_inplace(snake_parallel, channels, length, alpha);
    vec64 mag_parallel, phase_parallel;
    int frames_parallel = 0;
    stft(signal, static_cast<int>(signal.size()), n_fft, hop, window, mag_parallel, phase_parallel,
         frames_parallel);

    assert(frames == frames_parallel);
    for (size_t i = 0; i < norm_serial.size(); ++i)
        assert(norm_serial[i] == norm_parallel[i]);
    for (size_t i = 0; i < snake_serial.size(); ++i)
        assert(snake_serial[i] == snake_parallel[i]);
    for (size_t i = 0; i < mag_serial.size(); ++i)
        assert(mag_serial[i] == mag_parallel[i]);
    for (size_t i = 0; i < phase_serial.size(); ++i)
        assert(phase_serial[i] == phase_parallel[i]);

    std::puts("Kokoro host convolution: exact FP64 reference comparisons passed");
    std::puts("Kokoro host generator: one and four worker results are identical");
}
