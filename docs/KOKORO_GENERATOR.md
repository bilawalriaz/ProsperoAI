# Kokoro waveform generator: threaded FP64 host path

`vendor/lib/libkokoro-tts.a` keeps a correctness-first serial FP64 CPU decoder for the
waveform (`op_decoder_host`). The README describes that path as "not a real-time
path", and the console numbers agree: on FW 13.60 a short phrase takes about 80
seconds of wall time for two seconds of audio.

This change adds `src/kokoro_host_generator.cpp`, a definition of the same two entry
points the archive calls (`op_decoder_host`, `op_decoder_gpu_fp32`). The app's objects
are linked before the archives, so the definitions here are used and the archive's
serial implementation is never extracted. The archive itself is untouched.

What the generator does:

- Splits the expensive stages across worker threads: `conv1d`/`convtr1d` by output
  channel, `adain1d`/`snake_inplace` by channel, the STFT by frame, and the
  weight-norm reconstruction by output channel. The pool is created once and reused.
- Keeps FP64 accumulation and the reference reduction order. Convolution batches
  sixteen time samples per step so AVX2 can hold independent FP64 accumulators, and
  the file is compiled with `-O3 -mavx2 -ffp-contract=off`.
- Keeps the archive's FP32 comparison route reachable: `NNOPT_HOSTGEN_MODE=gpu_fp32`
  forwards to `op_decoder_gpu_fp32` instead of computing on the host.

Environment (all optional, all read once per call):

| variable | meaning |
| --- | --- |
| `NNOPT_HOSTGEN_THREADS` | worker count, 1 to 16; default 8; 1 is the serial path |
| `NNOPT_HOSTGEN_TIMING` | `1` prints one `[hostgen]` stage line per call |
| `NNOPT_HOSTGEN_MODE` | `gpu_fp32` uses the archive's FP32 decoder for comparison |

## Tests

```sh
python3 -m unittest discover -s tests -p 'test_kokoro_host_generator.py' -v
python3 -m unittest discover -s tests -p 'test_kokoro_pool.py' -v
```

`tests/kokoro_host_conv_test.cpp` runs the convolution serial and threaded and
compares **bit for bit** against an independent FP64 reference, under ASan and UBSan;
`tests/kokoro_pool_test.cpp` hammers repeated dispatch and thread-count switching,
which is where the first pool implementation failed (a worker without a slice
decremented the completion count, and the worker re-entered a wait without the
mutex).

## Measurements, FW 13.60, "Yo" (1.325 s of 24 kHz mono audio)

Same console, same prompt, three runs each; "before" is the archive's decoder,
measured with the same harness on the same day.

| | before (archive decoder) | after (this generator) |
| --- | ---: | ---: |
| app `/api/generate` wall | 53.29 s | 2.334 s |
| benchmark round, 1 worker | 51.66 s | 15.31 s |
| benchmark round, 4 workers | 51.15 s | 3.29 s |
| benchmark round, 8 workers | 51.15 s | 2.39 s |

That is **22.8x** on the app path and **21.4x** on the 8-worker benchmark round, and
19.8x the audio bytes for 20.0x the time: the cost still tracks the audio, not the
prompt. The generator's own stage line for the 8-worker round shows where the time
went before: `convolution=1.647 s` of `total=1.953 s`.

The FP32 route stays closed on this firmware: the console's OpenCL shim does not
implement `convtr1d_f32_c4x4_tex`, `convtr1d_f32_fast_c4`, `gf_convtr_pack_image` or
`leaky_relu_f32`, so `NNOPT_HOSTGEN_MODE=gpu_fp32` fails with `rc=8` and the host path
is the only working one.

## Output

The threaded path is **bit-identical** to the serial path: 1, 4 and 8 workers produce
the same WAV (same SHA-256) for every prompt tried, and the app writes that same file.

It is **not** bit-identical to the archive decoder's output. Measured on the console,
same prompt, both 31,800 frames of 24 kHz mono PCM16:

| | value |
| --- | ---: |
| max sample difference | 114 of 32768 (0.35 % of full scale) |
| RMS difference | 0.64 % of the signal RMS |
| SNR of the difference | 43.9 dB |

So it is the same speech with a small numeric difference, not a different rendering:
the signal energy is the same (RMS 1464.0 vs 1464.4), and the difference starts about
0.24 s in. Which stage introduces it has not been isolated; the convolution itself
matches an independent FP64 reference exactly. Anyone comparing the two files should
expect this difference, and it is worth isolating before this path replaces the
archive's decoder by default.

## Build

`tools/build-vulkan-native.sh` compiles the file with the flags above; nothing else
changed. `vendor/lib/libkokoro-tts.a` is untouched.
