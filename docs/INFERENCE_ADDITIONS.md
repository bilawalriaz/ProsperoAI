# Inference additions

This branch combines upstream PRs [#17](https://github.com/blackbearreloaded/ProsperoAI/pull/17),
[#18](https://github.com/blackbearreloaded/ProsperoAI/pull/18) and
[#19](https://github.com/blackbearreloaded/ProsperoAI/pull/19), based on upstream
`bd54e3c74d805b8125ec83ae7efcead1021cbe70`.

It adds Vulkan prompt-prefix reuse, optional MiniCPM5 2B Q8_0, and settings for
up to 16384 context tokens and 4096 response tokens. Existing branding, version,
model selection and startup behaviour are retained. MiniCPM is downloaded and
selected through Models. Replies default to 128 tokens in native settings and
512 in APIs/browser settings. Model default context uses the existing load
configuration, including context_size.txt.

The shared Vulkan runtime edits from the independent PRs are resolved here.
See [prefix reuse](LLM_PREFIX_REUSE.md), [MiniCPM5](MINICPM5.md), and
[context controls](CONTEXT_AND_OUTPUT.md) for usage and measurements.

## Engineering checks, 2026-10-10

Linux make check passed: lint, 37 host tests (two optional skips), and the standard
app build. The full native Vulkan/AGC cross-build also passed. The actual
MiniCPM5 Q8 CPU cached/cold regression passed under ASan/UBSan in 45.166 seconds.
Each independent PR also passed its own host checks and both builds.
Console timings in the linked records came from the earlier integrated
performance build. This combined branch was checked on the host.
