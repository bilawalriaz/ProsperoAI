"""Opt-in real llama.cpp test; the ordinary suite needs no model or dependency build."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class LLMPrefixCacheTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('LLAMA_HOST_BUILD') and os.environ.get('LLAMA_TEST_MODEL'),
                         'set LLAMA_HOST_BUILD and LLAMA_TEST_MODEL for real-model cache checks')
    def test_cached_and_cold_outputs(self):
        dependency = ROOT / '.deps/llama.cpp'
        build = Path(os.environ['LLAMA_HOST_BUILD']).resolve()
        archives = [build / name for name in ['src/libllama.a', 'ggml/src/libggml.a',
                    'ggml/src/libggml-cpu.a', 'ggml/src/libggml-base.a']]
        with tempfile.TemporaryDirectory(prefix='llm-prefix-') as tmp:
            binary = Path(tmp) / 'test'
            subprocess.run([os.environ.get('HOST_CXX', 'clang++'), '-std=c++20', '-O2',
                            '-ffp-contract=off', '-fsanitize=address,undefined', '-I'+str(ROOT/'include'),
                            '-I'+str(dependency/'include'), '-I'+str(dependency/'ggml/include'),
                            str(ROOT/'tests/llm_prefix_cache_test.cpp'),
                            *map(str, archives), '-pthread', '-ldl', '-lm', '-o', str(binary)], check=True)
            subprocess.run([str(binary), os.environ['LLAMA_TEST_MODEL']], check=True, timeout=120)
