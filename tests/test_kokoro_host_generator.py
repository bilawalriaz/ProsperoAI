import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class KokoroHostTests(unittest.TestCase):
    def test_exact_fp64_convolution(self):
        with tempfile.TemporaryDirectory(prefix='kokoro-host-') as tmp:
            binary = Path(tmp) / 'test'
            subprocess.run([
                os.environ.get('HOST_CXX', 'clang++'), '-std=c++20', '-O3',
                '-ffp-contract=off', '-fsanitize=address,undefined',
                '-DCL_TARGET_OPENCL_VERSION=120', '-I'+str(ROOT/'vendor/include'),
                str(ROOT/'tests/kokoro_host_conv_test.cpp'), '-o', str(binary)
            ], check=True)
            subprocess.run([str(binary)], check=True)
