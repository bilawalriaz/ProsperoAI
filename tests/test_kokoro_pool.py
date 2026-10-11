import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class KokoroPoolTests(unittest.TestCase):
    # The splitter's lock discipline and completion count are timing dependent:
    # the failures they caused (a dead body context, and libpthread aborting on
    # a wait without the mutex) showed up in roughly half of the runs. Run the
    # fixture several times so a regression is not a coin flip.
    def test_repeated_dispatch_and_thread_switch(self):
        with tempfile.TemporaryDirectory(prefix='kokoro-pool-') as tmp:
            binary = Path(tmp) / 'test'
            subprocess.run([
                os.environ.get('HOST_CXX', 'clang++'), '-std=c++20', '-O2',
                '-ffp-contract=off', '-fsanitize=address,undefined',
                '-DCL_TARGET_OPENCL_VERSION=120', '-I' + str(ROOT / 'vendor/include'),
                str(ROOT / 'tests/kokoro_pool_test.cpp'), '-o', str(binary)
            ], check=True)
            for _ in range(3):
                subprocess.run([str(binary)], check=True, stdout=subprocess.PIPE)
