"""The page of a backend no reference engine runs says so.

    python3 -m unittest discover scripts/tests
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import report as R


class BackendNote(unittest.TestCase):
    def test_an_opencl_page_says_no_reference_engine_runs_it(self):
        note = " ".join(R.backend_note("opencl"))
        self.assertIn("No reference engine runs a language model through OpenCL", note)

    def test_the_other_backends_have_peers_and_carry_no_note(self):
        for backend in ("rocm", "cuda", "vulkan", "unstated"):
            self.assertEqual(R.backend_note(backend), [])


if __name__ == "__main__":
    unittest.main()
