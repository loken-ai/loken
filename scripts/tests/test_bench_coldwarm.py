"""The report's cold and warm split, on two real result files (gpu long prompt, cpu short prompt).

The fixtures are copies of results/qwen3_8b_4096_stream_long.json and
results/qwen3_8b_4096_stream_cpu.json with the per-sample traces dropped.

    python3 -m unittest discover scripts/tests
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import bench_row as B
import scoreboard as SB

FIX = pathlib.Path(__file__).resolve().parent / "fixtures"
GPU = ("qwen3:8b", 4096, "long", "stream", "GPU")
CPU = ("qwen3:8b", 4096, "short", "stream", "CPU")


def key(r):
    return (r[0], int(r[1])) + tuple(r[2:5])


class ColdWarm(unittest.TestCase):
    def setUp(self):
        self.merged = B.load_all(FIX)

    def test_cold_is_first_iteration_and_warm_the_median_of_the_rest(self):
        self.assertEqual(B.cold_warm([{"x": 1}, {"x": 5}, {"x": 9}], lambda i: i["x"]), (1, 7))
        self.assertEqual(B.cold_warm([{"x": 1}, {"x": 5}, {"x": 9}, {"x": 2}], lambda i: i["x"]), (1, 5))
        self.assertEqual(B.cold_warm([{"x": 4}], lambda i: i["x"]), (4, None))

    def test_prompt_cache_shows_in_the_warm_prefill(self):
        o = self.merged[GPU]["ollama@0.35.1"]
        self.assertAlmostEqual(o["pre_c"], 1548.6, 1)
        self.assertAlmostEqual(o["pre_w"], 11326.3, 1)
        self.assertAlmostEqual(o["ttft_c"], 151.75, 2)

    def test_delta_against_the_peer_is_per_temperature(self):
        loken = next(i for r, i in B.cells_detail(self.merged) if i["plain"] and i["key"] == GPU)
        self.assertAlmostEqual(loken["dec"][0], (66.1758 / 80.7002 - 1) * 100, 2)
        self.assertAlmostEqual(loken["dec"][1], (72.9941 / 84.3990 - 1) * 100, 2)
        # time to first token: positive means loken is faster
        self.assertAlmostEqual(loken["ttft"][0], (151.7508 / 664.7946 - 1) * 100, 2)
        self.assertLess(loken["ttft"][1], loken["ttft"][0])

    def test_table_prints_cold_and_warm_side_by_side(self):
        row = next(r for r in B.cells(self.merged) if key(r) == GPU and "loken" in r[5])
        self.assertIn("353.5 / 689.5", row[B.COLS.index("Prefill tok/s")])
        self.assertIn("66.2 / 73.0", row[B.COLS.index("Decode tok/s")])
        self.assertIn("-18.0% / -13.5%", row[B.COLS.index("Δ decode")])

    def test_incoherent_answer_withholds_both_temperatures(self):
        loken = next(i for r, i in B.cells_detail(self.merged) if i["plain"] and i["key"] == CPU)
        self.assertEqual(loken["mark"], "incoherent")
        self.assertEqual(loken["dec"], (None, None))

    def test_single_iteration_has_cold_only(self):
        for doc in self.merged.values():
            for e in doc.values():
                e["pre_w"] = e["ttft_w"] = e["dec_w"] = None
        info = next(i for r, i in B.cells_detail(self.merged) if i["plain"] and i["key"] == GPU)
        self.assertIsNone(info["dec"][1])
        row = next(r for r in B.cells(self.merged) if key(r) == GPU and "loken" in r[5])
        self.assertIn("66.2 / -", row[B.COLS.index("Decode tok/s")])

    def test_scoreboard_counts_each_temperature(self):
        detail = B.cells_detail(self.merged)
        cold, warm = SB.tally(detail, 0), SB.tally(detail, 1)
        self.assertEqual((cold["total"], cold["won"], cold["lost_decode"]), (1, 0, 1))
        self.assertEqual((warm["total"], warm["ours"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
