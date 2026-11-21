"""续跑、栅栏与磁盘两轮界定的回归测试。

纯标准库：在仓库根目录外任意位置执行 `python3 backend/tests/test_resume.py`，
或在 `backend/` 下执行 `python3 -m unittest tests.test_resume -v`。
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.domain import SPEC, solve  # noqa: E402

EXPECTED_ORDER = ["BX-05", "BX-09", "BX-02", "BX-06", "BX-07", "BX-04", "BX-08", "BX-01", "BX-03"]


class Fenced(Exception):
    """模拟旧尝试提交时被接手方的令牌栅栏拒绝。"""


class Crash(Exception):
    """模拟后台在一轮提交前崩溃。"""


class Store:
    """模拟 business_runs 与 run_checkpoints 的最小共享状态。"""

    def __init__(self):
        self.token = None
        self.running = True
        self.checkpoint = None
        self.history = []    # 每次确认的 (轮次, 段清单)
        self.snapshots = []  # 每次确认时的 (轮次, 目录内容)


class Checkpointer:
    """与 AttemptCheckpoint 相同的语义：载入最近检查点，提交前做令牌栅栏。"""

    def __init__(self, store, token, directory, hook=None):
        self.store = store
        self.token = token
        self.tag = token
        self.directory = Path(directory)
        self.hook = hook

    def load(self):
        return copy.deepcopy(self.store.checkpoint)

    def commit(self, round_number, run_names, phases):
        if self.hook is not None:
            self.hook(round_number)
        if not self.store.running or self.store.token != self.token:
            raise Fenced(f"执行权已被 {self.store.token} 接手")
        self.store.checkpoint = {
            "round": round_number,
            "runs": list(run_names),
            "phases": [dict(phase) for phase in phases],
            "token": self.token,
        }
        self.store.history.append((round_number, list(run_names)))
        self.store.snapshots.append(
            (round_number, sorted(path.name for path in self.directory.iterdir()))
        )


def run_solve(payload, store, token, directory, hook=None):
    return solve(payload, None, directory=directory,
                 checkpointer=Checkpointer(store, token, directory, hook))


def baseline_result(tmp):
    store = Store()
    store.token = "attempt-A"
    directory = Path(tmp) / "baseline"
    result = run_solve(SPEC["sample"], store, "attempt-A", directory)
    return result, store, directory


class BaselineTest(unittest.TestCase):
    def test_demo_matches_spec(self):
        with TemporaryDirectory() as tmp:
            result, store, directory = baseline_result(tmp)
            for key, value in SPEC["expect"].items():
                self.assertEqual(result["metrics"][key], value, key)
            self.assertEqual(result["metrics"]["初始段峰值字节"], 102)
            rows = result["sections"][0]["rows"]
            self.assertEqual([row["馆藏件号"] for row in rows], EXPECTED_ORDER)
            # 同号馆藏保持原登记顺序：BX-02(第2顺位) 先于 BX-06(第6顺位)
            self.assertEqual([row["原登记顺位"] for row in rows],
                             [5, 9, 2, 6, 7, 4, 8, 1, 3])
            phases = result["sections"][1]["rows"]
            self.assertEqual([row["阶段"] for row in phases],
                             ["生成初始段", "归并第 1 轮", "归并第 2 轮", "归并第 3 轮", "读取上架顺序单"])
            self.assertEqual([row["归并组数"] for row in phases], [0, 2, 1, 1, 0])
            self.assertEqual([row["直接结转段数"] for row in phases], [0, 1, 1, 0, 0])
            # 确认序列完整：初始段 + 3 轮归并
            self.assertEqual([round_number for round_number, _ in store.history], [0, 1, 2, 3])
            # 执行结束后目录中只剩最后一轮的一个段
            self.assertEqual(sorted(p.name for p in directory.iterdir()),
                             store.history[-1][1])


class ResumeTest(unittest.TestCase):
    def test_resume_from_last_confirmed_round(self):
        with TemporaryDirectory() as tmp:
            expected, _, _ = baseline_result(tmp)

            store = Store()
            store.token = "attempt-A"
            directory = Path(tmp) / "run-9"

            def crash_before_round_2(round_number):
                if round_number == 2:
                    raise Crash("后台重启")

            with self.assertRaises(Crash):
                run_solve(SPEC["sample"], store, "attempt-A", directory,
                          hook=crash_before_round_2)

            # 第 2 轮未完整提交，第 1 轮已确认的段必须全部保留
            self.assertEqual(store.checkpoint["round"], 1)
            for name in store.checkpoint["runs"]:
                self.assertTrue((directory / name).is_file(), name)
            # 崩溃现场留下未确认的残留段与未写全的临时段
            (directory / "run-2-0-attempt-A.jsonl.tmp").write_bytes(b'[1,"BX","GJ-2026-1",')
            debris = sorted(p.name for p in directory.iterdir())
            self.assertGreater(len(debris), len(store.checkpoint["runs"]))

            # 接手方从最后完整的一轮续跑
            store.token = "attempt-B"
            result = run_solve(SPEC["sample"], store, "attempt-B", directory)

            # 只出一份上架顺序单，与未中断的执行完全一致（含读写汇总）
            self.assertEqual(result, expected)
            # 确认序列为 0、1（旧后台）+ 2、3（接手方），没有从头重排
            self.assertEqual([round_number for round_number, _ in store.history],
                             [0, 1, 2, 3])
            self.assertEqual(store.checkpoint["token"], "attempt-B")
            # 残留段与临时段被清扫，只剩最终段
            self.assertEqual(sorted(p.name for p in directory.iterdir()),
                             store.history[-1][1])

    def test_crash_after_final_round_republishes_without_remerge(self):
        with TemporaryDirectory() as tmp:
            expected, _, _ = baseline_result(tmp)

            # 第一次尝试完成全部归并，但在发布结果前崩溃
            store = Store()
            store.token = "attempt-A"
            directory = Path(tmp) / "run-14"
            run_solve(SPEC["sample"], store, "attempt-A", directory)
            self.assertEqual(store.checkpoint["round"], 3)
            confirmed = sorted(p.name for p in directory.iterdir())
            self.assertEqual(len(confirmed), 1)

            # 接手方从最后确认的一轮续跑：只剩一个段，直接读取顺序单
            store.token = "attempt-B"
            result = run_solve(SPEC["sample"], store, "attempt-B", directory)
            self.assertEqual(result, expected)
            # 没有重新归并：确认序列仍是首次尝试的 0、1、2、3
            self.assertEqual([round_number for round_number, _ in store.history],
                             [0, 1, 2, 3])

    def test_missing_confirmed_segment_fails_run(self):
        with TemporaryDirectory() as tmp:
            store = Store()
            store.token = "attempt-A"
            directory = Path(tmp) / "run-10"

            def crash_before_round_1(round_number):
                if round_number == 1:
                    raise Crash("后台重启")

            with self.assertRaises(Crash):
                run_solve(SPEC["sample"], store, "attempt-A", directory,
                          hook=crash_before_round_1)
            self.assertEqual(store.checkpoint["round"], 0)
            # 已确认的段丢失属于磁盘故障：执行失败且不发布顺序单
            (directory / store.checkpoint["runs"][0]).unlink()
            store.token = "attempt-B"
            with self.assertRaises(RuntimeError) as ctx:
                run_solve(SPEC["sample"], store, "attempt-B", directory)
            self.assertIn("临时磁盘读写失败", str(ctx.exception))


class FencingTest(unittest.TestCase):
    def test_stale_attempt_cannot_overwrite_confirmed_progress(self):
        with TemporaryDirectory() as tmp:
            expected, _, _ = baseline_result(tmp)

            store = Store()
            store.token = "attempt-A"
            directory = Path(tmp) / "run-11"
            holder = {}

            def takeover(round_number):
                if round_number == 2:
                    # 旧后台第 2 轮尚未提交，新后台接手并续跑完成
                    store.token = "attempt-B"
                    holder["result"] = run_solve(SPEC["sample"], store, "attempt-B", directory)

            with self.assertRaises(Fenced):
                run_solve(SPEC["sample"], store, "attempt-A", directory, hook=takeover)

            # 接手方确认的进度未被旧后台改掉
            self.assertEqual(store.checkpoint["round"], 3)
            self.assertEqual(store.checkpoint["token"], "attempt-B")
            self.assertEqual(len(store.checkpoint["runs"]), 1)
            # 目录里只剩接手方确认的最终段，旧后台的段全部被清扫
            self.assertEqual(sorted(p.name for p in directory.iterdir()),
                             store.checkpoint["runs"])
            # 接手方的结果与未中断的执行一致，且只此一份
            self.assertEqual(holder["result"], expected)


class DiskBoundTest(unittest.TestCase):
    def test_only_two_adjacent_rounds_on_disk(self):
        with TemporaryDirectory() as tmp:
            _, store, directory = baseline_result(tmp)
            history = dict()
            for round_number, names in store.history:
                history[round_number] = set(names)
            snapshots = dict()
            for round_number, listing in store.snapshots:
                snapshots[round_number] = set(listing)
            # 每轮确认时：上一轮的段仍在场（新段写全前保留已确认的段），
            # 且目录内不出现更早轮次的段（至多相邻两轮）
            for round_number in (1, 2, 3):
                self.assertLessEqual(history[round_number - 1], snapshots[round_number])
                self.assertLessEqual(
                    snapshots[round_number],
                    history[round_number - 1] | history[round_number],
                )
            # 结转段沿用原文件并进入下一轮清单
            carried = history[0] & history[1]
            self.assertEqual(len(carried), 1)
            # 全部确认后只剩最后一个段
            self.assertEqual({p.name for p in directory.iterdir()}, history[3])


class EdgeCaseTest(unittest.TestCase):
    def test_empty_records(self):
        with TemporaryDirectory() as tmp:
            store = Store()
            store.token = "attempt-A"
            payload = {"batch_no": "WX-EMPTY", "run_bytes": 110, "fan_in": 2, "records": []}
            result = run_solve(payload, store, "attempt-A", Path(tmp) / "run-12")
            self.assertEqual(result["sections"][0]["rows"], [])
            self.assertEqual(result["metrics"]["上架件数"], 0)
            self.assertEqual(result["metrics"]["磁盘读取字节"], 0)
            self.assertEqual(result["metrics"]["磁盘写入字节"], 0)
            self.assertEqual(result["metrics"]["归并轮数"], 0)
            self.assertEqual(store.checkpoint["runs"], [])

    def test_single_record_needs_no_merge(self):
        with TemporaryDirectory() as tmp:
            store = Store()
            store.token = "attempt-A"
            payload = {
                "batch_no": "WX-ONE", "run_bytes": 110, "fan_in": 2,
                "records": [{"item_id": "BX-01", "accession_no": "GJ-2026-10", "title": "江南舆图"}],
            }
            result = run_solve(payload, store, "attempt-A", Path(tmp) / "run-13")
            self.assertEqual([row["馆藏件号"] for row in result["sections"][0]["rows"]], ["BX-01"])
            self.assertEqual(result["metrics"]["归并轮数"], 0)
            self.assertEqual(result["metrics"]["初始磁盘段数"], 1)


if __name__ == "__main__":
    unittest.main()
