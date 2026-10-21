import time

from scripts.benchmark_live_serving import run_threaded


def test_run_threaded_keeps_only_bounded_requests_in_flight():
    items = [{"query_id": index} for index in range(12)]

    def execute(item):
        started = time.perf_counter_ns()
        time.sleep(0.005)
        finished = time.perf_counter_ns()
        return {
            "query_id": item["query_id"],
            "service_s": (finished - started) / 1e9,
            "finished_ns": finished,
        }

    rows, _ = run_threaded(items, concurrency=2, execute=execute)

    assert len(rows) == len(items)
    assert max(row["response_s"] for row in rows) < 0.04
    assert max(row["submitted_ns"] for row in rows) - min(row["submitted_ns"] for row in rows) > 10_000_000
