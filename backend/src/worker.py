import logging
import shutil
from pathlib import Path
from threading import Event, Thread
from time import sleep

from src.config.settings import ARTIFACTS, cache, engine
from src.services.domain import solve
from src.services.workflow import (AttemptCheckpoint, FencedError, claim,
                                   finish, sweep_run_artifacts)

HEARTBEAT = Path(ARTIFACTS) / 'worker-heartbeat'


def heartbeat(stop):
    while not stop.is_set():
        HEARTBEAT.touch()
        stop.wait(2)


def execute(run_id, token, payload):
    directory = Path(ARTIFACTS) / 'runs' / str(run_id)
    store = AttemptCheckpoint(run_id, token)
    committed = False
    try:
        result = solve(payload, engine, directory=directory, checkpointer=store)
        committed = finish(run_id, token, result=result)
    except FencedError as exc:
        # 执行权已被接手：进度以接手方为准，本次尝试静默放弃。
        logging.info('Run %s attempt superseded: %s', run_id, exc)
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        committed = finish(run_id, token, error=str(exc)[:500])
    except Exception:
        logging.exception('Domain run failed: %s', run_id)
        committed = finish(run_id, token, error='后台执行失败，请检查输入或查看 worker 日志。')
    if committed:
        shutil.rmtree(directory, ignore_errors=True)


def work():
    HEARTBEAT.unlink(missing_ok=True)
    sweep_run_artifacts()
    while True:
        try:
            job = claim()
            HEARTBEAT.touch()
            if not job:
                cache.brpop('worker:wake', timeout=1)
                continue
            run_id, token, payload = job
            stop = Event()
            thread = Thread(target=heartbeat, args=(stop,), daemon=True)
            thread.start()
            try:
                execute(run_id, token, payload)
            finally:
                stop.set()
                thread.join()
        except Exception:
            logging.exception('Worker loop failure')
            sleep(2)


if __name__ == '__main__':
    work()
