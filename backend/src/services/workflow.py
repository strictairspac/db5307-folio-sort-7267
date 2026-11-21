import hashlib
import shutil
import uuid
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import and_, or_, select

from src.config.settings import ARTIFACTS, cache, engine
from src.constants.workflow import Status
from src.models.tables import audit, cases, checkpoints, evidence, runs
from src.services import domain
from src.utils.encoding import canonical
from src.utils.time import utc_time


def log(conn, actor, action, reference):
    conn.execute(audit.insert().values(actor=actor, action=action, reference=str(reference), created_at=utc_time()))


def recent(table):
    with engine.connect() as conn:
        return [dict(row) for row in conn.execute(select(table).order_by(table.c.id.desc()).limit(100)).mappings()]


def found(conn, table, identity):
    row = conn.execute(select(table).where(table.c.id == identity)).mappings().first()
    if row is None:
        raise HTTPException(404, '找不到该记录')
    return dict(row)


def make_case(title, payload, actor):
    title = title.strip()
    if not title:
        raise HTTPException(422, '档案名称不能为空')
    try:
        if len(canonical(payload)) > 512 * 1024:
            raise ValueError('业务输入超过 512 KiB')
        if callable(getattr(domain, 'validate', None)):
            domain.validate(payload)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from None
    with engine.begin() as conn:
        case_id = conn.execute(cases.insert().values(slug=uuid.uuid4().hex, title=title, payload=payload,
                               author=actor, created_at=utc_time())).inserted_primary_key[0]
        log(conn, actor, 'create_case', case_id)
        return found(conn, cases, case_id)


def enqueue(case_id, actor):
    with engine.begin() as conn:
        found(conn, cases, case_id)
        run_id = conn.execute(runs.insert().values(case_id=case_id, status=Status.QUEUED,
                              owner=actor, created_at=utc_time())).inserted_primary_key[0]
        log(conn, actor, 'enqueue', run_id)
        result = found(conn, runs, run_id)
    try:
        cache.lpush('worker:wake', 'queued')
        cache.ltrim('worker:wake', 0, 99)
    except Exception:
        # The durable database queue is polled even if Redis is temporarily unavailable.
        pass
    return result


def claim():
    ready = or_(runs.c.status == Status.QUEUED,
                and_(runs.c.status == Status.RUNNING, runs.c.started_at < utc_time(-600)))
    with engine.begin() as conn:
        run = conn.execute(select(runs.c.id, runs.c.case_id).where(ready).order_by(runs.c.id)
                           .limit(1).with_for_update(skip_locked=True)).first()
        if not run:
            return None
        token = str(uuid.uuid4())
        conn.execute(runs.update().where(runs.c.id == run.id).values(
            started_at=utc_time(), status=Status.RUNNING, attempt_token=token))
        return run.id, token, found(conn, cases, run.case_id)['payload']


class FencedError(Exception):
    """本尝试的执行权已被接手方取代，禁止再写入任何进度。"""


class AttemptCheckpoint:
    """单个执行尝试的断点读写；提交与尝试令牌栅栏在同一事务内完成。

    提交成功即表示该轮已被当前执行权确认，同时刷新 started_at 充当心跳；
    执行权已被接手时提交被拒绝，旧后台无法改掉接手方确认的进度。
    """

    def __init__(self, run_id, token):
        self.run_id = run_id
        self.token = token
        self.tag = token

    def load(self):
        with engine.connect() as conn:
            row = conn.execute(select(checkpoints)
                               .where(checkpoints.c.run_id == self.run_id)).mappings().first()
        if row is None:
            return None
        return {'round': row['round'], 'runs': list(row['manifest']),
                'phases': list(row['phases'])}

    def commit(self, round_number, run_names, phases):
        values = {'round': round_number, 'token': self.token, 'manifest': list(run_names),
                  'phases': list(phases), 'updated_at': utc_time()}
        with engine.begin() as conn:
            count = conn.execute(runs.update().where(runs.c.id == self.run_id,
                                                     runs.c.attempt_token == self.token,
                                                     runs.c.status == Status.RUNNING)
                                 .values(started_at=utc_time())).rowcount
            if not count:
                raise FencedError(f'执行 {self.run_id} 已由接手的后台确认进度，本次尝试放弃写入')
            updated = conn.execute(checkpoints.update()
                                   .where(checkpoints.c.run_id == self.run_id)
                                   .values(**values)).rowcount
            if not updated:
                conn.execute(checkpoints.insert().values(run_id=self.run_id, **values))


def finish(run_id, token, result=None, error=None):
    digest = hashlib.sha256(canonical(result)).hexdigest() if result is not None else None
    status = Status.FAILED if error else Status.COMPLETE
    with engine.begin() as conn:
        count = conn.execute(runs.update().where(runs.c.id == run_id,
                        runs.c.attempt_token == token, runs.c.status == Status.RUNNING)
                        .values(status=status, finished_at=utc_time(), error=error)).rowcount
        if not count:
            return False
        conn.execute(checkpoints.delete().where(checkpoints.c.run_id == run_id))
        if result is not None:
            conn.execute(evidence.insert().values(run_id=run_id, report=result, digest=digest))
        log(conn, 'worker', status.value, run_id)
        return True


def sweep_run_artifacts():
    """清理已终结执行残留的段目录；排队或执行中的目录保留供续跑。"""
    root = Path(ARTIFACTS) / 'runs'
    if not root.is_dir():
        return
    with engine.connect() as conn:
        active = {row.id for row in conn.execute(
            select(runs.c.id).where(runs.c.status.in_([Status.QUEUED, Status.RUNNING])))}
    for child in root.iterdir():
        if child.is_dir() and child.name.isdigit() and int(child.name) not in active:
            shutil.rmtree(child, ignore_errors=True)


def review(identity, note, actor):
    note = note.strip()
    if not note:
        raise HTTPException(422, '复核意见不能为空')
    with engine.begin() as conn:
        found(conn, evidence, identity)
        count = conn.execute(evidence.update().where(evidence.c.id == identity, evidence.c.reviewed_at.is_(None))
                            .values(review_note=note, reviewer=actor, reviewed_at=utc_time())).rowcount
        if count != 1:
            raise HTTPException(409, '该证据已经复核')
        log(conn, actor, 'review', identity)
        return found(conn, evidence, identity)
