from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from dataclasses import fields
from pathlib import Path

from .config import DisturbanceModelConfig
from .models import DisturbanceSample, ModelMetrics


class DisturbanceStorage:
    def __init__(self, config: DisturbanceModelConfig, logger=None) -> None:
        self.config = config
        self._log = logger or (lambda _msg: None)
        self._queue: queue.Queue[DisturbanceSample | None] = queue.Queue(maxsize=int(config.storage_queue_size))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._init_db()

    @property
    def database_path(self) -> str:
        return self.config.database_path

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._writer_loop, name="disturbance-storage", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=2.0)

    def submit(self, sample: DisturbanceSample) -> bool:
        try:
            self._queue.put_nowait(sample)
            return True
        except queue.Full:
            self._log("[DISTURBANCE][STORAGE] queue full; sample dropped")
            return False

    def count_samples(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM disturbance_samples").fetchone()[0])

    def load_recent_samples(self, limit: int) -> list[DisturbanceSample]:
        names = [field.name for field in fields(DisturbanceSample)]
        sql = f"SELECT {','.join(names)} FROM disturbance_samples ORDER BY timestamp DESC LIMIT ?"
        with self._connect() as conn:
            rows = conn.execute(sql, (int(limit),)).fetchall()
        samples = [DisturbanceSample(**dict(zip(names, row))) for row in rows]
        samples.reverse()
        return samples

    def record_model_version(self, version: str, payload: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO model_versions(version, created_at, payload_json) VALUES(?,?,?)",
                (version, time.time(), json.dumps(payload, ensure_ascii=False)),
            )

    def record_metrics(self, version: str, metrics: ModelMetrics) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO model_metrics(version, timestamp, mae, rmse, r2, direction_accuracy, response_delay_error_ms) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    version,
                    time.time(),
                    metrics.mae,
                    metrics.rmse,
                    metrics.r2,
                    metrics.direction_accuracy,
                    metrics.response_delay_error_ms,
                ),
            )

    def _writer_loop(self) -> None:
        batch: list[DisturbanceSample] = []
        last_flush = time.time()
        while not self._stop_event.is_set():
            timeout = max(0.05, float(self.config.storage_flush_interval_s))
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                item = None
            if item is not None:
                batch.append(item)
            if batch and (len(batch) >= int(self.config.storage_batch_size) or time.time() - last_flush >= timeout or item is None):
                self._insert_batch(batch)
                batch.clear()
                last_flush = time.time()
            if item is None and self._stop_event.is_set():
                break
        if batch:
            self._insert_batch(batch)

    def _insert_batch(self, batch: list[DisturbanceSample]) -> None:
        names = [field.name for field in fields(DisturbanceSample)]
        placeholders = ",".join("?" for _ in names)
        sql = f"INSERT INTO disturbance_samples({','.join(names)}) VALUES({placeholders})"
        rows = [tuple(sample.to_dict().get(name) for name in names) for sample in batch]
        with self._connect() as conn:
            conn.executemany(sql, rows)

    def _init_db(self) -> None:
        Path(self.config.database_path).parent.mkdir(parents=True, exist_ok=True)
        names = [field.name for field in fields(DisturbanceSample)]
        columns = ", ".join(f"{name} {self._sqlite_type(name)}" for name in names)
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS experiments(id TEXT PRIMARY KEY, created_at REAL, note TEXT)")
            conn.execute(f"CREATE TABLE IF NOT EXISTS disturbance_samples(id INTEGER PRIMARY KEY AUTOINCREMENT, {columns})")
            conn.execute("CREATE TABLE IF NOT EXISTS model_versions(version TEXT PRIMARY KEY, created_at REAL, payload_json TEXT)")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS model_metrics("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, version TEXT, timestamp REAL, mae REAL, rmse REAL, r2 REAL, "
                "direction_accuracy REAL, response_delay_error_ms REAL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS disturbance_events("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL, experiment_id TEXT, name TEXT, stage TEXT, amplitude REAL)"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.config.database_path, timeout=1.0)
        conn.execute("PRAGMA journal_mode=MEMORY")
        conn.execute("PRAGMA temp_store=MEMORY")
        return conn

    @staticmethod
    def _sqlite_type(name: str) -> str:
        if name.endswith("_status") or name in {"vision_valid", "feedback_frozen"}:
            return "INTEGER"
        if name in {"experiment_id", "chip_id", "disturbance_name", "disturbance_stage", "run_state", "video_source_type", "vision_invalid_reason", "freeze_reason"}:
            return "TEXT"
        return "REAL"
