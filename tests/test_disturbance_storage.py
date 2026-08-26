from __future__ import annotations

from backend.disturbance_model.config import DisturbanceModelConfig
from backend.disturbance_model.models import DisturbanceSample
from backend.disturbance_model.storage import DATABASE_SCHEMA_VERSION, DisturbanceStorage


def test_storage_stop_drains_queue_and_sets_schema_version(tmp_path) -> None:
    database = tmp_path / "disturbance.sqlite3"
    config = DisturbanceModelConfig(
        database_path=str(database),
        model_path=str(tmp_path / "model.json"),
        storage_queue_size=20,
        storage_batch_size=50,
        storage_flush_interval_s=60.0,
    )
    storage = DisturbanceStorage(config)
    storage.start()
    for index in range(7):
        assert storage.submit(DisturbanceSample(timestamp=float(index)))
    storage.stop()

    assert storage.count_samples() == 7
    with storage._connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == DATABASE_SCHEMA_VERSION
