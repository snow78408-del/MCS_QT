from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

from backend.runtime_paths import user_data_dir

from .algorithms import get_algorithm


PROFILE_SCHEMA_VERSION = 1
BUILTIN_PROFILE_ID = "builtin-hybrid-v1"


@dataclass(frozen=True)
class AlgorithmProfile:
    profile_id: str
    name: str
    plugin_id: str
    parameters: dict[str, Any]
    protected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AlgorithmProfileStore:
    """Atomic user-data store for named algorithm configurations."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path is not None else user_data_dir() / "config" / "droplet_algorithms.json"
        self._profiles: dict[str, AlgorithmProfile] = {}
        self._active_profile_id = BUILTIN_PROFILE_ID
        self._load()

    def profiles(self) -> tuple[AlgorithmProfile, ...]:
        return tuple(self._profiles.values())

    def get(self, profile_id: str) -> AlgorithmProfile:
        try:
            return self._profiles[str(profile_id)]
        except KeyError as exc:
            raise ValueError(f"算法不存在：{profile_id}") from exc

    def active_profile(self) -> AlgorithmProfile:
        return self.get(self._active_profile_id)

    @property
    def active_profile_id(self) -> str:
        return self._active_profile_id

    def create(self, name: str, plugin_id: str, parameters: dict[str, Any] | None = None) -> AlgorithmProfile:
        normalized = self._validate_name(name)
        self._ensure_unique_name(normalized)
        plugin = get_algorithm(plugin_id)
        config = plugin.build_config(parameters)
        profile = AlgorithmProfile(
            profile_id=uuid4().hex,
            name=normalized,
            plugin_id=plugin.plugin_id,
            parameters=plugin.serialize_config(config),
        )
        self._profiles[profile.profile_id] = profile
        self._save()
        return profile

    def duplicate(self, profile_id: str, name: str) -> AlgorithmProfile:
        source = self.get(profile_id)
        return self.create(name, source.plugin_id, source.parameters)

    def update_parameters(self, profile_id: str, parameters: dict[str, Any]) -> AlgorithmProfile:
        current = self.get(profile_id)
        if current.protected:
            raise ValueError("内置算法为只读；请先复制后再调参")
        plugin = get_algorithm(current.plugin_id)
        updated = AlgorithmProfile(
            current.profile_id,
            current.name,
            current.plugin_id,
            plugin.serialize_config(plugin.build_config(parameters)),
            False,
        )
        self._profiles[profile_id] = updated
        self._save()
        return updated

    def rename(self, profile_id: str, name: str) -> AlgorithmProfile:
        current = self.get(profile_id)
        if current.protected:
            raise ValueError("内置算法不能重命名")
        normalized = self._validate_name(name)
        self._ensure_unique_name(normalized, excluding=profile_id)
        updated = AlgorithmProfile(current.profile_id, normalized, current.plugin_id, current.parameters, False)
        self._profiles[profile_id] = updated
        self._save()
        return updated

    def delete(self, profile_id: str) -> None:
        current = self.get(profile_id)
        if current.protected:
            raise ValueError("内置算法不能删除")
        if profile_id == self._active_profile_id:
            raise ValueError("正在运行的算法不能删除；请先启用其他算法")
        del self._profiles[profile_id]
        self._save()

    def activate(self, profile_id: str) -> AlgorithmProfile:
        profile = self.get(profile_id)
        self._active_profile_id = profile.profile_id
        self._save()
        return profile

    def export_profile(self, profile_id: str, path: str | os.PathLike[str]) -> None:
        profile = self.get(profile_id)
        Path(path).write_text(
            json.dumps(
                {"schema_version": PROFILE_SCHEMA_VERSION, "profile": profile.to_dict()},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def import_profile(self, path: str | os.PathLike[str], name: str | None = None) -> AlgorithmProfile:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != PROFILE_SCHEMA_VERSION:
            raise ValueError("算法文件版本不受支持")
        raw = payload.get("profile")
        if not isinstance(raw, dict):
            raise ValueError("算法文件缺少 profile")
        imported_name = str(name or raw.get("name") or "导入算法")
        if any(item.name.casefold() == imported_name.strip().casefold() for item in self.profiles()):
            imported_name = self._next_copy_name(imported_name)
        return self.create(imported_name, str(raw.get("plugin_id", "")), dict(raw.get("parameters") or {}))

    def next_copy_name(self, profile_id: str) -> str:
        return self._next_copy_name(f"{self.get(profile_id).name} 副本")

    def _load(self) -> None:
        builtin = self._builtin_profile()
        loaded: dict[str, AlgorithmProfile] = {}
        active = BUILTIN_PROFILE_ID
        for candidate in (self.path, self.path.with_suffix(self.path.suffix + ".bak")):
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                if payload.get("schema_version") != PROFILE_SCHEMA_VERSION:
                    continue
                for raw in payload.get("profiles", []):
                    if not isinstance(raw, dict) or raw.get("protected"):
                        continue
                    plugin = get_algorithm(str(raw.get("plugin_id", "")))
                    config = plugin.build_config(dict(raw.get("parameters") or {}))
                    profile = AlgorithmProfile(
                        str(raw["profile_id"]),
                        self._validate_name(str(raw["name"])),
                        plugin.plugin_id,
                        plugin.serialize_config(config),
                        False,
                    )
                    loaded[profile.profile_id] = profile
                active = str(payload.get("active_profile_id") or BUILTIN_PROFILE_ID)
                break
            except (FileNotFoundError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        self._profiles = {BUILTIN_PROFILE_ID: builtin, **loaded}
        self._active_profile_id = active if active in self._profiles else BUILTIN_PROFILE_ID

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": PROFILE_SCHEMA_VERSION,
                    "active_profile_id": self._active_profile_id,
                    "profiles": [item.to_dict() for item in self.profiles() if not item.protected],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        if self.path.is_file():
            shutil.copy2(self.path, self.path.with_suffix(self.path.suffix + ".bak"))
        os.replace(temporary, self.path)

    @staticmethod
    def _builtin_profile() -> AlgorithmProfile:
        plugin = get_algorithm("hybrid_v1")
        return AlgorithmProfile(
            BUILTIN_PROFILE_ID,
            "内置混合检测算法",
            plugin.plugin_id,
            plugin.default_parameters(),
            True,
        )

    @staticmethod
    def _validate_name(name: str) -> str:
        normalized = str(name).strip()
        if not normalized:
            raise ValueError("算法名称不能为空")
        if len(normalized) > 80:
            raise ValueError("算法名称不能超过 80 个字符")
        return normalized

    def _ensure_unique_name(self, name: str, excluding: str = "") -> None:
        if any(item.profile_id != excluding and item.name.casefold() == name.casefold() for item in self.profiles()):
            raise ValueError(f"算法名称已存在：{name}")

    def _next_copy_name(self, base: str) -> str:
        candidate = self._validate_name(base)
        if all(item.name.casefold() != candidate.casefold() for item in self.profiles()):
            return candidate
        index = 2
        while True:
            suffix = f" ({index})"
            candidate = f"{base[: max(1, 80 - len(suffix))]}{suffix}"
            if all(item.name.casefold() != candidate.casefold() for item in self.profiles()):
                return candidate
            index += 1
