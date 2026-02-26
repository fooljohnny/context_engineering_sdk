"""Cross-session user memory store: persists user-level data independently of sessions."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class UserMemory:
    """A single user memory entry that persists across sessions."""

    memory_id: str
    user_id: str
    category: str  # "preference", "profile", "fact", "session_summary"
    content: str
    metadata: dict = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    source_session_id: str = ""


class UserMemoryStore(Protocol):
    """Protocol for cross-session user memory persistence."""

    async def save_memory(self, memory: UserMemory) -> str: ...

    async def get_memories(
        self,
        user_id: str,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[UserMemory]: ...

    async def get_memory(self, user_id: str, memory_id: str) -> UserMemory | None: ...

    async def update_memory(
        self, user_id: str, memory_id: str, content: str, updated_at: str = ""
    ) -> bool: ...

    async def delete_memory(self, user_id: str, memory_id: str) -> bool: ...

    async def list_users(self) -> list[str]: ...


class InMemoryUserMemoryStore:
    """In-memory implementation for testing and prototyping."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, UserMemory]] = {}

    async def save_memory(self, memory: UserMemory) -> str:
        if memory.user_id not in self._store:
            self._store[memory.user_id] = {}
        self._store[memory.user_id][memory.memory_id] = copy.deepcopy(memory)
        return memory.memory_id

    async def get_memories(
        self,
        user_id: str,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[UserMemory]:
        user_data = self._store.get(user_id, {})
        results = list(user_data.values())
        if category:
            results = [m for m in results if m.category == category]
        results.sort(key=lambda m: m.updated_at or m.created_at, reverse=True)
        if limit:
            results = results[:limit]
        return [copy.deepcopy(m) for m in results]

    async def get_memory(self, user_id: str, memory_id: str) -> UserMemory | None:
        user_data = self._store.get(user_id, {})
        mem = user_data.get(memory_id)
        return copy.deepcopy(mem) if mem else None

    async def update_memory(
        self, user_id: str, memory_id: str, content: str, updated_at: str = ""
    ) -> bool:
        user_data = self._store.get(user_id, {})
        mem = user_data.get(memory_id)
        if mem is None:
            return False
        mem.content = content
        if updated_at:
            mem.updated_at = updated_at
        return True

    async def delete_memory(self, user_id: str, memory_id: str) -> bool:
        user_data = self._store.get(user_id, {})
        if memory_id in user_data:
            del user_data[memory_id]
            return True
        return False

    async def list_users(self) -> list[str]:
        return list(self._store.keys())


class FileUserMemoryStore:
    """File-based user memory store, one JSON file per user."""

    def __init__(self, base_dir: str | Path) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _user_path(self, user_id: str) -> Path:
        safe_id = user_id.replace("/", "_").replace("\\", "_")
        return self._base_dir / f"user_{safe_id}.json"

    def _load(self, user_id: str) -> dict[str, dict]:
        path = self._user_path(user_id)
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, user_id: str, data: dict[str, dict]) -> None:
        path = self._user_path(user_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _to_dict(mem: UserMemory) -> dict:
        return {
            "memory_id": mem.memory_id,
            "user_id": mem.user_id,
            "category": mem.category,
            "content": mem.content,
            "metadata": mem.metadata,
            "created_at": mem.created_at,
            "updated_at": mem.updated_at,
            "source_session_id": mem.source_session_id,
        }

    @staticmethod
    def _from_dict(d: dict) -> UserMemory:
        return UserMemory(
            memory_id=d["memory_id"],
            user_id=d["user_id"],
            category=d.get("category", ""),
            content=d.get("content", ""),
            metadata=d.get("metadata", {}),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            source_session_id=d.get("source_session_id", ""),
        )

    async def save_memory(self, memory: UserMemory) -> str:
        data = self._load(memory.user_id)
        data[memory.memory_id] = self._to_dict(memory)
        self._save(memory.user_id, data)
        return memory.memory_id

    async def get_memories(
        self,
        user_id: str,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[UserMemory]:
        data = self._load(user_id)
        results = [self._from_dict(d) for d in data.values()]
        if category:
            results = [m for m in results if m.category == category]
        results.sort(key=lambda m: m.updated_at or m.created_at, reverse=True)
        if limit:
            results = results[:limit]
        return results

    async def get_memory(self, user_id: str, memory_id: str) -> UserMemory | None:
        data = self._load(user_id)
        d = data.get(memory_id)
        return self._from_dict(d) if d else None

    async def update_memory(
        self, user_id: str, memory_id: str, content: str, updated_at: str = ""
    ) -> bool:
        data = self._load(user_id)
        if memory_id not in data:
            return False
        data[memory_id]["content"] = content
        if updated_at:
            data[memory_id]["updated_at"] = updated_at
        self._save(user_id, data)
        return True

    async def delete_memory(self, user_id: str, memory_id: str) -> bool:
        data = self._load(user_id)
        if memory_id not in data:
            return False
        del data[memory_id]
        self._save(user_id, data)
        return True

    async def list_users(self) -> list[str]:
        users = []
        for path in self._base_dir.glob("user_*.json"):
            user_id = path.stem.removeprefix("user_")
            users.append(user_id)
        return users
