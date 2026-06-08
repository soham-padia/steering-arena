"""Data access. Two implementations behind one duck-typed interface:

- `SupabaseDatabase` — production, via PostgREST (supabase-py).
- `InMemoryDatabase` — tests / local boot without Supabase.

Methods kept narrow and query-shaped so both backends stay simple.
"""

from __future__ import annotations

from datetime import datetime, timezone


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── In-memory (tests / no-Supabase local dev) ────────────────

class InMemoryDatabase:
    def __init__(self):
        self.seasons: list[dict] = []
        self.submissions: list[dict] = []
        self._sub_id = 0

    # seasons
    def get_active_season(self) -> dict | None:
        for s in self.seasons:
            if s.get("active"):
                return s
        return None

    def get_season(self, season_id: int) -> dict | None:
        return next((s for s in self.seasons if s["id"] == season_id), None)

    def add_season(self, **fields) -> dict:
        sid = fields.get("id", len(self.seasons) + 1)
        row = {"id": sid, "active": True, "scoring_mode": "cosine_steering_shift",
               "token_budget": 10, **fields}
        self.seasons.append(row)
        return row

    # submissions
    def leaderboard(self, season_id: int, limit: int, ascending: bool = False) -> list[dict]:
        rows = [s for s in self.submissions if s["season_id"] == season_id]
        rows.sort(key=lambda r: (r["score"] if ascending else -r["score"], r["created_at"]))
        return rows[:limit]

    def find_submission(self, season_id: int, norm_key: str) -> dict | None:
        return next(
            (s for s in self.submissions if s["season_id"] == season_id and s["norm_key"] == norm_key),
            None,
        )

    def insert_submission(self, row: dict) -> dict:
        self._sub_id += 1
        stored = {"id": self._sub_id, "created_at": _now().isoformat(), **row}
        self.submissions.append(stored)
        return stored

    def rank_for(self, season_id: int, score: float, higher_is_better: bool = True) -> int:
        if higher_is_better:
            better = sum(1 for s in self.submissions if s["season_id"] == season_id and s["score"] > score)
        else:
            better = sum(1 for s in self.submissions if s["season_id"] == season_id and s["score"] < score)
        return better + 1

    def count_ip_since(self, ip_hash: str, since_iso: str) -> int:
        return sum(
            1 for s in self.submissions
            if s.get("ip_hash") == ip_hash and s["created_at"] >= since_iso
        )

    def count_global_since(self, since_iso: str) -> int:
        return sum(1 for s in self.submissions if s["created_at"] >= since_iso)


# ── Supabase (production) ────────────────────────────────────

class SupabaseDatabase:
    def __init__(self, url: str, service_key: str):
        from supabase import create_client  # lazy: tests don't need supabase installed

        self.client = create_client(url, service_key)

    def get_active_season(self) -> dict | None:
        res = self.client.table("seasons").select("*").eq("active", True).limit(1).execute()
        return res.data[0] if res.data else None

    def get_season(self, season_id: int) -> dict | None:
        res = self.client.table("seasons").select("*").eq("id", season_id).limit(1).execute()
        return res.data[0] if res.data else None

    def leaderboard(self, season_id: int, limit: int, ascending: bool = False) -> list[dict]:
        res = (
            self.client.table("submissions")
            .select("user_handle, sequence_text, score, created_at")
            .eq("season_id", season_id)
            .order("score", desc=not ascending)
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
        return res.data or []

    def find_submission(self, season_id: int, norm_key: str) -> dict | None:
        res = (
            self.client.table("submissions")
            .select("*")
            .eq("season_id", season_id)
            .eq("norm_key", norm_key)
            .limit(1)
            .execute()
        )
        return res.data[0] if res.data else None

    def insert_submission(self, row: dict) -> dict:
        res = self.client.table("submissions").insert(row).execute()
        return res.data[0] if res.data else row

    def rank_for(self, season_id: int, score: float, higher_is_better: bool = True) -> int:
        q = (
            self.client.table("submissions")
            .select("id", count="exact", head=True)
            .eq("season_id", season_id)
        )
        q = q.gt("score", score) if higher_is_better else q.lt("score", score)
        return (q.execute().count or 0) + 1

    def count_ip_since(self, ip_hash: str, since_iso: str) -> int:
        res = (
            self.client.table("submissions")
            .select("id", count="exact", head=True)
            .eq("ip_hash", ip_hash)
            .gte("created_at", since_iso)
            .execute()
        )
        return res.count or 0

    def count_global_since(self, since_iso: str) -> int:
        res = (
            self.client.table("submissions")
            .select("id", count="exact", head=True)
            .gte("created_at", since_iso)
            .execute()
        )
        return res.count or 0
