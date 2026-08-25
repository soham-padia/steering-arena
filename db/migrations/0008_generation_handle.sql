-- 0008: show who made each generation, using the same handle rules as the leaderboard.
--
-- The public feed needs an author, but the account's email must never be it. Players
-- pick a handle exactly as they do when submitting a sequence (app/submission.py:
-- validate_handle — 1-32 chars, letters/digits/space/_-.), and that is what appears.
-- The link from handle to account stays server-side in user_hash, which is not exported
-- and not in any public response.

alter table generation_events add column if not exists handle text;
