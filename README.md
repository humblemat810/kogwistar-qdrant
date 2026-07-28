# kogwistar-qdrant

Standalone Qdrant adapter for Kogwistar's Chroma-shaped backend surface.

## Local test

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest
```

Tests also cover Qdrant on-disk tmpdir reopen and a SQLite SQL-meta/outbox
harness: rollback, durable pending job, requeue/retry, commit, and idempotent
upsert replay. The harness is contract coverage, not the production Kogwistar
meta schema.

For server integration:

```powershell
docker compose up -d qdrant
$env:QDRANT_URL = "http://127.0.0.1:6333"
.\.venv\Scripts\python.exe -m pytest -m integration
docker compose down
```

Qdrant is an eventually-consistent projection here. `transaction()` is a no-op;
authoritative graph/event state must remain in Kogwistar's transactional store.
