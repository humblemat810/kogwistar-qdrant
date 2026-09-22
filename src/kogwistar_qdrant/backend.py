from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Mapping, Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import QdrantClient, models

try:
    from kogwistar.engine_core.embedding_profile import EmbeddingStorageState
    from kogwistar.engine_core.storage_backend import TwoStageProjectionCapability
except ImportError:
    @dataclass(frozen=True)
    class EmbeddingStorageState:
        backend_kind: str
        storage_scope: str
        persistent: bool
        vector_count: int
        details: tuple[str, ...] = ()

COLLECTIONS = (
    "node_index", "node", "edge", "edge_endpoints", "document", "domain",
    "node_docs", "node_refs", "edge_refs",
)
DOCUMENT_KEY = "__gke_document"
ID_KEY = "__gke_id"
PENDING_KEY = "__gke_embedding_pending"
DIMENSION = 3
SENTINEL = [0.0] * DIMENSION


class NoopUnitOfWork:
    @contextmanager
    def transaction(self) -> Iterator[None]:
        yield


class AsyncNoopUnitOfWork:
    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        yield


try:
    from kogwistar.engine_core.storage_backend import TwoStageProjectionCapability
except ImportError:
    @dataclass(frozen=True)
    class TwoStageProjectionCapability:
        supports_two_stage: bool = False
        reason: str = "Qdrant adapter has no canonical event/revision promotion"


def _awaitable(value: Any) -> Any:
    class AwaitableValue:
        def __init__(self, value: Any) -> None:
            self.value = value
        def __await__(self):
            async def done() -> Any:
                return self.value
            return done().__await__()
        def __getattr__(self, name: str) -> Any:
            return getattr(self.value, name)
        def __getitem__(self, key: Any) -> Any:
            return self.value[key]
        def __iter__(self):
            return iter(self.value)
        def __len__(self) -> int:
            return len(self.value)
        def __eq__(self, other: Any) -> bool:
            return self.value == other
    return AwaitableValue(value)


def _condition(key: str, value: Any) -> models.Condition:
    if not isinstance(value, Mapping):
        return models.FieldCondition(key=key, match=models.MatchValue(value=value))
    parts: list[models.Condition] = []
    for op, item in value.items():
        if op == "$in":
            parts.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(item))))
        elif op == "$eq":
            parts.append(models.FieldCondition(key=key, match=models.MatchValue(value=item)))
        elif op in {"$gt", "$gte", "$lt", "$lte"}:
            field = op[1:]
            parts.append(models.FieldCondition(key=key, range=models.Range(**{field: item})))
        elif op == "$contains":
            parts.append(models.FieldCondition(key=key, match=models.MatchValue(value=item)))
        else:
            raise ValueError(f"unsupported where operator: {op}")
    if len(parts) != 1:
        raise ValueError(f"compound field predicate unsupported: {value}")
    return parts[0]


def _filter(where: Mapping[str, Any] | None) -> models.Filter | None:
    if not where:
        return None
    must: list[Any] = []
    must_not: list[Any] = []
    should: list[Any] = []
    for key, value in where.items():
        if key == "$and":
            must.extend([nested for item in value if (nested := _filter(item))])
        elif key == "$or":
            should.extend([nested for item in value if (nested := _filter(item))])
        elif isinstance(value, Mapping) and "$nin" in value:
            must_not.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(value["$nin"]))))
        elif isinstance(value, Mapping) and "$ne" in value:
            must_not.append(models.FieldCondition(key=key, match=models.MatchValue(value=value["$ne"])))
        else:
            must.append(_condition(key, value))
    must_not.append(models.FieldCondition(key=PENDING_KEY, match=models.MatchValue(value=True)))
    return models.Filter(must=must or None, must_not=must_not or None, should=should or None)


class QdrantBackend:
    """Chroma-shaped adapter over one Qdrant collection per logical collection.

    Non-vector materializations use a sentinel vector only because one Qdrant
    collection has one vector schema. Their query path is filtered scroll, not
    similarity search. Event log/Postgres remains authoritative.
    """

    supports_transactions = False
    consistency = "eventual"

    def __init__(self, client: QdrantClient, *, prefix: str = "kogwistar", dimension: int = DIMENSION, storage_scope: str | None = None, persistent: bool = True):
        self.client, self.prefix, self.dimension = client, prefix, dimension
        self.uow = NoopUnitOfWork()
        self.unit_of_work = self.uow
        self.async_unit_of_work = AsyncNoopUnitOfWork()
        self.supports_two_stage = False
        self.two_stage_projection_capability = TwoStageProjectionCapability()
        self._storage_scope = storage_scope or f"qdrant:{prefix}"
        self._persistent = persistent
        self._ensure_collections()

    @classmethod
    def local(cls, path: str | None = None, **kwargs: Any) -> "QdrantBackend":
        client = QdrantClient(location=":memory:") if path is None else QdrantClient(path=path)
        scope = f"qdrant:memory:{kwargs.get('prefix', 'kogwistar')}" if path is None else f"qdrant:path:{hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:16]}"
        return cls(client, storage_scope=scope, persistent=path is not None, **kwargs)

    @classmethod
    def remote(cls, url: str, **kwargs: Any) -> "QdrantBackend":
        scope = f"qdrant:url:{hashlib.sha256(url.encode()).hexdigest()[:16]}"
        return cls(QdrantClient(url=url), storage_scope=scope, persistent=True, **kwargs)

    def embedding_storage_scope(self) -> str:
        return self._storage_scope

    def embedding_storage_scope_aliases(self) -> tuple[str, ...]:
        return ()

    def inspect_embedding_storage(self) -> dict[str, Any]:
        counts = {key: int(self.client.count(self._name(key), exact=True).count) for key in ("node_index", "node", "edge", "document", "domain")}
        return EmbeddingStorageState(backend_kind="qdrant", storage_scope=self._storage_scope, persistent=self._persistent, vector_count=sum(counts.values()), details=tuple(f"{key}={count}" for key, count in counts.items()))

    def _name(self, key: str) -> str:
        return f"{self.prefix}_{key}"

    def _ensure_collections(self) -> None:
        for key in COLLECTIONS:
            name = self._name(key)
            if not self.client.collection_exists(name):
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(size=self.dimension, distance=models.Distance.COSINE),
                )

    @staticmethod
    def _include(include: Sequence[str] | None) -> set[str]:
        return set(include or ("documents", "metadatas"))

    def _point(self, id_: str, document: str, metadata: Mapping[str, Any], vector: Sequence[float] | None) -> models.PointStruct:
        payload = dict(metadata)
        payload[DOCUMENT_KEY] = document
        payload[ID_KEY] = id_
        if vector is None:
            payload[PENDING_KEY] = True
        return models.PointStruct(id=str(uuid5(NAMESPACE_URL, f"kogwistar:{id_}")), vector=list(vector or [0.0] * self.dimension), payload=payload)

    @staticmethod
    def _provider_id(id_: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"kogwistar:{id_}"))

    def _scroll(self, key: str, where: Mapping[str, Any] | None, limit: int, include: set[str]) -> list[Any]:
        points, _ = self.client.scroll(
            collection_name=self._name(key), scroll_filter=_filter(where), limit=limit,
            with_payload=True, with_vectors=("embeddings" in include),
        )
        return points

    @staticmethod
    def _payload(point: Any) -> tuple[str | None, dict[str, Any]]:
        payload = dict(point.payload or {})
        document = payload.pop(DOCUMENT_KEY, None)
        payload.pop(ID_KEY, None)
        payload.pop(PENDING_KEY, None)
        return document, payload

    @staticmethod
    def _external_id(point: Any) -> str:
        payload = point.payload or {}
        return str(payload.get(ID_KEY, point.id))

    def _flat(self, points: Sequence[Any], include: set[str]) -> dict[str, Any]:
        docs, metas, vectors = [], [], []
        for point in points:
            document, metadata = self._payload(point)
            docs.append(document)
            metas.append(metadata)
            pending = bool((point.payload or {}).get(PENDING_KEY))
            vectors.append(None if pending else (list(point.vector) if point.vector is not None and not isinstance(point.vector, dict) else None))
        out: dict[str, Any] = {"ids": [self._external_id(point) for point in points]}
        if "documents" in include:
            out["documents"] = docs
        if "metadatas" in include:
            out["metadatas"] = metas
        if "embeddings" in include:
            out["embeddings"] = vectors
        return out

    def get(self, key: str, *, ids: Sequence[str] | None = None, where: Mapping[str, Any] | None = None, include: Sequence[str] | None = None, limit: int = 200) -> dict[str, Any]:
        inc = self._include(include)
        provider_ids = [self._provider_id(id_) for id_ in ids] if ids is not None else None
        points = self.client.retrieve(collection_name=self._name(key), ids=provider_ids, with_payload=True, with_vectors=("embeddings" in inc)) if provider_ids is not None else self._scroll(key, where, limit, inc)
        result = self._flat(points, inc)
        if provider_ids is not None:
            order = {id_: n for n, id_ in enumerate(result["ids"])}
            indexes = [order[id_] for id_ in ids if id_ in order]
            for name, values in list(result.items()):
                if name != "ids":
                    result[name] = [values[n] for n in indexes]
            result["ids"] = [result["ids"][n] for n in indexes]
        return _awaitable(result)

    def query(self, key: str, *, query_embeddings: Sequence[Sequence[float]] | None = None, n_results: int = 10, where: Mapping[str, Any] | None = None, include: Sequence[str] | None = None) -> dict[str, Any]:
        inc = self._include(include) | {"documents", "metadatas"}
        if query_embeddings is None:
            flat = self._flat(self._scroll(key, where, n_results, inc), inc)
            return _awaitable({name: [value] for name, value in flat.items()})
        semantic_where = dict(where or {})
        semantic_where[PENDING_KEY] = {"$ne": True}
        batches = [self.client.query_points(collection_name=self._name(key), query=list(vector), query_filter=_filter(semantic_where), limit=n_results, with_payload=True, with_vectors=("embeddings" in inc)).points for vector in query_embeddings]
        out: dict[str, Any] = {"ids": [[self._external_id(point) for point in batch] for batch in batches]}
        if "documents" in inc:
            out["documents"] = [[self._payload(point)[0] for point in batch] for batch in batches]
        if "metadatas" in inc:
            out["metadatas"] = [[self._payload(point)[1] for point in batch] for batch in batches]
        if "distances" in inc:
            out["distances"] = [[1.0 - float(point.score) for point in batch] for batch in batches]
        return _awaitable(out)

    def upsert(self, key: str, *, ids: Sequence[str], documents: Sequence[str], metadatas: Sequence[Mapping[str, Any]], embeddings: Sequence[Sequence[float]] | None = None) -> None:
        vectors = list(embeddings) if embeddings is not None else [None] * len(ids)
        points = [self._point(i, d, m, v) for i, d, m, v in zip(ids, documents, metadatas, vectors, strict=True)]
        return _awaitable(self.client.upsert(collection_name=self._name(key), points=points, wait=True))

    add = upsert

    def update(self, key: str, *, ids: Sequence[str], documents: Sequence[str | None] | None = None, metadatas: Sequence[Mapping[str, Any]] | None = None, embeddings: Sequence[Sequence[float]] | None = None) -> None:
        old = self.get(key, ids=ids, include=["documents", "metadatas", "embeddings"])
        if hasattr(old, "value"):
            old = old.value
        positions = {id_: n for n, id_ in enumerate(old["ids"])}
        for n, id_ in enumerate(ids):
            if id_ not in positions:
                continue
            old_n = positions[id_]
            metadata = dict(old["metadatas"][old_n] or {})
            if metadatas is not None:
                metadata.update(metadatas[n])
            document = documents[n] if documents is not None and documents[n] is not None else old["documents"][old_n]
            vector = embeddings[n] if embeddings is not None else old["embeddings"][old_n]
            self.upsert(key, ids=[id_], documents=[document], metadatas=[metadata], embeddings=[vector])
        return _awaitable(None)

    def delete(self, key: str, *, ids: Sequence[str] | None = None, where: Mapping[str, Any] | None = None) -> None:
        target_ids = [self._provider_id(id_) for id_ in ids] if ids is not None else [str(point.id) for point in self._scroll(key, where, 10000, set())]
        if target_ids:
            return _awaitable(self.client.delete(collection_name=self._name(key), points_selector=models.PointIdsList(points=target_ids), wait=True))
        return _awaitable(None)

    def call(self, collection_key: str, method: str, **kwargs: Any) -> Any:
        if collection_key not in COLLECTIONS or method not in {"get", "query", "add", "upsert", "update", "delete"}:
            raise ValueError(f"unsupported collection/method: {collection_key}.{method}")
        return getattr(self, f"{collection_key}_{method}")(**kwargs)

    def __getattr__(self, name: str) -> Any:
        for key in COLLECTIONS:
            if name.startswith(key + "_") and name[len(key) + 1:] in {"get", "query", "add", "upsert", "update", "delete"}:
                method = name[len(key) + 1:]
                return lambda **kwargs: getattr(self, method)(key, **kwargs)
        raise AttributeError(name)
