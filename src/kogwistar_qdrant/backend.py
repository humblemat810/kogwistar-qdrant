from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import QdrantClient, models

COLLECTIONS = (
    "node_index", "node", "edge", "edge_endpoints", "document", "domain",
    "node_docs", "node_refs", "edge_refs",
)
DOCUMENT_KEY = "__gke_document"
ID_KEY = "__gke_id"
DIMENSION = 3
SENTINEL = [0.0] * DIMENSION


class NoopUnitOfWork:
    @contextmanager
    def transaction(self) -> Iterator[None]:
        yield


def _condition(key: str, value: Any) -> models.Condition:
    if not isinstance(value, Mapping):
        return models.FieldCondition(key=key, match=models.MatchValue(value=value))
    parts: list[models.Condition] = []
    for op, item in value.items():
        if op == "$in":
            parts.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(item))))
        elif op in {"$gt", "$gte", "$lt", "$lte"}:
            field = op[1:]
            parts.append(models.FieldCondition(key=key, range=models.Range(**{field: item})))
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
    return models.Filter(must=must or None, must_not=must_not or None, should=should or None)


class QdrantBackend:
    """Chroma-shaped adapter over one Qdrant collection per logical collection.

    Non-vector materializations use a sentinel vector only because one Qdrant
    collection has one vector schema. Their query path is filtered scroll, not
    similarity search. Event log/Postgres remains authoritative.
    """

    supports_transactions = False
    consistency = "eventual"

    def __init__(self, client: QdrantClient, *, prefix: str = "kogwistar", dimension: int = DIMENSION):
        self.client, self.prefix, self.dimension = client, prefix, dimension
        self.uow = NoopUnitOfWork()
        self._ensure_collections()

    @classmethod
    def local(cls, path: str | None = None, **kwargs: Any) -> "QdrantBackend":
        client = QdrantClient(location=":memory:") if path is None else QdrantClient(path=path)
        return cls(client, **kwargs)

    @classmethod
    def remote(cls, url: str, **kwargs: Any) -> "QdrantBackend":
        return cls(QdrantClient(url=url), **kwargs)

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
            vectors.append(list(point.vector) if point.vector is not None and not isinstance(point.vector, dict) else None)
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
        return self._flat(points, inc)

    def query(self, key: str, *, query_embeddings: Sequence[Sequence[float]] | None = None, n_results: int = 10, where: Mapping[str, Any] | None = None, include: Sequence[str] | None = None) -> dict[str, Any]:
        inc = self._include(include) | {"documents", "metadatas"}
        if query_embeddings is None:
            flat = self._flat(self._scroll(key, where, n_results, inc), inc)
            return {name: [value] for name, value in flat.items()}
        batches = [self.client.query_points(collection_name=self._name(key), query=list(vector), query_filter=_filter(where), limit=n_results, with_payload=True, with_vectors=("embeddings" in inc)).points for vector in query_embeddings]
        out: dict[str, Any] = {"ids": [[self._external_id(point) for point in batch] for batch in batches]}
        if "documents" in inc:
            out["documents"] = [[self._payload(point)[0] for point in batch] for batch in batches]
        if "metadatas" in inc:
            out["metadatas"] = [[self._payload(point)[1] for point in batch] for batch in batches]
        if "distances" in inc:
            out["distances"] = [[1.0 - float(point.score) for point in batch] for batch in batches]
        return out

    def upsert(self, key: str, *, ids: Sequence[str], documents: Sequence[str], metadatas: Sequence[Mapping[str, Any]], embeddings: Sequence[Sequence[float]] | None = None) -> None:
        vectors = embeddings or [[0.0] * self.dimension for _ in ids]
        points = [self._point(i, d, m, v) for i, d, m, v in zip(ids, documents, metadatas, vectors, strict=True)]
        self.client.upsert(collection_name=self._name(key), points=points, wait=True)

    add = upsert

    def update(self, key: str, *, ids: Sequence[str], documents: Sequence[str | None] | None = None, metadatas: Sequence[Mapping[str, Any]] | None = None, embeddings: Sequence[Sequence[float]] | None = None) -> None:
        old = self.get(key, ids=ids, include=["documents", "metadatas", "embeddings"])
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

    def delete(self, key: str, *, ids: Sequence[str] | None = None, where: Mapping[str, Any] | None = None) -> None:
        target_ids = [self._provider_id(id_) for id_ in ids] if ids is not None else [str(point.id) for point in self._scroll(key, where, 10000, set())]
        if target_ids:
            self.client.delete(collection_name=self._name(key), points_selector=models.PointIdsList(points=target_ids), wait=True)

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
