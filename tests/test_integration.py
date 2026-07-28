import os

import pytest

from kogwistar_qdrant import QdrantBackend


@pytest.mark.integration
def test_qdrant_server_contract():
    url = os.getenv("QDRANT_URL")
    if not url:
        pytest.skip("QDRANT_URL not set")
    try:
        backend = QdrantBackend.remote(url, prefix="integration")
    except Exception as exc:
        pytest.skip(f"Qdrant service unavailable: {exc}")
    backend.node_upsert(ids=["server-node"], documents=["server"], metadatas=[{"doc_id": "server-doc"}], embeddings=[[1, 0, 0]])
    assert backend.node_get(ids=["server-node"])["ids"] == ["server-node"]
    assert backend.node_query(query_embeddings=[[1, 0, 0]], n_results=1)["ids"][0] == ["server-node"]
