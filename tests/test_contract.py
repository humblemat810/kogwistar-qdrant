from kogwistar_qdrant import QdrantBackend


def test_id_get_filter_vector_query_update_delete():
    backend = QdrantBackend.local(dimension=3)
    backend.node_add(
        ids=["n1", "n2"],
        documents=["alpha", "beta"],
        metadatas=[{"doc_id": "d1", "kind": "a"}, {"doc_id": "d1", "kind": "b"}],
        embeddings=[[1, 0, 0], [0, 1, 0]],
    )
    assert backend.node_get(ids=["n1"])["ids"] == ["n1"]
    assert set(backend.node_get(where={"doc_id": "d1"})["ids"]) == {"n1", "n2"}
    assert backend.node_query(query_embeddings=[[1, 0, 0]], n_results=1)["ids"][0] == ["n1"]
    backend.node_update(ids=["n1"], metadatas=[{"new": True}])
    assert backend.node_get(ids=["n1"])["metadatas"][0]["kind"] == "a"
    backend.node_delete(ids=["n2"])
    assert backend.node_get(ids=["n2"])["ids"] == []

