import sys

import pytest

pytestmark = pytest.mark.skipif(sys.implementation.name != "pypy", reason="PyPy compatibility lane only")


def test_provider_free_import_and_contract_surface():
    from kogwistar_qdrant import QdrantBackend

    backend = object.__new__(QdrantBackend)
    assert backend.__class__.__name__ == "QdrantBackend"
    assert hasattr(QdrantBackend, "remote")
