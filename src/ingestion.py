import os

import sys
sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config


def ingest_to_chroma(documents, ids, embeddings, db_name="pubmed_chroma", persist_dir=None):
    from langchain_chroma import Chroma

    persist_dir = persist_dir or str(config.VECTORSTORE_DIR / db_name)

    vector_store = Chroma(
        collection_name=db_name,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        batch_size = 5000
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            vector_store.add_documents(documents=batch_docs, ids=batch_ids)
            print(f"Ingested batch {i // batch_size + 1}: {len(batch_docs)} chunks")
        print(f"Total: {len(documents)} chunks ingested into ChromaDB")
    else:
        print(f"Loaded existing ChromaDB from {persist_dir}")

    return vector_store


def ingest_to_qdrant(documents, ids, embeddings, collection_name="pubmed_qdrant", path=None):
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    path = path or str(config.VECTORSTORE_DIR / collection_name)

    sample_embedding = embeddings.embed_query("test")
    embedding_dim = len(sample_embedding)

    client = QdrantClient(path=path)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    existing = client.get_collection(collection_name).points_count
    if existing == 0:
        vector_store.add_documents(documents=documents, ids=ids)
        print(f"Ingested {len(documents)} chunks into Qdrant")
    else:
        print(f"Loaded existing Qdrant collection ({existing} points)")

    return vector_store


def ingest_to_lancedb(documents, ids, embeddings, table_name="pubmed_lance", path=None):
    import lancedb

    path = path or str(config.VECTORSTORE_DIR / "lancedb")
    db = lancedb.connect(path)

    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    existing_tables = db.table_names()
    if table_name not in existing_tables:
        vectors = embeddings.embed_documents(texts)
        data = [
            {"id": doc_id, "text": text, "vector": vec, **meta}
            for doc_id, text, vec, meta in zip(ids, texts, vectors, metadatas)
        ]
        table = db.create_table(table_name, data=data)
        print(f"Ingested {len(documents)} chunks into LanceDB")
    else:
        table = db.open_table(table_name)
        print(f"Loaded existing LanceDB table '{table_name}' ({table.count_rows()} rows)")

    return db, table


def ingest_to_weaviate(documents, ids, embeddings, url=None, api_key=None):
    import weaviate
    from langchain_weaviate import WeaviateVectorStore

    url = url or os.getenv("WEAVIATE_URL", "")
    api_key = api_key or os.getenv("WEAVIATE_API_KEY", "")

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=weaviate.auth.AuthApiKey(api_key),
    )

    vector_store = WeaviateVectorStore(
        client=client,
        index_name="PubMedAbstracts",
        text_key="text",
        embedding=embeddings,
    )

    vector_store.add_documents(documents=documents, ids=ids)
    print(f"Ingested {len(documents)} chunks into Weaviate")

    return vector_store, client

def get_vector_store(db_type, documents, ids, embeddings, **kwargs):
    factories = {
        "chroma": ingest_to_chroma,
        "qdrant": ingest_to_qdrant,
        "lancedb": ingest_to_lancedb,
        "weaviate": ingest_to_weaviate,
    }
    if db_type not in factories:
        raise ValueError(f"Unknown db_type '{db_type}'. Choose from: {list(factories)}")
    return factories[db_type](documents, ids, embeddings, **kwargs)
