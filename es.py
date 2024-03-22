import os
from uuid import uuid4

from elasticsearch import Elasticsearch
import polars as pl
import openai

REVIEW_DATA_PATH = "data/reviews.csv"
data_reviews = pl.read_csv(REVIEW_DATA_PATH)

es = Elasticsearch(hosts="http://elastic:duynguyen@localhost:9200")
index_name = "review_vectors"
openai_client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def es_create_indices():
    res = es.indices.create(
        index="review_vectors",
        mappings={
            "properties": {
                "embedding": {
                    "type": "dense_vector",
                },
                "metadata": {"type": "object"},
                "content": {"type": "text"},
            }
        },
    )
    print(res)


def prepare_dataset():
    reviews = []
    payloads = []
    for row in data_reviews.rows(named=True):
        reviews.append(row["review"])
        payload = {
            "patient_name": row["patient_name"],
            "physician_name": row["physician_name"],
            "hospital_name": row["hospital_name"],
        }
        payloads.append(payload)

    embedding_vectors = openai_client.embeddings.create(
        input=reviews, model=embedding_model
    )
    return zip(embedding_vectors.data, reviews, payloads)


def es_bulk_upload():
    docs_ls = []
    for vector, review, payload in prepare_dataset():
        doc2 = {"content": review, "embedding": vector.embedding, "metadata": payload}
        doc1 = {"index": {"_index": "review_vectors", "_id": uuid4().hex}}
        docs_ls.append(doc1)
        docs_ls.append(doc2)
        if len(docs_ls) == 1024:
            es.bulk(operations=docs_ls, index=index_name)
            docs_ls = []

    if docs_ls:
        es.bulk(operations=docs_ls, index=index_name)


def es_insert(doc: dict):
    es.index(index=index_name, document=doc, id=uuid4().hex)


def es_knn_search(text_input: str):
    embedding_vector = openai_client.embeddings.create(
        input=text_input, model=embedding_model
    )
    page = es.search(
        index=index_name,
        fields=["content"],
        # query={
        #     "bool": {
        #         "must": [{"match": {"hospital_name": {"query": "LLC", "boost": 0.2}}}]
        #     }
        # },
        # query={
        #     "bool": {
        #         "should": [
        #             {"match": {"hospital_name": {"query": "LLC", "boost": 0.2}}},
        #             {"match": {"patient_name": {"query": "Johnson", "boost": 0.2}}},
        #         ]
        #     }
        # },
        query={
            "text_expansion": {
                "ml.tokens": {
                    "model_id": ".elser_model_2",
                    "model_text": text_input,
                    "boost": 0.4,
                }
            }
        },
        knn={
            "field": "embedding",
            "query_vector": embedding_vector.data[0].embedding,
            "k": 5,
            "num_candidates": 100,
            "boost": 0.7,
        },
    )
    # print("Got {} hits:".format(page["hits"]["total"]["value"]))
    return page["hits"]["hits"]

    # iterate over the document hits for this page
    # for hit in page["hits"]["hits"]:
    #     print(hit["_source"]["metadata"]["hospital_name"])
    #     print(hit["_source"]["metadata"]["patient_name"])
    #     print(hit["_source"]["content"])


# es_knn_search("How do patients feel about hospital's service")
