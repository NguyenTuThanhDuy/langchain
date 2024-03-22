import os
from uuid import uuid4

from dotenv import load_dotenv
import openai
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointStruct,
    VectorParams,
    Distance,
    OptimizersConfigDiff,
)
import polars as pl
from langchain_openai import OpenAIEmbeddings
from typing import List

load_dotenv()

openai_client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

qdrant_db = QdrantClient(port=6333, host="localhost")

REVIEW_DATA_PATH = "data/reviews.csv"
data_reviews = pl.read_csv(REVIEW_DATA_PATH)
print(data_reviews.shape)

# reviews = []
# patient_names = []
# physician_names = []
# hospital_names = []

print(data_reviews.head(1))

# for row in data_reviews.rows(named=True):
#     reviews.append(row["review"])
#     patient_names.append(row["patient_name"])
#     physician_names.append(row["physician_name"])
#     hospital_names.append(row["hospital_name"])
#     break

embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"), model=embedding_model
)

# result = openai_client.embeddings.create(input=reviews, model=embedding_model)

# points = [
#     PointStruct(
#         id=idx,
#         vector={"review_vector": data.embedding},
#         payload={
#             "review": review,
#             "physician_name": physician_name,
#             "hospital_name": hospital_name,
#             "patient_name": patient_name,
#         },
#     )
#     for idx, (data, review, patient_name, physician_name, hospital_name) in enumerate(
#         zip(result.data, reviews, patient_names, physician_names, hospital_names)
#     )
# ]

# collection_name = "healthcare_collection"

# success = qdrant_db.create_collection(
#     collection_name,
#     vectors_config={
#         "review_vector": VectorParams(
#             size=1536,
#             distance=Distance.COSINE,
#         ),
#     },
# )
# print(success)

# qdrant_db.upload_points(collection_name, points, batch_size=128)

# success = qdrant_db.update_collection(
#     collection_name=collection_name,
#     optimizers_config=OptimizersConfigDiff(
#         indexing_threshold=20000,
#     ),
# )
# print(success)

# results = qdrant_db.search(
#     collection_name=collection_name,
#     query_vector=(
#         "review_vector",
#         openai_client.embeddings.create(
#             input=[
#                 "What did Christy Johnson review when she stayed at Wallace-Hamilton hospital?"
#             ],
#             model=embedding_model,
#         )
#         .data[0]
#         .embedding,
#     ),
#     with_payload=["hospital_name", "patient_name", "review"],
#     limit=5,
# )
# print(results)


class QDrantDatabase:

    def __init__(
        self,
        host: str,
        port: int,
        openai_api_key: str,
        embedding_model: str,
        collection_name: str,
        vector_name: str = None,
        **kwargs
    ):
        self.openai_api_key = openai_api_key
        self.host = host
        self.port = port
        self.embedding_model = embedding_model
        self.qdrant_client = QdrantClient(host=host, port=port)
        self.openai_client = openai.Client(api_key=openai_api_key)
        self.vector_name = vector_name
        self.collection_name = collection_name

    def create_collection(self, **kwargs):
        if not self.qdrant_client.collection_exists(self.collection_name):
            success = self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=(
                    {
                        self.vector_name: VectorParams(
                            size=1536,
                            distance=Distance.COSINE,
                        ),
                    }
                    if self.vector_name
                    else VectorParams(
                        size=1536,
                        distance=Distance.COSINE,
                    )
                ),
                **kwargs,
            )
            return success
        return True

    def create_embedding_vectors_from_inputs(self, inputs: List[str], **kwargs):
        return self.openai_client.embeddings.create(
            input=inputs, model=self.embedding_model, **kwargs
        )

    def convert_input_to_point(self, embedding_vector_data, payload):
        return PointStruct(
            id=uuid4().hex,
            vector=(
                {self.vector_name: embedding_vector_data.embedding}
                if self.vector_name
                else embedding_vector_data.embedding
            ),
            payload=payload,
        )

    def search_by_similarity(
        self,
        inputs: List[str],
        with_payload: List[str] = None,
        limit: int = 5,
        skip: int = 0,
        query_filter=None,
        **kwargs
    ):
        return self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=(
                self.vector_name,
                self.create_embedding_vectors_from_inputs(inputs).data[0].embedding,
            ),
            with_payload=with_payload,
            limit=limit,
            offset=skip,
            query_filter=query_filter,
            **kwargs,
        )

    def upload_points(self, points: List[PointStruct], **kwargs):
        self.qdrant_client.update_collection(
            collection_name=self.collection_name,
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=0,
            ),
        )
        success = self.qdrant_client.upload_points(
            self.collection_name, points=points, **kwargs
        )

        self.qdrant_client.update_collection(
            collection_name=self.collection_name,
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000,
            ),
        )
        return success
