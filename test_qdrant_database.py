import os
from dotenv import load_dotenv
import polars as pl
from langchain.document_loaders.csv_loader import CSVLoader
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from qdrant_database import QDrantDatabase

load_dotenv()

REVIEW_DATA_PATH = "data/reviews.csv"
data_reviews = pl.read_csv(REVIEW_DATA_PATH)
print(data_reviews.shape)

loader = CSVLoader(file_path=REVIEW_DATA_PATH, source_column="review")
reviews = loader.load()

qddb = QDrantDatabase(
    host="localhost",
    port=6333,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL"),
    collection_name=os.getenv("HEALTHCARE_COLLECTION"),
    vector_name=os.getenv("HEALTHCARE_VECTOR"),
)

qddb.create_collection()

reviews = []
payloads = []
for row in data_reviews.rows(named=True):
    reviews.append(row["review"])
    payload = {
        "review": row["review"],
        "patient_name": row["patient_name"],
        "physician_name": row["physician_name"],
        "hospital_name": row["hospital_name"],
    }
    payloads.append(payload)

# embedding_vectors = qddb.create_embedding_vectors_from_inputs(reviews)

# list_points = [
#     qddb.convert_input_to_point(data, payload)
#     for data, payload in zip(embedding_vectors.data, payloads)
# ]

# qddb.upload_points(list_points)

res = qddb.search_by_similarity(
    ["How do patients feel about hospital's service"],
    with_payload=["review", "hospital_name", "patient_name"],
    # query_filter=Filter(
    #     must_not=[
    #         FieldCondition(
    #             key="payload.patient_name", match=MatchValue(value="Johnson")
    #         ),
    #         FieldCondition(key="payload.patient_name", match=MatchValue(value="John")),
    #     ]
    # ),
)
print(res)
