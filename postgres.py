import os
from uuid import uuid4

import psycopg2
from pgvector.psycopg2 import register_vector
import polars as pl
import openai
from dotenv import load_dotenv
import numpy as np

load_dotenv(verbose=True)

REVIEW_DATA_PATH = "data/reviews.csv"
data_reviews = pl.read_csv(REVIEW_DATA_PATH)

openai_client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

conn = psycopg2.connect(
    host="localhost", database="duynguyen", user="postgres", password="postgres"
)

cur = conn.cursor()
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
register_vector(conn)


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


def insert_to_postgres():
    for vector, review, payload in prepare_dataset():
        data = (
            review,
            np.array(vector.embedding),
            payload["hospital_name"],
            payload["patient_name"],
            payload["physician_name"],
        )
        try:
            q = "INSERT INTO reviews (content, embedding, hospital_name, patient_name, physician_name) VALUES (%s, %s, %s, %s, %s)"
            cur.execute(q, data)
            print("Insert success")
        except Exception as e:
            conn.rollback()
            print(f"Error happens when inserting {e.__repr__()}")
        # insert_data.append(data)
    conn.commit()


# insert_to_postgres()
text = "How do patients feel about hospital's service"
embedding_vector = openai_client.embeddings.create(input=text, model=embedding_model)
embedding_vector = np.array(embedding_vector.data[0].embedding)

# cosine_ops = cosine distance
# ip_ops = inner product
# l2_ops = l2_norm
INDEX_LIST = ["vector_cosine_ops", "vector_ip_ops", "vector_l2_ops"]
INDEX_OPERATOR_DICT = {"vector_cosine_ops": "<=>", "vector_l2_ops": "<->"}

cur.execute(
    """
    SELECT content, hospital_name, patient_name, physician_name
    FROM reviews
    WHERE hospital_name in ('Pearson LLC', 'Schultz-Powers')
    ORDER BY embedding <=> %s LIMIT 5""",
    (embedding_vector,),
)
records = cur.fetchall()
# print(records)
print(records)

# Parameter description m Maximum number of connections per layer (default 16) ef_construction Size of dynamic candidate list for graph construction (default 64)
cur.execute(
    "CREATE INDEX ON reviews USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
)

cur.execute(
    """
    SELECT content, hospital_name, patient_name, physician_name
    FROM reviews
    WHERE hospital_name in ('Pearson LLC', 'Schultz-Powers')
    ORDER BY embedding <=> %s LIMIT 5""",
    (embedding_vector,),
)
records = cur.fetchall()
# print(records)
print(records)

# cur.execute(
#     "CREATE INDEX ON reviews USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
# )
# cur.execute(
#     "SELECT content, hospital_name, patient_name, physician_name FROM reviews ORDER BY embedding <=> %s LIMIT 5",
#     (embedding_vector,),
# )
# records = cur.fetchall()
# print(records)
# print(records[::-5])
