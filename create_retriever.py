import os
from dotenv import load_dotenv
from langchain_community.vectorstores.qdrant import Qdrant
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents.base import Document
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import polars as pl

from test_prompt_template import review_prompt_template, chat_model

load_dotenv()

# output_parser = StrOutputParser()

collection_name = os.getenv("HEALTHCARE_COLLECTION", "healthcare_collection")
embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-ada-002",
    deployment="text-embedding-ada-002",
)

qdrant_client = QdrantClient(port=6333, host="localhost")

reviews_vector_db = Qdrant(
    client=qdrant_client,
    embeddings=embeddings,
    collection_name=collection_name,
    vector_name=os.getenv("HEALTHCARE_VECTOR"),
)

# REVIEW_DATA_PATH = "data/reviews.csv"
# data_reviews = pl.read_csv(REVIEW_DATA_PATH)
# reviews = []
# metadatas = []
# for row in data_reviews.rows(named=True):
#     reviews.append(row["review"])
#     data = {
#         "patient_name": row["patient_name"],
#         "physician_name": row["physician_name"],
#         "hospital_name": row["hospital_name"],
#     }
#     metadatas.append(data)

# ls_docs = [
#     Document(page_content=review, metadata=metadata)
#     for review, metadata in zip(reviews, metadatas)
# ]

# inserted_ls = reviews_vector_db.add_documents(ls_docs)
reviews_retriever = reviews_vector_db.as_retriever(k=5)

search_value = reviews_vector_db.similarity_search(
    "How do patients feel about hospital's service",
    k=5,
    # filter=Filter(
    #     must=[
    #         FieldCondition(
    #             key="metadata.patient_name", match=MatchValue(value="Joel Robertson")
    #         ),
    #     ]
    # ),
)
# print(search_value)

# review_chain = (
#     {"context": reviews_retriever, "question": RunnablePassthrough()}
#     | review_prompt_template
#     | chat_model
#     | StrOutputParser()
# )
# question = """Has anyone complained about communication with the hospital staff?"""
# res = review_chain.invoke(question)
# print(res)
