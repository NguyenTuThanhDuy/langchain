import os

from fastapi import FastAPI, HTTPException
from pydantic import (
    BaseModel,
    validator,
    root_validator,
    field_validator,
    model_validator,
)
from typing import List
import openai
from dotenv import load_dotenv
import numpy as np

from create_retriever import reviews_vector_db
from postgres import conn as pg_conn
from es import es_knn_search

load_dotenv(verbose=True)

openai_client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def create_embedding_vector(input_str: str):
    return (
        openai_client.embeddings.create(input=input_str, model=embedding_model)
        .data[0]
        .embedding
    )


class RequestModel(BaseModel):
    input_str: str

    # @validator("input_str", pre=True)  # <-
    # def validate_name(cls, v):
    #     if len(v) < 5:
    #         raise ValueError("Input String is too short")
    #     return v

    # @root_validator(pre=True)  # <-
    # def validate_root(cls, values):
    #     return values

    @field_validator("input_str", mode="before")  # <-
    def validate_name(cls, v):
        if len(v) <= 5:
            raise HTTPException(status_code=400, detail="Input String is too short")
        return v

    @model_validator(mode="before")  # <-
    def validate_root(cls, values):
        return values


class ResponseModel(BaseModel):
    qdrant_records: List[dict]
    pg_records: List[dict]
    es_records: List[dict]


app = FastAPI()
pg_cursor = pg_conn.cursor()


async def get_info_from_qdrant(embedding_vector: list):
    records = reviews_vector_db.similarity_search_by_vector(
        embedding=embedding_vector, k=5
    )
    return list(
        map(
            lambda x: {
                "content": x.page_content,
                "hospital_name": x.metadata.get("hospital_name", ""),
                "patient_name": x.metadata.get("patient_name", ""),
                "physician_name": x.metadata.get("physician_name", ""),
            },
            records,
        )
    )


async def get_info_from_postgres(embedding_vector: list):
    embedding_vector = np.array(embedding_vector)
    pg_cursor.execute(
        """
    SELECT content, hospital_name, patient_name, physician_name 
    FROM reviews 
    ORDER BY embedding <=> %s LIMIT 5""",
        (embedding_vector,),
    )
    records = pg_cursor.fetchall()
    return list(
        map(
            lambda x: {
                "content": x[0],
                "hospital_name": x[1],
                "patient_name": x[2],
                "physician_name": x[3],
            },
            records,
        )
    )


async def get_info_from_es(text_input: str):
    records = es_knn_search(text_input=text_input)
    return list(
        map(
            lambda x: {
                "content": x["_source"]["content"],
                "hospital_name": x["_source"]["metadata"]["hospital_name"],
                "patient_name": x["_source"]["metadata"]["patient_name"],
                "physician_name": x["_source"]["metadata"]["physician_name"],
            },
            records,
        )
    )


@app.get("/")
async def health_check():
    return {"message": "OK"}


@app.post("/query/info", status_code=200, response_model=ResponseModel)
async def info_retrieval(request: RequestModel):
    embedding_vector = create_embedding_vector(request.input_str)
    return {
        "qdrant_records": await get_info_from_qdrant(embedding_vector),
        "pg_records": await get_info_from_postgres(embedding_vector),
        "es_records": await get_info_from_es(request.input_str),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", reload=True, port=8001)
