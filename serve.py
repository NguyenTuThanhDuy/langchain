import os

from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import OpenAIEmbeddings

## Select by similarity
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import chroma

load_dotenv()

## Select by Similarity


## Embedding
embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
embeddings = embeddings_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!",
    ]
)
print(len(embeddings), len(embeddings[0]))
embedded_query = embeddings_model.embed_query("conversation?")
print(len(embedded_query))

## Prompt

llm = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,
    model=os.getenv("OPENAI_MODEL"),
)
template = """You are responsible for developing new products. Products contain 2 parts: Frontend ReactJS and Backend FastAPI. You have to implement load balancing to scale this system using Nginx.
Assume you are running either your backend and frontend in your local machine (localhost), only using docker for nginx. You have to provide the nginx config to devops team to execute this product.
Target: {target}
Person in charge: Below is an idea for a nginx config."""
prompt_template = PromptTemplate(input_variables=["target"], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)

llm = OpenAI(temperature=0.7)
template = """You are a devops engineer. You need to know the URL, host of either backend and frontend of nginx config to run project smoothly.
Idea:
{idea}
Executive:
"""
prompt_template = PromptTemplate(input_variables=["idea"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template)

overall_chain = SimpleSequentialChain(
    chains=[synopsis_chain, review_chain], verbose=True
)
review = overall_chain.run(
    "Devops engineer can access to localhost:3000 to check if React app and FastAPI are working probably"
)
