from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import uvicorn
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

systm_template = "translate into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ("system",systm_template),
    ("user", "{text}")
])

parser = StrOutputParser()

chain = prompt_template|model|parser

app = FastAPI(title="Langchain server",
              version="1.0",
              description="a simple api server using langchain interface")

add_routes(
 app,
 chain,
 path="/chain"
)


if __name__=="__main__":
 
 uvicorn.run(app,host="127.0.0.1", port=8088)