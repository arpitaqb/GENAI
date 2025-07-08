import os 
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

#langchain api keys
os.environ['LANGCHAIN_API_KEY']= os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'True'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

#prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "act as you are ai assistant. respond to the asked questions."),
        ("user", "question : {question}")
    ]
)

#streamlit framework
st.title("demo of langchain")
input_txt = st.text_input("ask your question..")

#ollama model
llm = Ollama(model="llama3")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser


if input_txt:
    st.write(chain.invoke({"question": input_txt}))