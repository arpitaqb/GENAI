import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel
from fastapi import FastAPI,HTTPException
from typing import Literal, List, Optional

os.environ['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY')

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

class Email(BaseModel):
    id_: str
    from_: str
    to: List[str]
    cc: List[str]
    bcc: List[str]
    subject: str
    body: str
    parent_email_id: Optional[str] = None

class EmailInput(BaseModel):
    mode: Literal["reply", "general"]
    email : List[Email]
    prompt : str

class EmailDrafter(BaseModel):
    body : str

parser = PydanticOutputParser(pydantic_object=EmailDrafter)

reply_prompt = ChatPromptTemplate.from_messages([
    ('system', (
        "You are a highly efficient and professional AI assistant specializing in email drafting and replies. "
        "Carefully read and understand the incoming email provided by the user. "
        "Incorporate any additional context provided to inform your response. "
        "Use earlier emails for context only if necessary.\n\n"
        "Your responses must always be clear, polite, concise, and professional/formal, unless overridden by specific instructions. "
        "Respond strictly in JSON format as specified below:\n"
        "{format_instructions}"
    )),
    ('user', (
        "email thread( most recent last):\n"
        "{email_thread}\n\n"
        "user instruction:\n{prompt}"
    ))
])

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a highly efficient and professional AI assistant specializing in email drafting and replies. "
        "Carefully read and understand the incoming email provided by the user. "
        "Incorporate any additional context provided to inform your response. "
        "Your responses must always be clear, polite, concise, and professional/formal, unless overridden by specific instructions. "
        "Respond strictly in JSON format as specified below:\n"
        "{format_instructions}"
    )),
    ("user", "{prompt}")
])

app = FastAPI(title="email drafter")

@app.post("/draft")
def email_draft(data: EmailInput):
    try:
        if data.mode == "reply":
            if not (data.email and data.prompt):
                raise HTTPException(status_code=400, detail="Missing 'email' or 'reply_prompt' for reply mode.")
            
            sorted_emails = sorted(data.email, key=lambda x: x.id_)
            email_thread = "\n```\n".join([
                f"From: {email.from_}\nTo: {', '.join(email.to)}\nsubject: {email.subject}\n\n {email.body}"
                for email in sorted_emails
            ])

            chain = reply_prompt | llm | parser
            result: EmailDrafter = chain.invoke({
                "email_thread" : email_thread,
                "prompt": data.prompt,
                "format_instructions": parser.get_format_instructions()
            })


        elif data.mode == "general":
            if not data.prompt:
                raise HTTPException(status_code=400, detail="Missing 'prompt' for general mode.")

            chain = prompt | llm | parser
            result: Email = chain.invoke({
                "prompt": data.prompt,
                "format_instructions": parser.get_format_instructions()
            })


        else:
            raise HTTPException(status_code=400, detail="internal server error")
        return {
            "body" : result.body
        }
    except Exception as e:
        print(e)