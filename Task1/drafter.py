import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel
from fastapi import FastAPI,HTTPException


app = FastAPI(title="Email Drafter")


os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


# Structured Output
class EmailDrafer(BaseModel):
    to : str
    from_ : str
    subject : str
    body : str

# input class    
class Email_Input(BaseModel):
    email : dict
    context : str  
    reply_prompt : str


parser = PydanticOutputParser(pydantic_object=EmailDrafer)


prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a highly efficient and professional AI assistant specializing in email drafting and replies."
        "Carefully read and understand the incoming email provided by the user."
        "Incorporate any additional context provided to inform your response."
        "Your responses must always be clear, polite, concise, and professional/formal, unless explicitly overridden by specific instructions."
        "Respond strictly in JSON format as specified below:\n"
        "{format_instructions}" # For structured output or length constraints
        )
    ),
    
    ("user", (
        "Incoming Email:\n```\n{email}\n```\n\n" 
        "Optional Context (if any):\n```\n{context}\n```\n\n" 
        "My Reply Instructions:\n{reply_prompt}\n\n" 
        "Please draft the email reply now."
        )
    )
])

chain = prompt | llm | parser



@app.post("/draft")
def classifier(data : Email_Input):

    email_class: EmailDrafer = chain.invoke(
        {
            "email" : data.email,
            "context" : data.context,
            "reply_prompt" : data.reply_prompt ,
            "format_instructions" : parser.get_format_instructions()
        }
    )

    try:
        if email_class: 
            return {
                "to": email_class.to,
                "from": email_class.from_,
                "subject": email_class.subject,
                "body": email_class.body,
            }
        else:
            raise HTTPException(status_code=500,detail="Intenal Server Error")
    except Exception as e:
        print(e)