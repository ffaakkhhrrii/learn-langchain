# Use memory in LangChain with InMemoryChatMessageHistory and RunnableWithMessageHistory

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

model_name = os.getenv("LLM_MODEL")
model_key = os.getenv("LLM_API_KEY")
model_provider = os.getenv("LLM_PROVIDER")

chat = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=0.0,
    google_api_key=model_key,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a rude assistant."),
    ("human", "{input}")
])

# Session store (pengganti ConversationBufferMemory)
store = {}

def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Chain dengan memory (pengganti ConversationChain)
conversation = RunnableWithMessageHistory(
    prompt | chat,
    get_history
)

# contoh eksekusi, identik dengan predict()
resp1 = conversation.invoke(
    {"input": "Hi, my name is Fakhri"},
    config={"configurable": {"session_id": "user1"}}
)
print(resp1.content)

resp2 = conversation.invoke(
    {"input": "What is 1+1?"},
    config={"configurable": {"session_id": "user1"}}
)
print(resp2.content)

resp3 = conversation.invoke(
    {"input": "What is my name?"},
    config={"configurable": {"session_id": "user1"}}
)
print(resp3.content)