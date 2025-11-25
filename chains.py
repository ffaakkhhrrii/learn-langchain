# Using chain to process multiple inputs with LangChain

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.base import RunnableSequence
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import pandas as pd

load_dotenv()

model_name = os.getenv("LLM_MODEL")
model_key = os.getenv("LLM_API_KEY")
model_provider = os.getenv("LLM_PROVIDER")

chat = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=0.9,
    google_api_key=model_key,
)

df = pd.read_csv("data/Product.csv")

prompt_template = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}?"
)


chain = prompt_template | chat

product = df.head()

response = chain.invoke({
    "product": product
})

print(response.content)