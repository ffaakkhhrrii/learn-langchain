# Use langchain to parse customer reviews into structured JSON data, with JSONOutputParser from langchain_core.output_parsers.

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

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

chat


customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

parser = JsonOutputParser()

schema = {
    "gift": "Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.",
    "delivery_days": "How many days\
                                      did it take for the product\
                                      to arrive? If this \
                                      information is not found,\
                                      output -1.",
    "price_value": "Extract any\
                                    sentences about the value or \
                                    price, and output them as a \
                                    comma separated Python list."
}

review_template_2 = """\
For the following text, extract the following information:

JSON Keys: 
{schema}

text: {text}

{format_instructions}
"""

prompt_template_2 = ChatPromptTemplate.from_template(review_template_2)
messages = prompt_template_2.format_messages(text=customer_review, 
                                format_instructions=parser.get_format_instructions(),
                                schema=schema)

response = chat.invoke(messages)

result = parser.parse(response.content)

print(result)
print(type(result))
print(result.get("gift"))
