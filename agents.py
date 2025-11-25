# Create agent tools and setup agent with LangChain 1.1.0, using Wikipedia search as an example

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import wikipedia

load_dotenv()


model_name = os.getenv("LLM_MODEL")
model_key = os.getenv("LLM_API_KEY")
model_provider = os.getenv("LLM_PROVIDER")

chat = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=0.9,
    google_api_key=model_key,
)


# ========== TOOL DEFINITION ==========

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information about a topic. 
    Use this when you need to find factual information about people, places, or things."""
    try:
        result = wikipedia.summary(query, sentences=3)
        return result
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found. Please be more specific. Options: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return "Could not find information on that topic."
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"


# ========== AGENT SETUP (LangChain 1.1.0) ==========

# Bind tools to LLM
tools = [search_wikipedia]
llm_with_tools = chat.bind_tools(tools)

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant named AnjingPintar with access to Wikipedia search tool.
    
    When users ask about factual information, people, places, events, or things, 
    use the search_wikipedia tool to get accurate information.
    
    Always provide clear and informative responses based on the tool results."""),
    ("human", "{input}"),
])


def run_agent(user_input: str) -> str:
    """Run the agent with prompt template and tool calling"""
    # Format prompt
    messages = prompt.format_messages(input=user_input)
    
    # Get LLM response
    response = llm_with_tools.invoke(messages)
    
    # Check if tool was called
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        print(f"🔧 Tool called: {tool_call['name']}")
        print(f"📝 Arguments: {tool_call['args']}")
        
        # Execute the tool
        tool_result = search_wikipedia.invoke(tool_call['args'])
        print(f"📖 Tool Result:\n{tool_result}\n")
        
        # Send tool result back to LLM for final answer
        messages_with_tool = messages + [
            response,
            {"role": "tool", "content": tool_result, "tool_call_id": tool_call['id']}
        ]
        final_response = chat.invoke(messages_with_tool)
        return final_response.content
    else:
        return response.content


# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    print("LANGCHAIN AGENT WITH WIKIPEDIA TOOL")
    print("\n[Example 2: Wikipedia Search - Albert Einstein]")
    result = run_agent("Hello, who are you? I want to ask, Who is Albert Einstein?")
    print(f"✅ Final Answer: {result}")
    print("Demo completed!")