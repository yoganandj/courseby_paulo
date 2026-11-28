"""
Working with LLMs in LangChain V.1
Multiple providers, configuration, and streaming, and cost optimization.
"""

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


def demo_init_chat_model():
    """Demonstrates initializing a chat model with different providers and configurations."""
    
    # Initialize a chat model with NVIDIA's API
    chat_model = init_chat_model(
        model_provider="nvidia",
        model="nvidia/nemotron-3.5-super-vl-120b-a12b",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=0.7,
    )
    
    # 2. Wrap the model with LangChain's built-in retry logic
    chat_model = chat_model.with_retry(
        stop_after_attempt=3,          # Total attempts (1 initial + 2 retries)
        wait_exponential_jitter=True,  # Adds smart backoff delay between retries
    )
    
    response_stream = chat_model.stream([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is Langchain and how does it work?")
    ])
    print("Streaming response from NVIDIA's API:")
    # Print only the text content of each chunk
    for chunk in response_stream:
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()  # Newline after streaming output
    
def demo_messages():
    
    model = ChatOpenAI(
        model="nvidia/nemotron-3.5-super-vl-120b-a12b",
        temperature=0.7,
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY")
    )
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?")
    ]
    
    print("Using message objects:")
    print(f"Messages: {messages[0]} | {messages[1]}")
    
    response = model.invoke(messages)
    print("Response from NVIDIA's API:")
    print(response.content)
    print()  # Newline after response output
    
    messages.append(response)  # Add the model's response to the conversation
    
if __name__ == "__main__":
    # demo_init_chat_model()    
    demo_messages()


    
   