from dotenv import load_dotenv
import os
load_dotenv()

from langchain_core import __version__ as core_version
from langchain_openai import ChatOpenAI, ChatOpenAI, __version__ as openai_version



def main():
    # print("Hello from courseby-paulo!")
    # print(f"langchain-core version: {core_version}")
    # print(f"langchain-openai version: {openai_version}")
    
    # Initialize the OpenAI chat client targeting NVIDIA's API
    llm = ChatOpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
        model="nvidia/nemotron-3.5-super-vl-120b-a12b"
    )
    response = llm.invoke([{"role": "user", "content": "What is the capital of France?"}])
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
