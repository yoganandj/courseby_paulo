from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

def demo_basic_chain():
    """Demonstrates a basic chain using LCEL and Runnables"""
    
    # Create a prompt template using LCEL
    prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer in one sentence: {question}")
    
    model = ChatOpenAI(model="nvidia/nemotron-3.5-super-vl-120b-a12b", temperature=0.7, base_url="https://integrate.api.nvidia.com/v1", api_key=os.getenv("NVIDIA_API_KEY"))
    parser = StrOutputParser()
    
    # Compose with pipe operator
    chain = prompt | model | parser
    
    # Execute the chain with a question
    question = "What is the capital of France?"
    answer = chain.invoke({"question": question})
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    
def demo_batch_execution():
    """Demonstrates batch execution for multiple questions using the same chain"""
    
    prompt = ChatPromptTemplate.from_template("Translate to French: {text}")
    model = ChatOpenAI(model="nvidia/nemotron-3.5-super-vl-120b-a12b", temperature=0.7, base_url="https://integrate.api.nvidia.com/v1", api_key=os.getenv("NVIDIA_API_KEY"))
    parser = StrOutputParser()
    
    chain = prompt | model | parser
    
    # Batch - run with multiple inputs
    inputs = [{"text": "Hello, how are you?"}, {"text": "Good morning!"}, {"text": "What is your name?"}]
    answers = chain.batch(inputs)
    
    for i, input_data in enumerate(inputs):
        print(f"Input: {input_data['text']}")
        print(f"Output: {answers[i]}")
        print("---")
    
def demo_streaming():
    """Demonstrates streaming for real-time output."""
    prompt = ChatPromptTemplate.from_template("Write a haiku about: {topic}")
    model = ChatOpenAI(model="nvidia/nemotron-3.5-super-vl-120b-a12b", temperature=0.7, base_url="https://integrate.api.nvidia.com/v1", api_key=os.getenv("NVIDIA_API_KEY"), streaming=True)
    parser = StrOutputParser()
    
    chain = prompt | model | parser
    
    # Streaming - run with streaming enabled
    print("Streaming output: ")
    for chunk in chain.stream({"topic": "sunset"}):
        print(chunk, end="", flush=True)
    print()    
 
def demo_schema_inspection():
    """Demonstrates schema inspection for a chain."""
    prompt = ChatPromptTemplate.from_template("Summarize the following text: {topic}")
    model = ChatOpenAI(model="nvidia/nemotron-3.5-super-vl-120b-a12b", temperature=0.7, base_url="https://integrate.api.nvidia.com/v1", api_key=os.getenv("NVIDIA_API_KEY"))
    parser = StrOutputParser()
    
    chain = prompt | model | parser
    
    # Inspect input and output schema
    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()
    
    print(f"Input Schema: {input_schema}")
    print(f"Output Schema: {output_schema}")
    
    
def exercise_first_chain():
    """
    Exercise: Create a chain that:
    1. Takes a product name and target audience
    2. Generates a marketing tagline
    3. Return just tagline as a string
    
    Test with: product="AI Course", audience="developers"
    """
    
    prompt = ChatPromptTemplate.from_template("Create a marketing tagline for the product '{product}' targeting '{audience}'. And return only the tagline as a string.")
    model = ChatOpenAI(model="nvidia/nemotron-3.5-super-vl-120b-a12b", temperature=0.7, base_url="https://integrate.api.nvidia.com/v1", api_key=os.getenv("NVIDIA_API_KEY"))
    parser = StrOutputParser()  
    
    chain = prompt | model | parser
    
    result = chain.invoke({"product": "AI Course", "audience": "developers"})
    print(f"Marketing Tagline: {result}")
        
 
    
if __name__ == "__main__":
    # demo_basic_chain()    
    # demo_batch_execution()
    # demo_streaming()
    # demo_schema_inspection()
    exercise_first_chain()
