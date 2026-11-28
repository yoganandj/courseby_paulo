from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import init_chat_model


from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, parser
import os
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


parser = StrOutputParser()


# prompt = ChatPromptTemplate.from_template("Write a poem about {topic}")

llm = init_chat_model( 
    model_provider="nvidia",
    model="nvidia/nemotron-3.5-super-vl-120b-a12b",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.7,
)

# chain = prompt | llm | parser

# response = chain.invoke({"topic": "sunset"})

# print(f"Response: {response}")

# print()
# print(type(response))


from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_template(
    "Return a JSON object with the following fields: name, age, and occupation. The description is: {description}"
)

chain = prompt | llm | parser

# response = chain.invoke({"description": "A 25 year old developer Yoganand"})


# print(f"Response: {response}")



# PydanticOutputParser example
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    occupation: str = Field(description="The person's occupation")
    
parser = PydanticOutputParser(pydantic_object=Person)

prompt = ChatPromptTemplate.from_template(
    "Return a JSON object with the following fields: name, age, and occupation. The description is: {description}"
).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser

response = chain.invoke({"description": "A 25 year old developer Yoganand"})
print(f"Response: {response}")    
    

# StructuredOutputParser example


class MovieReview(BaseModel):
    title: str = Field(description="The title of the movie")
    review: str = Field(description="A brief review of the movie")
    rating: float = Field(description="The rating of the movie out of 10")
    
structured_model = llm.with_structured_output(MovieReview)

result = structured_model.invoke("Review: Inception is a mind-bending thriller that keeps you on the edge of your seat. The plot is complex and layered, but the execution is flawless. The performances are top-notch, and the visuals are stunning. A must-watch for sci-fi fans. Rating: 9/10")


print(f"Structured Output: {result}")    
        

