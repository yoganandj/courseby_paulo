from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from  langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


# chatprompttemplate

# prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}.")

# # format and inspect
# messages = prompt.format_messages(adjective="funny", topic="chickens")
# print(f"Formatted messages: {messages}")
# print()


# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system", "You are a helpful assistant that translates {input_language} to {output_language}."
#         ),
#         ("human", "Translate the following text : {text}"),
#     ]
# )

# messages = prompt.format_messages(
#     input_language="English", output_language="French", text="I love programming."
# )


# model = init_chat_model(
#     model_provider="nvidia",
#     model="nvidia/nemotron-3.5-super-vl-120b-a12b",
#     base_url="https://integrate.api.nvidia.com/v1",
#     api_key=os.getenv("NVIDIA_API_KEY"),
#     temperature=0.7,
# )

# response = model.invoke(messages)
# print(f"Response: {response.content}")


# Fewshot example
from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "up", "output": "down"},
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)


final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Give the opposite of each word."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

model = init_chat_model(
    model_provider="nvidia",
    model="nvidia/nemotron-3.5-super-vl-120b-a12b",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.7,
)

response = model.invoke(final_prompt.format_messages(input="hot"))
print(response.content)