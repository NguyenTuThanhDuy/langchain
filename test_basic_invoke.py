from langchain.schema.messages import HumanMessage, SystemMessage
from chatbot import chat_model

messages = [
    SystemMessage(
        content="""You're an assistant knowledgeable about
         healthcare. Only answer healthcare-related questions."""
    ),
    HumanMessage(content="What is Medicaid managed care?"),
]

answer = chat_model.invoke(messages)
print(answer)
