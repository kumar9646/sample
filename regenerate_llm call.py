from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# Prompt template for reformulation
prompt = PromptTemplate(
    input_variables=["history", "new_query"],
    template=(
        "Given the following conversation history:\n{history}\n\n"
        "Reformulate the user's latest query so it is self-contained, "
        "contextually clear, and suitable for retrieval or reasoning. "
        "Latest user query: {new_query}\n\nReformulated query:"
    ),
)

# Example setup
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
memory = ConversationBufferMemory(return_messages=True)

conversation = ConversationChain(llm=llm, memory=memory, prompt=prompt)

reformulated = conversation.run({
    "history": """User: In which meeting Gao Lee and Jang participated?
AI: Following are the list of meetings: X1 meeting, X2 meeting, and X3 meeting.""",
    "new_query": "Can you summarize the project discussed in the above meetings?"
})

print(reformulated)
