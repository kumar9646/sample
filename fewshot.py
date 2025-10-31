from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain

# -----------------------------
# Step 1. Chat History Setup
# -----------------------------
chat_history = [
    {"role": "human", "content": "in which meeting gao lee and jang participated"},
    {"role": "AI", "content": "following are the list of meetings: x1 meeting, x2 meetings and x3 meetings"},
    {"role": "user", "content": "can you summarize the project discussed in the above meetings"}
]

chat_history_text = "\n".join(
    [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[:-1]]
)
latest_query = chat_history[-1]["content"]

# -----------------------------
# Step 2. Few-Shot Examples
# -----------------------------
examples = [
    {
        "input": (
            "User: In which meetings Alice and Bob participated?\n"
            "AI: They joined M1, M2, and M3.\n"
            "User: Summarize the topics discussed in the above meetings."
        ),
        "output": (
            "Summarize the topics discussed in meetings M1, M2, and M3 where Alice and Bob participated."
        ),
    },
    {
        "input": (
            "User: In which sessions Maria and John attended?\n"
            "AI: They attended sessions S1 and S2.\n"
            "User: What decisions were made in those sessions?"
        ),
        "output": (
            "Summarize the decisions made in sessions S1 and S2 attended by Maria and John."
        ),
    },
    {
        "input": (
            "User: In which projects did Rahul and Meena work together?\n"
            "AI: They worked on Project A and Project B.\n"
            "User: What were the outcomes of the above projects?"
        ),
        "output": (
            "Summarize the outcomes of Project A and Project B where Rahul and Meena worked together."
        ),
    }
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Example Conversation:\n{input}\nReformulated Query:\n{output}\n"
)

fewshot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="You are a smart query reformulator. Your task is to rewrite the user's latest query so it is fully self-contained and unambiguous based on prior chat history.",
    suffix="Conversation:\n{chat_history}\nUser's latest query: {user_query}\n\nReformulated query:",
    input_variables=["chat_history", "user_query"]
)

# -----------------------------
# Step 3. LLM and Chain Setup
# -----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
reformulate_chain = LLMChain(llm=llm, prompt=fewshot_prompt)

# -----------------------------
# Step 4. Generate Reformulated Query
# -----------------------------
reformulated_query = reformulate_chain.run({
    "chat_history": chat_history_text,
    "user_query": latest_query
})

print("🧩 Reformulated Query:")
print(reformulated_query.strip())
