# app/memory_management.py

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from app.config import *



# ============================================================
# SHARED LLM
# ============================================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)


# ============================================================
# MEMORY TYPE 1: IN-MEMORY CHAT HISTORY (Manual Buffer)
# ============================================================
def demo_buffer_memory():
    """
    Full history buffer — stores every message, grows unbounded.
    """
    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful HR assistant for TechCorp."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    conversations = [
        "My name is Sarah and I work in Engineering.",
        "What is the remote work internet speed requirement?",
        "What department did I say I work in?",
        "What was the first thing I told you?",
    ]

    session_id = "buffer-session-1"

    for i, user_input in enumerate(conversations, 1):
        response = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        history = get_session_history(session_id).messages
        print(f"\n[Turn {i}]")
        print(f"  User : {user_input}")
        print(f"  Agent: {response.content}")
        print(f"  [Buffer size: {len(history)} messages]")


# ============================================================
# MEMORY TYPE 2: WINDOW MEMORY (Last K Messages)
# ============================================================
def demo_window_memory(k: int = 4):
    """
    Sliding window — only last K messages survive. Older context is dropped.
    """
    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful HR assistant for TechCorp."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    def trim_messages(chain_input):
        messages = chain_input.get("history", [])
        if len(messages) > k:
            chain_input["history"] = messages[-k:]
        return chain_input

    chain = RunnableLambda(trim_messages) | prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    conversations = [
        "My name is Sarah.",
        "I work in the Engineering department.",
        "My employee ID is EMP-4521.",
        "What is the on-call stipend?",
        "What is my employee ID?",
        "What is my name?",
    ]

    session_id = "window-session-1"

    for i, user_input in enumerate(conversations, 1):
        response = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        history = get_session_history(session_id).messages
        visible = history[-k:] if len(history) > k else history
        print(f"\n[Turn {i}]")
        print(f"  User : {user_input}")
        print(f"  Agent: {response.content}")
        print(f"  [Total stored: {len(history)} | Visible to LLM: {len(visible)} (k={k})]")


# ============================================================
# MEMORY TYPE 3: SUMMARY MEMORY
# ============================================================
def demo_summary_memory():
    """
    Periodic summarization — every 2 turns the buffer gets compressed
    into a running summary. Trades detail for scale.
    """
    summary = ""
    buffer = ChatMessageHistory()
    summary_threshold = 2

# AFTER
    summarizer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a conversation summarizer. "
     "Your output must be ONLY the updated summary — no labels, no preamble, no repetition of input.\n\n"
     "Existing summary (may be empty): {existing_summary}\n\n"
     "New conversation turns to incorporate:\n{new_messages}"),
    ("human", "Write the updated summary now. Output only the summary text itself."),
])

    summarizer_chain = summarizer_prompt | llm

    def get_response(user_input: str, current_summary: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful HR assistant for TechCorp.\n"
             "Conversation so far (summarized): {summary}"),
            ("human", "{input}"),
        ])
        chain = prompt | llm
        response = chain.invoke({"summary": current_summary, "input": user_input})
        return response.content

    def maybe_summarize(current_summary: str, recent_messages) -> str:
        """Compress recent messages into the running summary."""
        formatted = "\n".join(
            f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in recent_messages
        )
        result = summarizer_chain.invoke({
            "existing_summary": current_summary,
            "new_messages": formatted,
        })
        return result.content

    conversations = [
        "I'm looking for info about the expense reimbursement process.",
        "Specifically, how long do I have to submit receipts?",
        "What is the cap for hotel reimbursement?",
        "Does the company cover co-working spaces?",
        "What was my first question about?",
        "Summarize everything we discussed.",
    ]

    nonlocal_summary = {"value": ""}

    for i, user_input in enumerate(conversations, 1):
        # Add human message to buffer
        buffer.add_message(HumanMessage(content=user_input))

        # Generate response using current summary as context
        answer = get_response(user_input, nonlocal_summary["value"])

        # Add AI response to buffer
        buffer.add_message(AIMessage(content=answer))

        print(f"\n[Turn {i}]")
        print(f"  User : {user_input}")
        print(f"  Agent: {answer}")

        # Summarize every `summary_threshold` turns
        if i % summary_threshold == 0:
            # Take only the newest batch of messages (last 2 turns = 4 messages)
            recent = buffer.messages[-(summary_threshold * 2):]
            nonlocal_summary["value"] = maybe_summarize(nonlocal_summary["value"], recent)
            print(f"\n  [Summary updated]: {nonlocal_summary['value']}")
            # Clear the buffer — history now lives in the summary
            buffer.clear()


# ============================================================
# MEMORY TYPE 4: VECTOR MEMORY (Semantic Retrieval)
# ============================================================
def demo_vector_memory():
    """
    FAISS-backed semantic memory — embeds each Q&A pair, retrieves
    the most relevant past exchanges per new query.
    Scales to infinite history; cost stays flat per turn.
    """
    embedding_model = OpenAIEmbeddings()
    memory_store = None
    memory_docs = []

    def add_memory(question: str, answer: str):
        nonlocal memory_store
        doc = Document(
            page_content=f"Q: {question}\nA: {answer}",
            metadata={"type": "memory"},
        )
        memory_docs.append(doc)
        memory_store = FAISS.from_documents(memory_docs, embedding_model)

    def retrieve_relevant_memories(query: str, k: int = 2) -> str:
        if memory_store is None or len(memory_docs) == 0:
            return "No previous conversation."
        results = memory_store.similarity_search(query, k=k)
        return "\n".join(doc.page_content for doc in results)

    def get_response(user_input: str, memories: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful HR assistant for TechCorp.\n"
             "Relevant past conversation:\n{memories}"),
            ("human", "{input}"),
        ])
        chain = prompt | llm
        return (chain.invoke({"memories": memories, "input": user_input})).content

    conversations = [
        "What is the minimum password length at TechCorp?",
        "How often must passwords be rotated?",
        "What is the travel per diem for international trips?",
        "What are the hotel rate caps?",
        "Remind me what we discussed about passwords.",
        "What travel costs did we cover?",
    ]

    retrieval_turns = {5, 6}

    for i, user_input in enumerate(conversations, 1):
        memories = retrieve_relevant_memories(user_input)

        if i in retrieval_turns:
            print(f"\n  [Retrieved memories for Turn {i}]:")
            for line in memories.split("\n"):
                print(f"    {line}")

        answer = get_response(user_input, memories)
        add_memory(user_input, answer)

        print(f"\n[Turn {i}]")
        print(f"  User : {user_input}")
        print(f"  Agent: {answer}")
        print(f"  [Memory store size: {len(memory_docs)} docs]")


# ============================================================
# COMPARISON RUNNER
# ============================================================
def run_memory_comparison():
    """
    Runs all 4 memory demos. DONE — do not modify.
    """
    print("=" * 70)
    print("MEMORY MANAGEMENT COMPARISON")
    print("=" * 70)

    print("\n" + "─" * 70)
    print("TYPE 1: BUFFER MEMORY (Full History)")
    print("─" * 70)
    demo_buffer_memory()

    print("\n" + "─" * 70)
    print("TYPE 2: WINDOW MEMORY (Last K=4 Messages)")
    print("─" * 70)
    demo_window_memory(k=4)

    print("\n" + "─" * 70)
    print("TYPE 3: SUMMARY MEMORY")
    print("─" * 70)
    demo_summary_memory()

    print("\n" + "─" * 70)
    print("TYPE 4: VECTOR MEMORY (Semantic Retrieval)")
    print("─" * 70)
    demo_vector_memory()

    print("\n" + "=" * 70)
    print("MEMORY COMPARISON COMPLETE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    run_memory_comparison()