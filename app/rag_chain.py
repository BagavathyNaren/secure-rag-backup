# app/rag_chain.py
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pydantic import BaseModel, Field
from app.config import * 

# ------------------------------
# 1. Load and split the document
# ------------------------------
loader = TextLoader("data/company_policy.txt")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# ------------------------------
# 2. Create vector store + retriever
# ------------------------------
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ------------------------------
# 3. Define Pydantic output schema
# ------------------------------
class PolicyAnswer(BaseModel):
    answer: str = Field(description="Direct answer to the question based on the policy")
    confidence: str = Field(description="HIGH, MEDIUM, or LOW based on how well the context supports the answer")
    sources: list[str] = Field(description="Relevant snippets from the policy that support the answer")

# ------------------------------
# 4. Initialize the LLM with structured output
# ------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

structured_llm = llm.with_structured_output(PolicyAnswer)

# ------------------------------
# 5. Build the LCEL chain
# ------------------------------
# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions based strictly on the provided company policy context.
Answer ONLY from the given context. If the context does not contain the answer, say "The policy does not address this question."
Provide a confidence level: HIGH if the answer is directly supported, MEDIUM if partially supported, LOW if not supported.
Include the exact source snippets you used from the context."""),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

# Setup parallel inputs
setup = RunnableParallel(
    context=retriever,
    question=RunnablePassthrough()
)

# Assemble the chain
chain = setup | prompt | structured_llm

# ------------------------------
# 6. (Optional) Test the chain with sample questions
# ------------------------------
if __name__ == "__main__":
    questions = [
        "What internet speed is required for remote work?",
        "Can interns work remotely?",
        "What happens if I get bad performance reviews?",
        "What is the company's policy on cryptocurrency trading?"  # Not in document
    ]
    
    for q in questions:
        print(f"\nQuestion: {q}")
        result = chain.invoke(q)
        print(f"Answer: {result.answer}")
        print(f"Confidence: {result.confidence}")
        print(f"Sources: {result.sources}")
        print("-" * 50)