import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from typing import List
from operator import itemgetter

from app.config import *
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field


# -------------------------------------------------------------------
# 1. Structured output model
# -------------------------------------------------------------------
class PolicyAnswer(BaseModel):
    answer: str = Field(description="Direct answer based on the policy")
    confidence: str = Field(description="HIGH, MEDIUM, or LOW")
    sources: List[str] = Field(description="Exact snippets from the policy")
    department: str = Field(description="Which department policy was consulted")
    retrieved_contexts: List[str] = Field(
        default_factory=list,
        description="Raw retrieved chunks from FAISS (set by chain, not LLM)"
    )

# -------------------------------------------------------------------
# 2. Helper to create a retriever from a file
# -------------------------------------------------------------------
def create_retriever(file_path: str):
    loader = TextLoader(file_path)
    docs = loader.load()
    # FIX #2: Increased chunk_size 300→600, overlap 50→100
    # Prevents policy sentences from being split mid-way,
    # improving context precision scores
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# Create the three retrievers
hr_retriever = create_retriever("data/company_policy.txt")
security_retriever = create_retriever("data/security_policy.txt")
finance_retriever = create_retriever("data/finance_policy.txt")


# -------------------------------------------------------------------
# 3. Router: classifies the question into a department
# -------------------------------------------------------------------
router_prompt = ChatPromptTemplate.from_template(
    """Classify this question into exactly one category.

Categories:
- "hr": Questions about remote work, PTO, work hours, equipment, performance reviews
- "security": Questions about passwords, data classification, incidents, access control, vendors
- "finance": Questions about expenses, travel, reimbursement, budgets, corporate cards

Question: {question}

Respond with ONLY the category name, nothing else."""
)

router_llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
router_chain = router_prompt | router_llm | StrOutputParser()

# -------------------------------------------------------------------
# 4. Department-specific RAG chains
# -------------------------------------------------------------------
def create_department_chain(retriever, department_name: str):
    """Build a RAG chain that exposes real retrieved chunks for RAGAS evaluation."""

    prompt = ChatPromptTemplate.from_template(
        f"""You are an assistant for the {department_name} department.
Answer the question **only** using the policy excerpts below.
If the answer is not explicitly stated, say "I don't know".

Every claim in your answer must come directly from the excerpts.
Do not add outside knowledge or interpretations.

Policy excerpts:
{{context}}

Question: {{question}}

Answer:"""
    )

    primary_llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
    backup_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    structured_llm = primary_llm.with_fallbacks([backup_llm]).with_structured_output(PolicyAnswer)

    # FIX #1: Capture actual FAISS-retrieved chunks here,
    # NOT the LLM-generated sources field.
    # Previously, evaluate.py used result.sources (LLM output) as contexts,
    # which made RAGAS faithfulness evaluate answer vs. hallucinated snippets.
    def retrieve_and_format(inputs: dict) -> dict:
        docs = retriever.invoke(inputs["question"])
        return {
            "context": "\n\n".join(doc.page_content for doc in docs),
            "question": inputs["question"],
            "raw_docs": [doc.page_content for doc in docs],  # real chunks
        }

    def build_answer(inputs: dict) -> PolicyAnswer:
        chain_input = {"context": inputs["context"], "question": inputs["question"]}
        answer: PolicyAnswer = (prompt | structured_llm).invoke(chain_input)
        answer.department = department_name
        answer.retrieved_contexts = inputs["raw_docs"]  # attach real chunks
        return answer

    return RunnableLambda(retrieve_and_format) | RunnableLambda(build_answer)

hr_chain = create_department_chain(hr_retriever, "hr")
security_chain = create_department_chain(security_retriever, "security")
finance_chain = create_department_chain(finance_retriever, "finance")


# -------------------------------------------------------------------
# 5. RunnableBranch for routing
# -------------------------------------------------------------------
branch = RunnableBranch(
    (lambda x: x["route"] == "hr", hr_chain),
    (lambda x: x["route"] == "security", security_chain),
    (lambda x: x["route"] == "finance", finance_chain),
    hr_chain,  # default fallback
)

full_chain = (
    RunnablePassthrough.assign(route=router_chain)
    | branch
)

# -------------------------------------------------------------------
# 6. Test (if run directly)
# -------------------------------------------------------------------
if __name__ == "__main__":
    test_questions = [
        "What is the minimum password length?",            # security
        "Can I fly business class to London?",             # finance
        "What happens after two bad performance reviews?", # hr
        "How quickly must I report a security breach?",    # security
        "What is the per diem for international travel?",  # finance
        "Do I need VPN on public Wi-Fi?",                  # hr/security
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        try:
            result = full_chain.invoke({"question": q})
            print(f"Department: {result.department}")
            print(f"Answer: {result.answer}")
            print(f"Confidence: {result.confidence}")
            print("Sources:")
            print("DONE")
            for src in result.sources:
                print(f"  - {src}")
            print("Retrieved Contexts (for RAGAS):")
            for ctx in result.retrieved_contexts:
                print(f"  [chunk] {ctx[:80]}...")
        except Exception as e:
            print(f"Error: {e}")

"""

    test_questions = [
        "What is the minimum password length?",            # security
        "Can I fly business class to London?",             # finance
        "What happens after two bad performance reviews?", # hr
        "How quickly must I report a security breach?",    # security
        "What is the per diem for international travel?",  # finance
        "Do I need VPN on public Wi-Fi?",                  # hr/security
    ]


"""