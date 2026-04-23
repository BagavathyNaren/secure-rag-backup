# app/chain.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

from app.config import * 


# -------------------------------------------------------------------
# 1. Initialize the LLM (ensure OPENAI_API_KEY is set in your env)
# -------------------------------------------------------------------
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

# -------------------------------------------------------------------
# 2. Define the two prompts
# -------------------------------------------------------------------
prompt1 = ChatPromptTemplate.from_template(
    "Summarize the following text in 3 sentences:\n\n{text}"
)

prompt2 = ChatPromptTemplate.from_template(
    "Extract all named entities (people, organizations, locations) from this summary. "
    "Return them as a comma-separated list, with no additional text.\n\nSummary: {summary}"
)

# -------------------------------------------------------------------
# 3. Build the chain using LCEL
# -------------------------------------------------------------------
chain = (
    # Step 0: Wrap the input string into a dict with key "text"
    {"text": RunnablePassthrough()}
    # Step 1: Generate the summary and add it to the state under "summary"
    | RunnablePassthrough.assign(summary=prompt1 | model | StrOutputParser())
    # Step 2: Generate the raw entity string (comma-separated) using the summary
    | RunnablePassthrough.assign(entities_raw=prompt2 | model | StrOutputParser())
    # Step 3: Parse the comma-separated string into a clean list of entities
    | RunnableLambda(
        lambda x: {
            "summary": x["summary"],
            "entities": [e.strip() for e in x["entities_raw"].split(",") if e.strip()]
        }
    )
)

if __name__ == "__main__":
    test_input = """
    OpenAI announced a partnership with Microsoft to invest $10 billion in 
    artificial intelligence research. The deal, led by CEO Sam Altman and 
    Microsoft's Satya Nadella, will focus on developing AGI safely. The 
    partnership will be based in San Francisco and Redmond, Washington. 
    Critics, including Elon Musk and researchers at Google DeepMind in 
    London, have raised concerns about the concentration of AI power in 
    few corporations. The European Union is also considering new regulations 
    through its AI Act, which could impact operations across Brussels and 
    member states.
    """

    result = chain.invoke(test_input)
    print(result)