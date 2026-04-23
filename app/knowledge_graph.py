# app/knowledge_graph.py

from app.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


# ============================================================
# 1. CONNECT TO NEO4J AND BUILD THE KNOWLEDGE GRAPH
# ============================================================
def build_knowledge_graph():
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:

        # ── CLEAR ────────────────────────────────────────────
        session.run("MATCH (n) DETACH DELETE n")
        print("  Cleared existing graph data")

        # ── DEPARTMENTS ──────────────────────────────────────
        session.run("""
            CREATE (:Department {name: 'Engineering',  head: 'VP of Engineering', headcount: 152})
            CREATE (:Department {name: 'Sales',        head: 'VP of Sales',       headcount: 91})
            CREATE (:Department {name: 'Marketing',    head: 'VP of Marketing',   headcount: 36})
            CREATE (:Department {name: 'Security',     head: 'CISO',              headcount: 25})
            CREATE (:Department {name: 'Finance',      head: 'CFO',               headcount: 20})
            CREATE (:Department {name: 'HR',           head: 'VP of HR',          headcount: 15})
        """)
        print("  Created: Department nodes")

        # ── DATA CLASSIFICATIONS ─────────────────────────────
        session.run("""
            CREATE (:DataClassification {level: 'Confidential'})
            CREATE (:DataClassification {level: 'Internal'})
            CREATE (:DataClassification {level: 'Public'})
        """)
        print("  Created: DataClassification nodes")

        # ── POLICIES ─────────────────────────────────────────
        session.run("""
            MATCH (hr:Department       {name: 'HR'})
            MATCH (security:Department {name: 'Security'})
            MATCH (finance:Department  {name: 'Finance'})
            MATCH (eng:Department      {name: 'Engineering'})

            CREATE (rw:Policy {
                name: 'Remote Work Policy',
                effective_date: '2025-01-01',
                document_source: 'company_policy.txt'
            })
            CREATE (isp:Policy {
                name: 'Information Security Policy',
                effective_date: '2025-01-01',
                document_source: 'security_policy.txt'
            })
            CREATE (fep:Policy {
                name: 'Finance and Expense Policy',
                effective_date: '2025-01-01',
                document_source: 'finance_policy.txt'
            })
            CREATE (es:Policy {
                name: 'Engineering Standards',
                effective_date: '2025-01-01',
                document_source: 'engineering_standards.docx'
            })

            CREATE (hr)-[:GOVERNS]->(rw)
            CREATE (security)-[:GOVERNS]->(isp)
            CREATE (finance)-[:GOVERNS]->(fep)
            CREATE (eng)-[:GOVERNS]->(es)
        """)
        print("  Created: Policy nodes + GOVERNS relationships")

        # ── VENDORS ──────────────────────────────────────────
        session.run("""
            MATCH (eng:Department      {name: 'Engineering'})
            MATCH (security:Department {name: 'Security'})
            MATCH (confidential:DataClassification {level: 'Confidential'})

            CREATE (ch:Vendor {
                name: 'CloudHost Inc.',
                contract_value:  2400000,
                contract_start:  '2025-01-01',
                contract_end:    '2027-12-31',
                services:        'Cloud infrastructure hosting (AWS managed services)',
                sla_uptime:      99.95,
                termination_notice_days: 90,
                soc2_certified:  false
            })
            CREATE (sa:Vendor {
                name: 'SecureAuth Systems',
                contract_value:  180000,
                contract_start:  '2025-03-01',
                contract_end:    '2026-02-28',
                services:        'Identity and access management platform, SSO integration',
                sla_uptime:      99.9,
                auto_renews:     true,
                soc2_certified:  true
            })
            CREATE (dp:Vendor {
                name: 'DataPipe Analytics',
                contract_value:  350000,
                contract_start:  '2025-06-01',
                contract_end:    '2027-05-31',
                services:        'Data pipeline management, ETL processing, real-time analytics dashboard',
                sla_uptime:      99.5,
                termination_notice_days: 60,
                soc2_certified:  false
            })

            CREATE (ch)-[:PROVIDES_SERVICE_TO]->(eng)
            CREATE (sa)-[:PROVIDES_SERVICE_TO]->(security)
            CREATE (dp)-[:PROVIDES_SERVICE_TO]->(eng)

            CREATE (ch)-[:HANDLES_DATA]->(confidential)
            CREATE (sa)-[:HANDLES_DATA]->(confidential)
            CREATE (dp)-[:HANDLES_DATA]->(confidential)
        """)
        session.run("""
            MATCH (sa:Vendor {name: 'SecureAuth Systems'})
            CREATE (cert:Certification {type: 'SOC 2 Type II'})
            CREATE (sa)-[:HAS_CERTIFICATION]->(cert)
        """)
        print("  Created: Vendor nodes + relationships")

        # ── SERVERS ──────────────────────────────────────────
        session.run("""
            CREATE (:Region      {name: 'us-east-1'})
            CREATE (:Region      {name: 'us-west-2'})
            CREATE (:Region      {name: 'eu-west-1'})
            CREATE (:Environment {name: 'production'})
            CREATE (:Environment {name: 'staging'})
            CREATE (:Environment {name: 'development'})
        """)
        session.run("""
            MATCH (use1:Region      {name: 'us-east-1'})
            MATCH (usw2:Region      {name: 'us-west-2'})
            MATCH (euw1:Region      {name: 'eu-west-1'})
            MATCH (prod:Environment {name: 'production'})
            MATCH (stag:Environment {name: 'staging'})
            MATCH (dev:Environment  {name: 'development'})

            CREATE (s1:Server {server_id: 'SRV-001', hostname: 'prod-api-01',    cpu_cores: 32, ram_gb: 128, status: 'active'})
            CREATE (s2:Server {server_id: 'SRV-002', hostname: 'prod-api-02',    cpu_cores: 32, ram_gb: 128, status: 'active'})
            CREATE (s3:Server {server_id: 'SRV-003', hostname: 'prod-db-01',     cpu_cores: 64, ram_gb: 256, status: 'active'})
            CREATE (s4:Server {server_id: 'SRV-004', hostname: 'staging-api-01', cpu_cores: 16, ram_gb: 64,  status: 'active'})
            CREATE (s5:Server {server_id: 'SRV-005', hostname: 'dev-api-01',     cpu_cores: 8,  ram_gb: 32,  status: 'active'})
            CREATE (s6:Server {server_id: 'SRV-006', hostname: 'prod-api-03',    cpu_cores: 32, ram_gb: 128, status: 'active'})
            CREATE (s7:Server {server_id: 'SRV-007', hostname: 'prod-cache-01',  cpu_cores: 16, ram_gb: 64,  status: 'maintenance'})

            CREATE (s1)-[:HOSTED_IN]->(use1)  CREATE (s1)-[:BELONGS_TO]->(prod)
            CREATE (s2)-[:HOSTED_IN]->(use1)  CREATE (s2)-[:BELONGS_TO]->(prod)
            CREATE (s3)-[:HOSTED_IN]->(use1)  CREATE (s3)-[:BELONGS_TO]->(prod)
            CREATE (s4)-[:HOSTED_IN]->(usw2)  CREATE (s4)-[:BELONGS_TO]->(stag)
            CREATE (s5)-[:HOSTED_IN]->(usw2)  CREATE (s5)-[:BELONGS_TO]->(dev)
            CREATE (s6)-[:HOSTED_IN]->(euw1)  CREATE (s6)-[:BELONGS_TO]->(prod)
            CREATE (s7)-[:HOSTED_IN]->(use1)  CREATE (s7)-[:BELONGS_TO]->(prod)
        """)
        print("  Created: Server nodes + relationships")

        # ── SOFTWARE LICENSES ────────────────────────────────
        session.run("""
            CREATE (:License {software: 'GitHub Enterprise', vendor: 'GitHub',     license_type: 'enterprise',   seats: 200, annual_cost: 42000, renewal_date: '2025-06-01'})
            CREATE (:License {software: 'Jira',              vendor: 'Atlassian',  license_type: 'cloud',        seats: 300, annual_cost: 63000, renewal_date: '2025-09-15'})
            CREATE (:License {software: 'Slack Business+',   vendor: 'Salesforce', license_type: 'annual',       seats: 350, annual_cost: 43750, renewal_date: '2025-04-01'})
            CREATE (:License {software: 'Datadog',           vendor: 'Datadog',    license_type: 'enterprise',   seats: 50,  annual_cost: 95000, renewal_date: '2025-12-01'})
            CREATE (:License {software: 'Figma',             vendor: 'Figma',      license_type: 'organization', seats: 40,  annual_cost: 18000, renewal_date: '2025-07-15'})
        """)
        session.run("""
            MATCH (gh:License    {software: 'GitHub Enterprise'})
            MATCH (jira:License  {software: 'Jira'})
            MATCH (slack:License {software: 'Slack Business+'})
            MATCH (dd:License    {software: 'Datadog'})
            MATCH (fig:License   {software: 'Figma'})
            MATCH (eng:Department      {name: 'Engineering'})
            MATCH (sales:Department    {name: 'Sales'})
            MATCH (mktg:Department     {name: 'Marketing'})
            MATCH (security:Department {name: 'Security'})
            MATCH (finance:Department  {name: 'Finance'})
            MATCH (hr:Department       {name: 'HR'})

            CREATE (gh)-[:USED_BY]->(eng)
            CREATE (jira)-[:USED_BY]->(eng)
            CREATE (jira)-[:USED_BY]->(security)
            CREATE (jira)-[:USED_BY]->(finance)
            CREATE (slack)-[:USED_BY]->(eng)
            CREATE (slack)-[:USED_BY]->(sales)
            CREATE (slack)-[:USED_BY]->(mktg)
            CREATE (slack)-[:USED_BY]->(security)
            CREATE (slack)-[:USED_BY]->(finance)
            CREATE (slack)-[:USED_BY]->(hr)
            CREATE (dd)-[:USED_BY]->(eng)
            CREATE (dd)-[:USED_BY]->(security)
            CREATE (fig)-[:USED_BY]->(eng)
            CREATE (fig)-[:USED_BY]->(mktg)
        """)
        print("  Created: License nodes + relationships")

    driver.close()
    print("✅ Knowledge graph built successfully")


# ============================================================
# 2. QUERY THE KNOWLEDGE GRAPH WITH NATURAL LANGUAGE
# ============================================================
def create_graph_qa_chain():
    from neo4j import GraphDatabase
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda

    SCHEMA = """
    Node labels and properties:
    - Department {name, head, headcount}
    - Policy {name, effective_date, document_source}
    - Vendor {name, contract_value, contract_start, contract_end, services, sla_uptime, soc2_certified}
    - DataClassification {level}  ← exact values: 'Confidential', 'Internal', 'Public'
    - Certification {type}
    - Server {server_id, hostname, cpu_cores, ram_gb, status}
    - Region {name}  ← exact values: 'us-east-1', 'us-west-2', 'eu-west-1'
    - Environment {name}  ← exact values: 'production', 'staging', 'development'
    - License {software, vendor, license_type, seats, annual_cost, renewal_date}

    Relationships:
    - (Department)-[:GOVERNS]->(Policy)
    - (Vendor)-[:PROVIDES_SERVICE_TO]->(Department)
    - (Vendor)-[:HANDLES_DATA]->(DataClassification)
    - (Vendor)-[:HAS_CERTIFICATION]->(Certification)
    - (Server)-[:HOSTED_IN]->(Region)
    - (Server)-[:BELONGS_TO]->(Environment)
    - (License)-[:USED_BY]->(Department)

    IMPORTANT: Property values are case-sensitive. Always use exact casing as shown above.
    """

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    cypher_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Neo4j Cypher expert. Generate a Cypher query to answer the question.
Use ONLY the nodes, properties, and relationships defined in the schema below.
Return ONLY the raw Cypher query with no explanation, no markdown, no code fences.

Schema:
{schema}"""),
        ("human", "{question}")
    ])

    answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer the question based strictly on the query results provided.
The query results come directly from the database and are accurate — trust them completely.
Do NOT say you don't have data if results are present. 
Translate the results into a clear, natural language answer."""),
    ("human", "Question: {question}\n\nQuery results: {results}")
])

    def run_chain(inputs):
        question = inputs["query"]

        # Step 1: Generate Cypher
        cypher = (cypher_prompt | llm | StrOutputParser()).invoke({
            "schema": SCHEMA,
            "question": question
        }).strip()
        print(f"\nGenerated Cypher:\n{cypher}")

        # Step 2: Execute against Neo4j
        try:
            with driver.session() as session:
                result = session.run(cypher)
                records = [record.data() for record in result]
        except Exception as e:
            records = []
            print(f"  Cypher error: {e}")
        print(f"Results: {records}")

        # Step 3: Generate natural language answer
        answer = (answer_prompt | llm | StrOutputParser()).invoke({
            "question": question,
            "results": str(records)
        })

        return {
            "result": answer,
            "intermediate_steps": [{"query": cypher, "context": records}]
        }

    return RunnableLambda(run_chain)
# ============================================================
# 3. HYBRID RAG: VECTOR + GRAPH
# ============================================================
def create_hybrid_rag_chain():
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.runnables import RunnableLambda
    from app.ingestion import ingest_all
    from app.chunking import recursive_character_chunking

    # Build vector store
    docs = ingest_all()
    chunks = recursive_character_chunking(docs, chunk_size=500, chunk_overlap=100)
    vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Build graph chain
    graph_chain = create_graph_qa_chain()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    router_prompt = ChatPromptTemplate.from_template(
        """Classify this question into one of three categories:

- "vector": Factual questions about policies, rules, procedures
  (e.g., "What is the password policy?")
- "graph": Questions about relationships, structure, connections
  (e.g., "Which vendors handle confidential data?")
- "both": Complex questions needing both policy details and relationships
  (e.g., "What security certifications do vendors with confidential data access need?")

Question: {question}

Respond with ONLY: vector, graph, or both"""
    )
    router_chain = router_prompt | llm | StrOutputParser()

    synthesis_prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer the question using the context provided.

Question: {question}
Context: {context}

Answer:"""
    )

    def vector_handler(inputs):
        question = inputs["question"]
        docs = vector_retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        answer = (synthesis_prompt | llm | StrOutputParser()).invoke({
            "question": question,
            "context": context
        })
        return {"route": "vector", "answer": answer}

    def graph_handler(inputs):
        question = inputs["question"]
        result = graph_chain.invoke({"query": question})
        return {"route": "graph", "answer": result["result"]}

    def both_handler(inputs):
        question = inputs["question"]
        parallel = RunnableParallel(
            vector_docs=RunnableLambda(lambda q: vector_retriever.invoke(q)),
            graph_result=RunnableLambda(lambda q: graph_chain.invoke({"query": q}))
        )
        results = parallel.invoke(question)
        vector_context = "\n\n".join(d.page_content for d in results["vector_docs"])
        graph_context = results["graph_result"]["result"]
        combined_context = f"[Policy Documents]\n{vector_context}\n\n[Graph Data]\n{graph_context}"
        answer = (synthesis_prompt | llm | StrOutputParser()).invoke({
            "question": question,
            "context": combined_context
        })
        return {"route": "both", "answer": answer}

    def full_chain(inputs):
        question = inputs["question"]
        route = router_chain.invoke({"question": question}).strip().lower()
        print(f"  Routed to: {route}")
        if route == "vector":
            return vector_handler(inputs)
        elif route == "graph":
            return graph_handler(inputs)
        else:
            return both_handler(inputs)

    return RunnableLambda(full_chain)


# ============================================================
# 4. TEST
# ============================================================
if __name__ == "__main__":

    print("=" * 60)
    print("BUILDING KNOWLEDGE GRAPH")
    print("=" * 60)
    build_knowledge_graph()

    print("\n" + "=" * 60)
    print("GRAPH QA TESTS")
    print("=" * 60)
    graph_chain = create_graph_qa_chain()

    graph_questions = [
        "Which departments have more than 50 employees?",
        "Which vendors handle confidential data?",
        "What servers are in the us-east-1 region?",
        "Which vendor's contract expires first?",
        "How many production servers does TechCorp have?",
    ]

    for q in graph_questions:
        print(f"\nQ: {q}")
        result = graph_chain.invoke({"query": q})
        print(f"A: {result['result']}")

    print("\n" + "=" * 60)
    print("HYBRID RAG TESTS")
    print("=" * 60)
    hybrid_chain = create_hybrid_rag_chain()

    hybrid_questions = [
        "What is the minimum password length?",
        "Which vendors have access to confidential data?",
        "What security requirements apply to vendors handling confidential data?",
        "What is the number of environments in neo4j data?"
    ]

    for q in hybrid_questions:
        print(f"\nQ: {q}")
        result = hybrid_chain.invoke({"question": q})
        print(f"Route: {result.get('route', 'N/A')}")
        print(f"A: {result.get('answer', result)}")