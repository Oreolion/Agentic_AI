# =============================================================================
# MEMORY-AWARE AGENT (complete capstone — consolidated tutorial file)
#
# This is the top-level module that brings together all prior lessons:
#   Part 1: Memory manager — database, vector stores, MemoryManager
#   Part 2: Semantic tool use — tool registration & retrieval
#   Part 3: Knowledge-base ingestion from HuggingFace arXiv dataset
#   Part 4: Memory operations — context management & summarization
#   Part 5: The memory-aware agent loop (+ demo run at the bottom)
#
# LLM backend: NVIDIA NIM (OpenAI-compatible API) using NVIDIA_API_KEY.
# The model is controlled by AGENT_LLM_MODEL in .env (shared with helper.py).
# Embeddings remain local via sentence-transformers (no external API needed).
# =============================================================================

# ============================================================================
# PART 1: Setting Up Database, Vector Stores and Embedding Models
# ============================================================================

from helper import suppress_warnings

# Warning control
suppress_warnings()

from helper import load_env, setup_oracle_database, connect_to_oracle

load_env()

# One-time admin setup: configures tablespace, vector memory, and VECTOR user
setup_oracle_database()

# Connect as the VECTOR user for all subsequent operations
database_connection = connect_to_oracle(
    user="VECTOR",
    password="VectorPwd_2025",
    dsn="127.0.0.1:1521/FREEPDB1",
    program="devrel.deeplearning.course_1",
)

print("Using user:", database_connection.username)

import os
import json as json_lib
from typing import Any
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# LLM client — NVIDIA NIM (OpenAI-compatible API).
# Requires NVIDIA_API_KEY in .env; model shared with helper.py via AGENT_LLM_MODEL.
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
LLM_MODEL = os.getenv("AGENT_LLM_MODEL", "nvidia/nemotron-3-super-120b-a12b")

if not os.getenv("NVIDIA_API_KEY"):
    raise RuntimeError("NVIDIA_API_KEY not found — add it to your .env file.")

# timeout + max_retries: fail fast (seconds) if a model is gated/unresponsive
# instead of hanging the agent on silent retries.
client = OpenAI(
    base_url=NVIDIA_BASE_URL,
    api_key=os.getenv("NVIDIA_API_KEY"),
    timeout=60,
    max_retries=1,
)

# Initialize the embedding model (shared by all vector stores and the Toolbox)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-mpnet-base-v2"
)

# Table names for each memory type
CONVERSATIONAL_TABLE   = "CONVERSATIONAL_MEMORY"  # Episodic memory
KNOWLEDGE_BASE_TABLE   = "SEMANTIC_MEMORY"        # Semantic memory
WORKFLOW_TABLE         = "WORKFLOW_MEMORY"        # Procedural memory
TOOLBOX_TABLE          = "TOOLBOX_MEMORY"         # Procedural memory
ENTITY_TABLE           = "ENTITY_MEMORY"          # Semantic memory
SUMMARY_TABLE          = "SUMMARY_MEMORY"         # Semantic memory
TOOL_LOG_TABLE         = "TOOL_LOG_MEMORY"        # Tool execution logs

ALL_TABLES = [
    CONVERSATIONAL_TABLE,
    KNOWLEDGE_BASE_TABLE,
    WORKFLOW_TABLE,
    TOOLBOX_TABLE,
    ENTITY_TABLE,
    SUMMARY_TABLE,
    TOOL_LOG_TABLE,
]

# Drop existing tables to start fresh (lab reset semantics)
for table in ALL_TABLES:
    try:
        with database_connection.cursor() as cur:
            cur.execute(f"DROP TABLE {table} PURGE")
            print(f"  - {table} (dropped)")
    except Exception as e:
        if "ORA-00942" in str(e):
            print(f"  - {table} (not exists)")
        else:
            print(f"  ✗ {table}: {e}")

database_connection.commit()

# Create the SQL memory tables (helper versions are create-if-not-exists;
# the drop loop above guarantees a fresh create here)
from helper import create_conversational_history_table, create_tool_log_table

CONVERSATION_HISTORY_TABLE = create_conversational_history_table(database_connection, CONVERSATIONAL_TABLE)
TOOL_LOG_HISTORY_TABLE = create_tool_log_table(database_connection, TOOL_LOG_TABLE)

# Create vector stores for each memory type via helper's StoreManager
from langchain_community.vectorstores.utils import DistanceStrategy
from helper import StoreManager

store_manager = StoreManager(
    client=database_connection,
    embedding_function=embedding_model,
    table_names={
        'knowledge_base': KNOWLEDGE_BASE_TABLE,
        'workflow': WORKFLOW_TABLE,
        'toolbox': TOOLBOX_TABLE,
        'entity': ENTITY_TABLE,
        'summary': SUMMARY_TABLE,
    },
    distance_strategy=DistanceStrategy.COSINE,
    conversational_table=CONVERSATION_HISTORY_TABLE,
    tool_log_table=TOOL_LOG_HISTORY_TABLE,
)

# Get all stores via the manager
conversation_table = store_manager.get_conversational_table()
knowledge_base_vs = store_manager.get_knowledge_base_store()
workflow_vs = store_manager.get_workflow_store()
toolbox_vs = store_manager.get_toolbox_store()
entity_vs = store_manager.get_entity_store()
summary_vs = store_manager.get_summary_store()
tool_log_table = store_manager.get_tool_log_table()

print("✅ All stores loaded via StoreManager")

# Creating vector indexes (COSINE, matching the stores' distance strategy)
from helper import safe_create_index

print("Creating vector indexes...")
safe_create_index(database_connection, knowledge_base_vs, "knowledge_base_vs_ivf")
safe_create_index(database_connection, workflow_vs, "workflow_vs_ivf")
safe_create_index(database_connection, toolbox_vs, "toolbox_vs_ivf")
safe_create_index(database_connection, entity_vs, "entity_vs_ivf")
safe_create_index(database_connection, summary_vs, "summary_vs_ivf")
print("All indexes created!")

# Initialize the MemoryManager — unified read/write across all memory types
from helper import MemoryManager, Toolbox

memory_manager = MemoryManager(
    conn=database_connection,
    conversation_table=CONVERSATION_HISTORY_TABLE,
    knowledge_base_vs=knowledge_base_vs,
    workflow_vs=workflow_vs,
    toolbox_vs=toolbox_vs,
    entity_vs=entity_vs,
    summary_vs=summary_vs,
    tool_log_table=TOOL_LOG_HISTORY_TABLE,
)

# Initialize Toolbox with embedding function (semantic tool storage/retrieval)
toolbox = Toolbox(memory_manager, client, embedding_model, model=LLM_MODEL)

print("✅ MemoryManager and Toolbox initialized")


# ============================================================================
# PART 2: Semantic Tool Use — Tool Registration and Retrieval
# ============================================================================

# Tools registered in this part:
#   read_toolbox                  - agent-callable semantic search over the toolbox
#   search_tavily                 - web search that persists results to the KB
#   get_current_time              - current date/time
#   arxiv_search_candidates       - arXiv discovery (IDs, titles, abstracts)
#   fetch_and_save_paper_to_kb_db - full paper download, chunking, KB storage
#   expand_summary / summarize_and_store (from helper.register_summary_tools)

# Expose toolbox retrieval as an agent-callable skill
@toolbox.register_tool(augment=True)
def read_toolbox(query: str, k: int = 3) -> list[dict]:
    """
    Search the toolbox for functions that can help solve a problem or complete a task.

    Use this tool when:
    - You encounter an error or unexpected output and need a different approach
    - The currently available tools don't seem sufficient for the task
    - You need to discover what capabilities are available for a specific problem
    - You want to find alternative functions that might handle edge cases better

    Args:
        query: A natural language description of what you're trying to accomplish
               or the problem you're trying to solve.
        k: Number of relevant tools to return (default: 3)

    Returns:
        A list of tool definitions that semantically match your query.
    """
    return memory_manager.read_toolbox(query, k=k)


# --- Web Access with Tavily (search-and-store pattern) ---
# Skipped gracefully when TAVILY_API_KEY is not configured.
import inspect
from datetime import datetime

if os.getenv("TAVILY_API_KEY"):
    from tavily import TavilyClient

    tavily_client = TavilyClient()

    @toolbox.register_tool(augment=True)
    def search_tavily(query: str, max_results: int = 5):
        """
        Use this function to search the web and store the results in the knowledge base.
        """
        response = tavily_client.search(query=query, max_results=max_results)
        results = response.get("results", [])

        # Write each result to the knowledge base (agent learns from its searches)
        for result in results:
            text = f"Title: {result.get('title', '')}\nContent: {result.get('content', '')}\nURL: {result.get('url', '')}"
            metadata = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "score": result.get("score", 0),
                "source_type": "tavily_search",
                "query": query,
                "timestamp": datetime.now().isoformat(),
            }
            memory_manager.write_knowledge_base(text, metadata)

        return results

    # Augmented vs. original docstring demo (uses search_tavily's registration)
    original = ("Use this function to search the web"
                " and store the results in the"
                " knowledge base.")
    fn = toolbox._tools_by_name["search_tavily"]
    source = inspect.getsource(fn)

    print("ORIGINAL DOCSTRING:")
    print(f'  "{original}"')
    print()
    augmented = toolbox._augment_docstring(original, source)
    print("AUGMENTED DOCSTRING (LLM-enhanced):")
    print(f"  {augmented}")
else:
    print("⚠️ TAVILY_API_KEY not set — skipping search_tavily registration")


@toolbox.register_tool(augment=True)
def get_current_time(detailed: bool = False) -> str:
    """
    Returns the current time.

    Args:
        detailed: If True, returns detailed format with microseconds

    Returns:
        str: Current time as formatted string
    """
    if detailed:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    else:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --- arXiv candidate retrieval (fast discovery before expensive ingestion) ---
import json
from urllib.parse import urlparse
from langchain_community.retrievers import ArxivRetriever

arxiv_retriever = ArxivRetriever(  # pyright: ignore[reportCallIssue] -- langchain fills arxiv_search/arxiv_exceptions via pydantic validator
    load_max_docs=8,
    get_full_documents=False,
    doc_content_chars_max=4000,
)


def _arxiv_id_from_entry_id(entry_id: str) -> str:
    """Convert 'http://arxiv.org/abs/2310.08560v2' -> '2310.08560v2'"""
    if not entry_id:
        return ""
    path = urlparse(entry_id).path
    return path.split("/abs/")[-1].strip("/")


@toolbox.register_tool(augment=False)
def arxiv_search_candidates(query: str, k: int = 5) -> str:
    """
    Search arXiv and return a JSON list of candidate papers with IDs + metadata.

    Output schema (JSON string):
    [
      {
        "arxiv_id": "2310.08560v2",
        "entry_id": "http://arxiv.org/abs/2310.08560v2",
        "title": "...",
        "authors": "...",
        "published": "2024-02-12",
        "abstract": "..."
      },
      ...
    ]
    """
    docs = arxiv_retriever.invoke(query)
    candidates = []
    for d in (docs or [])[:k]:
        meta = d.metadata or {}
        entry_id = meta.get("Entry ID", "")
        candidates.append({
            "arxiv_id": _arxiv_id_from_entry_id(entry_id),
            "entry_id": entry_id,
            "title": meta.get("Title", ""),
            "authors": meta.get("Authors", ""),
            "published": str(meta.get("Published", "")),
            "abstract": (d.page_content or "")[:2500],
        })
    return json.dumps(candidates, ensure_ascii=False, indent=2)


# --- Deep ingestion: fetch, chunk, and persist full papers ---
from datetime import timezone
from langchain_community.document_loaders import ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


@toolbox.register_tool(augment=True)
def fetch_and_save_paper_to_kb_db(
    arxiv_id: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> str:
    """
    Fetch full arXiv paper text (PDF -> text) and store it into the OracleVS
    knowledge base table as chunked records (avoids routing full content via the LLM).
    """
    loader = ArxivLoader(
        query=arxiv_id,
        load_max_docs=1,
        doc_content_chars_max=None,
    )
    docs = loader.load()
    if not docs:
        return f"No documents found for arXiv id: {arxiv_id}"

    doc = docs[0]
    title = (
        doc.metadata.get("Title")
        or doc.metadata.get("title")
        or f"arXiv {arxiv_id}"
    )
    entry_id = doc.metadata.get("Entry ID") or doc.metadata.get("entry_id") or ""
    published = doc.metadata.get("Published") or doc.metadata.get("published") or ""
    authors = doc.metadata.get("Authors") or doc.metadata.get("authors") or ""

    full_text = doc.page_content or ""
    if not full_text.strip():
        return f"Loaded arXiv {arxiv_id} but extracted empty text (PDF parsing issue)."

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(full_text)

    ts_utc = datetime.now(timezone.utc).isoformat()
    metadatas = []
    for i in range(len(chunks)):
        metadatas.append({
            "source": "arxiv",
            "arxiv_id": arxiv_id,
            "title": title,
            "entry_id": entry_id,
            "published": str(published),
            "authors": str(authors),
            "chunk_id": i,
            "num_chunks": len(chunks),
            "ingested_ts_utc": ts_utc,
        })

    memory_manager.write_knowledge_base(chunks, metadatas)

    return (
        f"Saved arXiv {arxiv_id} to {KNOWLEDGE_BASE_TABLE}: "
        f"{len(chunks)} chunks (title: {title})."
    )


# Summary tools (expand_summary, summarize_and_store) — registered ONCE here;
# this replaces memory_agent.py's register_common_tools call, which would have
# re-registered the arxiv/time tools already defined above.
from helper import register_summary_tools

register_summary_tools(toolbox=toolbox, memory_manager=memory_manager, llm_client=client)

# Validate semantic tool retrieval
import pprint

retrieved_tools = memory_manager.read_toolbox("Get more details on a paper on AI", k=1)
pprint.pprint(retrieved_tools)


# ============================================================================
# PART 3: Knowledge Base Ingestion (ArxivScout seed data)
# ============================================================================

# Stream arXiv paper records from the nick007x/arxiv-papers dataset and write
# each entry into the knowledge base with extracted fields stored as metadata.
from datasets import load_dataset
from itertools import islice

ds = load_dataset("nick007x/arxiv-papers", split="train", streaming=True)

for paper in islice(ds, 100):
    title = (paper.get("title") or "").strip()
    abstract = (paper.get("abstract") or "").strip()
    subjects = (paper.get("subjects") or paper.get("primary_subject") or "").strip()
    submission_date = (paper.get("submission_date") or "").strip()

    if not (title or abstract or subjects):
        continue

    text = "\n".join([part for part in (title, subjects, abstract) if part])

    memory_manager.write_knowledge_base(
        text=text,
        metadata={
            "arxiv_id": paper.get("arxiv_id"),
            "title": title,
            "subjects": subjects,
            "abstract": abstract,
            "submission_date": submission_date,
        },
    )

results = memory_manager.read_knowledge_base(query="space exploration")
print(results)


# ============================================================================
# PART 4: Memory Operations — Extraction, Consolidation, Self-Updating Memory
# ============================================================================

# Core memory consolidation pipeline:
#   Long Conversation → Monitor Usage → Summarize → Store Summary → Mark Original as Processed

import time
from helper import (
    calculate_context_usage,
    monitor_context_window,
    summarise_context_window,
    offload_to_summary,
    summarize_conversation,
    SAMPLE_RESEARCH_CONVERSATION,
)

print("✅ Context management functions loaded from helper.py")

# Step 1: Create a test thread with the sample conversation (30 messages)
TEST_THREAD_ID = f"test_summary_{int(time.time())}"
test_messages = SAMPLE_RESEARCH_CONVERSATION

print(f"📝 Creating test thread: {TEST_THREAD_ID}")
print("-" * 50)

for role, content in test_messages:
    memory_manager.write_conversational_memory(content, role, TEST_THREAD_ID)
    print(f"[{role.upper()}] {content[:70]}...")
    time.sleep(0.05)  # Small delay to ensure distinct timestamps

print(f"✅ Added {len(test_messages)} messages to thread")

# Step 2: Monitor context usage before summarization (baseline measurement)
current_context = memory_manager.read_conversational_memory(TEST_THREAD_ID, limit=100)

print("📊 CONTEXT WINDOW MONITOR (Before Summarization)")
print("=" * 50)
usage = monitor_context_window(current_context)
print(f"  Tokens: {usage['tokens']:,}")
print(f"  Max: {usage['max']:,}")
print(f"  Usage: {usage['percent']}%")
print(f"  Status: {usage['status'].upper()}")
print()
print("📄 Current conversation content:")
print("-" * 50)
print(current_context)

# Step 3: Summarize the conversation
# NOTE: helper's summarize_conversation requires (thread_id, memory_manager, llm_client)
print("🔄 SUMMARIZING CONVERSATION")
print("=" * 50)

summary_result = summarize_conversation(TEST_THREAD_ID, memory_manager, client)

print(f"\n📋 Summary Result:")
print(f"  ID: {summary_result['id']}")
print(f"  Description: {summary_result['description']}")
print(f"\n📝 Full Summary (for new context window):")
print("-" * 50)
print(summary_result['summary'])

# Step 4: Expand the summary to retrieve original conversations
print("🔍 EXPANDING SUMMARY - Retrieving Original Conversations")
print("=" * 50)
print(f"Summary ID: {summary_result['id']}")
print()

# expand_summary was registered in Part 2 via register_summary_tools
expand_fn = toolbox._tools_by_name['expand_summary']
expanded = expand_fn(summary_result['id'])
print(expanded)

# Step 5: Verify — after summarization, unsummarized messages should be empty
print("✅ VERIFICATION - Thread After Summarization")
print("=" * 50)

remaining = memory_manager.read_conversational_memory(TEST_THREAD_ID, limit=100)
print("Unsummarized messages in thread (memory API):")
print(remaining)

# Ground-truth check at DB row level
with memory_manager.conn.cursor() as cur:
    cur.execute(f"""
        SELECT COUNT(*)
        FROM {memory_manager.conversation_table}
        WHERE thread_id = :thread_id AND summary_id IS NULL
    """, {"thread_id": TEST_THREAD_ID})
    unsummarized_count = cur.fetchone()[0]

    cur.execute(f"""
        SELECT COUNT(*)
        FROM {memory_manager.conversation_table}
        WHERE thread_id = :thread_id AND summary_id IS NOT NULL
    """, {"thread_id": TEST_THREAD_ID})
    summarized_count = cur.fetchone()[0]

    cur.execute(f"""
        SELECT DISTINCT summary_id
        FROM {memory_manager.conversation_table}
        WHERE thread_id = :thread_id AND summary_id IS NOT NULL
        ORDER BY summary_id
    """, {"thread_id": TEST_THREAD_ID})
    summary_ids = [row[0] for row in cur.fetchall()]

print(f"\nDB verification:")
print(f"  Unsummarized rows: {unsummarized_count}")
print(f"  Summarized rows: {summarized_count}")
print(f"  Summary IDs applied: {summary_ids}")

if unsummarized_count == 0 and summarized_count > 0:
    print("✅ PASS: conversation units summarized and tagged with summary_id")
else:
    print("⚠️ CHECK: expected 0 unsummarized rows and >0 summarized rows")

print("\n" + "=" * 50)
print("🎉 MEMORY OPS TEST COMPLETE!")
print("=" * 50)


# ============================================================================
# PART 5: The Memory-Aware Agent Loop
# ============================================================================

# The agent loop orchestrates memory operations at each step:
#   User Query
#       ↓
#   1. BUILD CONTEXT — Read from all memory stores
#   2. CHECK USAGE — Monitor token count, summarize if >80%
#   3. SELECT TOOLS — Semantic search for relevant tools
#   4. EXECUTE — LLM reasoning + tool calls
#   5. PERSIST — Save conversation, workflow, entities
#       ↓
#   Final Answer

AGENT_SYSTEM_PROMPT = """
# Role
You are a memory-aware agentic research assistant with access to tools.

# Context Window Structure (Partitioned Segments)
The user input is a partitioned context window. It contains a `# Question` section followed by memory segments.
Treat each segment as a distinct memory store with a specific purpose:
- `## Conversation Memory`
- `## Knowledge Base Memory`
- `## Workflow Memory`
- `## Entity Memory`
- `## Summary Memory`

# Memory Store Semantics
- Conversation Memory: Recent thread-level dialogue and instructions. Use it for continuity, user preferences, and unresolved requests.
- Knowledge Base Memory: Retrieved documents/passages. Use it to ground factual and technical claims.
- Workflow Memory: Prior execution patterns and step sequences. Use it to plan tool usage; adapt patterns, do not copy blindly.
- Entity Memory: Named people/orgs/systems and descriptors. Use it to disambiguate references and keep naming consistent.
- Summary Memory: Compressed older context represented by summary IDs. When thread-scoped summaries exist, prefer summaries for the active thread_id.

# Summary Expansion Policy
If critical detail is only present in Summary Memory or appears ambiguous, call `expand_summary(summary_id)` before relying on it.

# Operating Rules
1. Start with the provided memory segments before using tools.
2. If segments conflict, prioritize: current `# Question` > latest Conversation Memory > Knowledge Base evidence > older summaries/workflows.
3. Use only the tools provided in this turn and choose the minimum necessary tool calls.
4. If memory is insufficient, state what is missing and then use an appropriate tool.
5. For conversation compaction, use `summarize_and_store` with `thread_id` so source conversation units are marked as summarized.
"""


def execute_tool(tool_name: str, tool_args: dict, current_thread_id: str | None = None) -> str:
    """Execute a tool by looking it up in the toolbox."""

    if tool_name not in toolbox._tools_by_name:
        return f"Error: Tool '{tool_name}' not found"

    args = dict(tool_args or {})

    # Ensure conversation summarization marks source rows in the active thread.
    if tool_name == "summarize_and_store" and "thread_id" not in args and current_thread_id is not None:
        args["thread_id"] = str(current_thread_id)

    return str(toolbox._tools_by_name[tool_name](**args) or "Done")


def call_openai_chat(messages: list, tools: list | None = None, model: str = LLM_MODEL):
    """Call OpenAI Chat Completions API with tools."""
    kwargs = {"model": model, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return client.chat.completions.create(**kwargs)


# ==================== MAIN AGENT LOOP ====================
def call_agent(query: str, thread_id: str = "1", max_iterations: int = 10) -> str:
    """Agent loop with context window monitoring and summarization."""
    thread_id = str(thread_id)
    steps = []
    summaries = []  # Track created summaries

    # 1. Build context from memory
    print("\n" + "=" * 50)
    print("🧠 BUILDING CONTEXT...")

    memory_context = ""
    memory_context += memory_manager.read_conversational_memory(thread_id) + "\n\n"
    memory_context += memory_manager.read_knowledge_base(query) + "\n\n"
    memory_context += memory_manager.read_workflow(query) + "\n\n"
    memory_context += memory_manager.read_entity(query) + "\n\n"
    memory_context += memory_manager.read_summary_context(query, thread_id=thread_id) + "\n\n"

    # 2. Check context usage - summarize if >80%
    usage = calculate_context_usage(memory_context)
    print(f"📊 Context: {usage['percent']}% ({usage['tokens']}/{usage['max']} tokens)")

    if usage['percent'] > 80:
        print("⚠️ Context >80% - offloading conversation context to summary memory...")
        memory_context, summaries = offload_to_summary(
            memory_context,
            memory_manager,
            client,
            thread_id=thread_id,
        )
        if summaries:
            print(f"🧾 Created {len(summaries)} summary reference(s): {[s['id'] for s in summaries]}")
        usage = calculate_context_usage(memory_context)
        print(f"📊 After offload: {usage['percent']}% ({usage['tokens']}/{usage['max']} tokens)")

    # Prepend the query (always preserved, never summarized)
    context = f"# Question\n{query}\n\n{memory_context}"

    print("====CONTEXT WINDOW=====\n")
    print(context)

    # 3. Get tools
    dynamic_tools = memory_manager.read_toolbox(query, k=5)
    print(f"🔧 Tools: {[t['function']['name'] for t in dynamic_tools]}")

    # 4. Store user message & extract entities
    memory_manager.write_conversational_memory(query, "user", thread_id)
    try:
        memory_manager.write_entity("", "", "", llm_client=client, text=query)
    except Exception:
        pass

    # 5. Agent loop
    messages: list[dict[str, Any]] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}, {"role": "user", "content": context}]
    final_answer = ""

    print("\n🤖 AGENT LOOP")
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        response = call_openai_chat(messages, tools=dynamic_tools)
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]})

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json_lib.loads(tc.function.arguments)
                args_display = {k: (v[:50] + '...' if isinstance(v, str) and len(v) > 50 else v)
                                for k, v in tool_args.items()}
                print(f"🛠️ {tool_name}({args_display})")

                try:
                    result = execute_tool(tool_name, tool_args, current_thread_id=thread_id)
                    status = "success"
                    error_message = None
                    steps.append(f"{tool_name}({args_display}) → success")
                except Exception as e:
                    result = f"Error: {e}"
                    status = "failed"
                    error_message = str(e)
                    steps.append(f"{tool_name}({args_display}) → failed")

                # Persist full tool output to TOOL_LOG_MEMORY
                log_id = memory_manager.write_tool_log(
                    thread_id=thread_id,
                    tool_call_id=tc.id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    result=result,
                    status=status,
                    error_message=error_message,
                    metadata={"iteration": iteration + 1},
                )

                # Next call gets only the immediate tool result (bounded for context control)
                if len(result) > 3000:
                    result_for_llm = result[:3000] + f"\n\n[Truncated for context. Full output saved in TOOL_LOG_MEMORY as log_id: {log_id}]"
                else:
                    result_for_llm = result

                result_display = result_for_llm[:200] + "..." if len(result_for_llm) > 200 else result_for_llm
                print(f"   → {result_display}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_for_llm})
        else:
            final_answer = msg.content or ""
            print(f"\n✅ DONE ({len(steps)} tool calls)")
            break
    else:
        print(f"\n⚠️ WARNING: Max iterations ({max_iterations}) reached without final answer")
        final_answer = "I was unable to complete the request within the allowed iterations."

    # 6. Save workflow & entities
    if steps:
        memory_manager.write_workflow(query, steps, final_answer)
    try:
        memory_manager.write_entity("", "", "", llm_client=client, text=final_answer)
    except Exception:
        pass
    memory_manager.write_conversational_memory(final_answer, "assistant", thread_id)

    print("\n" + "=" * 50 + f"\n💬 ANSWER:\n{final_answer}\n" + "=" * 50)
    return final_answer


# Demo run: builds context from memory, uses tools, persists each interaction
call_agent("Can you get me the paper MemGPT", thread_id="50000")
call_agent("Can you save the content of the paper", thread_id="50000")
call_agent("What are the main key takeaways from the paper", thread_id="50000")
call_agent("Summarize the converstation so far using your tool", thread_id="50000")
call_agent("What was my first question?", thread_id="50000")
