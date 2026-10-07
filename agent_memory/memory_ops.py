#  Memory Operations: Extraction, Consolidation, and Self-Updating Memory
# his lesson explores advanced memory operations that enable AI agents to manage long-running conversations efficiently. As conversations grow, they consume valuable context window space. Without proper management, agents lose important historical context or fail due to token limits.

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

from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings

client = OpenAI()

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-mpnet-base-v2"
)

# Table names for each memory type
CONVERSATIONAL_TABLE = "CONVERSATIONAL_MEMORY"
KNOWLEDGE_BASE_TABLE = "SEMANTIC_MEMORY"
WORKFLOW_TABLE = "WORKFLOW_MEMORY"
TOOLBOX_TABLE = "TOOLBOX_MEMORY"
ENTITY_TABLE = "ENTITY_MEMORY"
SUMMARY_TABLE = "SUMMARY_MEMORY"
TOOL_LOG_TABLE = "TOOL_LOG_MEMORY"


ALL_TABLES = [
    CONVERSATIONAL_TABLE,
    KNOWLEDGE_BASE_TABLE,
    WORKFLOW_TABLE,
    TOOLBOX_TABLE,
    ENTITY_TABLE,
    SUMMARY_TABLE,
    TOOL_LOG_TABLE]

# Drop existing tables to start fresh
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


# Create or retrieve the conversational history table
from helper import create_conversational_history_table, create_tool_log_table
CONVERSATION_HISTORY_TABLE = create_conversational_history_table(database_connection, CONVERSATIONAL_TABLE)
TOOL_LOG_HISTORY_TABLE = create_tool_log_table(database_connection, TOOL_LOG_TABLE)


# Store Manager Wiring: Vector stores are wired through `StoreManager`, then handles are retrieved for downstream use. The storage layer now expands from conversation memory into multi-memory support.

from langchain_oracledb.vectorstores import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from helper import StoreManager

# Create StoreManager instance
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
    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
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

# MemoryManager and Toolbox are composed into the working runtime. Raw stores from the prior step become high-level operations the agent can call consistently.

from helper import MemoryManager, Toolbox

# Initialize the MemoryManager instance
memory_manager = MemoryManager(
    conn=database_connection,
    conversation_table=conversation_table,
    knowledge_base_vs=knowledge_base_vs,
    workflow_vs=workflow_vs,
    toolbox_vs=toolbox_vs,
    entity_vs=entity_vs,
    summary_vs=summary_vs,
    tool_log_table=TOOL_LOG_HISTORY_TABLE
)

# Initialize Toolbox with embedding function
toolbox = Toolbox(memory_manager, client, embedding_model)

print("✅ MemoryManager and Toolbox initialized")

# Part 2: Context Window Management and Summarization

# Large Language Models have finite context windows. When conversations grow long, we face a critical challenge: how do we preserve important information while staying within token limits?

# This section implements the core memory consolidation pipeline:

# Long Conversation → Monitor Usage → Summarize → Store Summary → Mark Original as Processed

# Why This Matters
# Problem 	Solution
# Context overflow crashes the agent 	Monitor token usage and summarize proactively
# Summaries lose important details 	Capture technical, emotional, and entity information
# Can't access original conversation 	Store summary ID links back to original messages
# Re-summarizing already processed messages 	Mark messages with summary_id after processing

# Testing the Memory Pipeline: Now let's verify that our memory consolidation system works end-to-end.

# Step 1: Create a test thread with sample conversations
import time
from helper import SAMPLE_RESEARCH_CONVERSATION

TEST_THREAD_ID = f"test_summary_{int(time.time())}"

# Use the sample conversation from helper.py (30 messages about research papers)
test_messages = SAMPLE_RESEARCH_CONVERSATION

print(f"📝 Creating test thread: {TEST_THREAD_ID}")
print("-" * 50)

for role, content in test_messages:
    memory_manager.write_conversational_memory(content, role, TEST_THREAD_ID)
    print(f"[{role.upper()}] {content[:70]}...")
    time.sleep(0.05)  # Small delay to ensure distinct timestamps

print(f"✅ Added {len(test_messages)} messages to thread")

# Step 2: Monitor Context Usage, Pre-Summary Measurement: Baseline context usage is measured before any compression. This creates a reference point for evaluating the effect of summarization.

# Step 2: Monitor context usage before summarization
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
print("🔄 SUMMARIZING CONVERSATION")
print("=" * 50)

summary_result = summarize_conversation(TEST_THREAD_ID)

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

# Access the function via toolbox._tools_by_name (since decorator returns ID, not function)
expand_fn = toolbox._tools_by_name['expand_summary']
expanded = expand_fn(summary_result['id'])
print(expanded)

# Step 5: Verify - After summarization, unsummarized messages should be empty
print("✅ VERIFICATION - Thread After Summarization")
print("=" * 50)

# High-level check via memory API (should show no unsummarized messages)
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
print("🎉 TEST COMPLETE!")
print("=" * 50)
print("""
Summary of what happened:
1. ✅ Created test conversation with 30 messages
2. ✅ Monitored context window usage
3. ✅ Summarized conversation with a structured LLM prompt
4. ✅ Expanded summary to retrieve original messages with timestamps
5. ✅ Verified source conversation rows are marked with summary_id
""")


