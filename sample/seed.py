# seed.py
"""
Pre-load some memories for 'alice' so the chatbot has something to recall.
Run once:  python seed.py
"""

from client import SmritikoshClient

# Sign in as the admin account (which can store memories for any user)
client = SmritikoshClient(username="admin", password="changeme123")

USER = "alice"

memories = [
    "My name is Alice and I work as a machine learning engineer at a Series B startup.",
    "I prefer Python over other languages, especially for data pipelines and ML work.",
    "My favourite editor is Neovim with the lazy.nvim plugin manager.",
    "I am learning Rust in my spare time. I find the borrow checker confusing but rewarding.",
    "I use a MacBook Pro M3 Max for local development.",
    "My team is migrating from PyTorch to JAX for our training infrastructure.",
    "I dislike meetings before 10 am. My most productive hours are 9 pm to midnight.",
    "I recently read 'The Pragmatic Programmer' and found the chapter on orthogonality very useful.",
    "I deployed a RAG pipeline last week using pgvector and LangChain. Latency was higher than expected.",
    "My manager asked me to evaluate Smritikosh as a memory layer for our internal LLM assistant.",
]

print(f"Seeding {len(memories)} memories for user '{USER}'...\n")

for i, text in enumerate(memories, 1):
    result = client.remember(USER, text)
    importance = result.get("importance_score", 0)
    facts = result.get("facts_extracted", 0)
    print(f"  [{i:2d}] importance={importance:.2f}  facts={facts}  \"{text[:60]}...\"")

print("\nDone. Run 'python chatbot.py' to start the chatbot.")
