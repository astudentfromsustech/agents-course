You are the research leg of a study plan. Do WEB RESEARCH and report findings.

TOPIC: Building a knowledge base an LLM agent can actually use
QUESTION TO ANSWER: How do you give an agent reliable access to knowledge it was never trained on? For
a corpus of a few hundred domain PDFs (scientific papers plus cloned code repos), what design actually
works as of 2026-09-30 — embedding-based RAG, BM25/hybrid search, an agent that navigates a structured
index with grep/file tools, a knowledge graph, or long-context-instead-of-retrieval?
CONTEXT: module 4 of a course on developing agents. The reader will build this against a real corpus of
~200 scientific PDFs plus ~50 cloned tool repos, with an INDEX.md already describing them. Reader knows
transformers, the agent loop, and context management; is an expert computational biologist who has built
search and indexing systems before. Do not explain embeddings from scratch.
DEPTH: working knowledge — the reader will implement this and measure it.

FIND AND REPORT:
1. THE CENTRAL CONTESTED QUESTION: is a vector store needed at all? Find the actual evidence on
   "agentic search" (an agent using grep/find/file-reading over a well-organised corpus) vs embedding
   retrieval. Who has measured this head to head, on what, and what did they find? Report both sides
   honestly — including the argument that RAG is an artifact of small context windows.
2. Long context vs retrieval: what do measurements show about putting a large corpus in a 1M-token
   window vs retrieving? Cost, latency, and accuracy — with numbers and sources.
3. Chunking: what strategies exist, what each destroys (tables, figure captions, a Methods section
   spanning pages), and any published comparison. Include "contextual retrieval" / late chunking and
   whether the reported gains replicate.
4. Hybrid retrieval and reranking: what the current recommended stack is, and what each stage
   empirically adds. Name specific embedding and reranking models in current use.
5. SCIENTIFIC PDFs specifically: what actually works for parsing papers (layout, multi-column, tables,
   figures, supplements)? Which tools are current, and what is the reported error rate?
6. Knowledge graphs for agents: is GraphRAG-style structure worth its build cost? What does the
   evidence say vs the marketing?
7. Memory vs retrieval: how is agent "memory" distinguished from corpus retrieval in current practice,
   and what is the state of persistent-memory designs?
8. EVALUATING retrieval separately from the agent: recall@k, faithfulness/groundedness, citation
   accuracy, and how to detect an answer grounded in the wrong chunk. What benchmarks exist?
9. Provenance and staleness: published practice for citing sources and handling a corpus that changes.

OUTPUT FORMAT (markdown, no preamble):
## Answer
<direct answer to the question, 200-400 words>
## Key findings
<bulleted; EVERY factual claim carries an inline source URL>
## Primary sources
<table: title | year | venue/type | URL | why it matters>
## Caveats and open questions
<what you could not verify, and what is contested>

RULES:
- Every non-obvious claim needs a URL you actually visited. No URL = mark it "[unverified]".
- If search results are thin, SAY the evidence is thin. Do not pad with plausible-sounding filler.
- Prefer primary sources (papers, official docs, source code) over blog summaries.
- The reader is an expert. Do not explain basics.
- Do NOT write files. Do NOT modify anything. Report only.
