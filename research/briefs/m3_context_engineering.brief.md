You are the research leg of a study plan. Do WEB RESEARCH and report findings.

TOPIC: Context engineering for long-horizon agents
QUESTION TO ANSWER: An agent's only state is its context window. What are the actual, evidenced
techniques for managing it across a long multi-step run — compaction/summarization, external memory,
retrieval, subagent delegation, tool-result shaping — and what are the DOCUMENTED failure modes
(context rot, drift, tool-output flooding, lost-in-the-middle, error cascades)?
CONTEXT: module 3 of a plan on building a 3D-genomics agent. The agent's tool outputs are LARGE
scientific artefacts (contact matrices, QC tables, thousands of called loops, MD trajectories), so
observation shaping is the central design problem. Reader knows transformers and the agent loop.
DEPTH: working knowledge — the reader will implement this.

FIND AND REPORT:
1. "Context engineering" as a named practice in 2025-2026: what the term now covers, and the most
   credible primary write-ups (lab engineering blogs count as primary here if they are from the
   model developers; cite exact URLs).
2. Measured degradation with context length: what do benchmarks actually show about performance vs
   used-context-fraction? Name the benchmarks and numbers. Is "context rot" an established effect or
   a blog-coined term — what is the strongest evidence either way?
3. Compaction / summarization strategies for agent transcripts: what works, what is lost, and any
   published comparison of strategies.
4. Delegating to subagents specifically to keep tool output OUT of the main context: what is the
   evidence this helps, and what is the reported cost (coordination failure, lost nuance)?
5. Tool-result shaping: published guidance or results on truncation, pagination, structured summaries,
   returning file handles/paths instead of content.
6. Failure taxonomies for long-horizon agents from real deployments, with sources.
7. State as of 2026-09-30 — flag anything that changed in the last 12 months.

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
- The reader is an expert computational biologist with ML background. Do not explain basics.
- Do NOT write files. Do NOT modify anything. Report only.
