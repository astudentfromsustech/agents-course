You are the research leg of a study plan. Do WEB RESEARCH and report findings.

TOPIC: LLM post-training — what each stage adds
QUESTION TO ANSWER: Pretraining produces a next-token text simulator. What do SFT, preference
optimization (RLHF / DPO and successors), and RL-with-verifiable-rewards each ADD on top of it, and
which downstream AGENT behaviours (reliable tool-calling, multi-step instruction following, staying
on task over dozens of steps, declining to fabricate) are attributable to which stage?
CONTEXT: module 1 of a plan on building a domain-specific LLM agent. Reader already understands
transformers, attention, autoregressive decoding, cross-entropy training, and classical ML objectives.
DEPTH: orientation, but technically precise — the reader will read the papers next.

FIND AND REPORT:
1. The canonical stage decomposition as practised in 2025-2026 (pretrain / mid-train / SFT /
   preference optimization / RLVR / distillation). Which stages do frontier labs actually run, and
   what has CHANGED in the last 12 months?
2. RLHF vs DPO vs GRPO and newer variants: what each optimizes, and the empirical verdict on which is
   used in production now. Where do credible sources disagree?
3. RL with verifiable rewards (RLVR): what it is, what it reliably improves, and the evidence on
   whether it adds new capability or sharpens existing capability (this is actively contested — report
   both sides with sources).
4. TOOL USE specifically: is reliable tool-calling a post-training artifact? What is published about
   how tool-use ability is instilled and how it degrades? Any evidence on agentic post-training
   (multi-turn, long-horizon RL) as a distinct stage?
5. Primary sources: papers (title, year, authors, arXiv/DOI), official model/technical reports.
6. State as of 2026-09-30 — flag anything that changed in the last 12 months.

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
