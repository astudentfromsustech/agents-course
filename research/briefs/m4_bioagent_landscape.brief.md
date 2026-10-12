You are the research leg of a study plan. Do WEB RESEARCH and report findings.

TOPIC: LLM agents for genomics / bioinformatics — the competitor landscape, and whether a
3D-genomics agent exists yet
QUESTION TO ANSWER: What LLM-agent systems for genomics/bioinformatics exist as of 2026-09-30, what
does each actually automate, and has anyone built an agent for 3D genomics (Hi-C / Micro-C / HiChIP /
ChIA-PET analysis, or chromatin polymer/condensate simulation)?
CONTEXT: module 4 of a plan to build "FoldClaw", a 3D-genomics agent whose intended differentiator is
a closed loop between CONTACT DATA and PHASE-SEPARATION / POLYMER SIMULATION. This is a competitive
gap analysis for a resource/application paper.

ALREADY KNOWN — do NOT spend effort re-describing these, they are on the reader's disk. Report only
what is NEW about them (later versions, published venue, benchmark results, criticism):
BioDiscoveryAgent (2024), CellAgent (2024), GenoTEX (2024 benchmark), BIA / bioinformatics-agent
(2024), GeneAgent (Nat Methods 2025, self-verification), BioAgents (Sci Rep 2025), GenoMAS (2025
multi-agent gene expression).

FIND AND REPORT:
1. What has appeared SINCE mid-2025: new bio/omics LLM agents, agentic pipeline builders, and any
   "AI scientist" systems applied to genomics. For each: what it automates, what it is evaluated on,
   and whether code is released.
2. THE KEY QUESTION: any agent specifically for 3D genome organisation / Hi-C-family analysis? Search
   hard and report honestly if the answer is none. Include agentic or LLM-driven tooling for Hi-C
   pipelines, loop/TAD calling, scaffolding/curation, or chromatin simulation.
3. Foundation models for Hi-C (e.g. a Hi-C foundation model reported in Nature Methods around 2026 by
   a group including Wang): what it does, inputs/outputs, and whether it is an agent or a predictor.
   Give the citation and, if open, the repo.
4. Benchmarks for evaluating bio agents: what exists beyond GenoTEX, and what they measure.
5. Published CRITICISM of bio-agent papers — reproducibility, evaluation weakness, multi-agent claims
   that did not replicate. This matters more than the hype; find it.
6. Where credible sources DISAGREE about whether agents help in bioinformatics at all.

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
