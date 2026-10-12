You are the research leg of a study plan. Do WEB RESEARCH and report findings.

TOPIC: Evaluating agents — general methodology, and for scientific/bioinformatics agents
QUESTION TO ANSWER: How is an LLM agent's capability measured credibly, and what does a defensible
evaluation look like for a scientific domain agent where the "task" is running an analysis pipeline
and interpreting the result?
CONTEXT: module 6 of a plan to build a 3D-genomics agent. This eval is what the build is aimed at, and
it must survive review for a resource/application paper. Reader is experienced at benchmarking
bioinformatics tools (fixed baselines, fixed seeds, paired cases, reported variance).
DEPTH: working knowledge.

FIND AND REPORT:
1. Established agent benchmarks and what each actually measures — coding/SWE agents, tool-use
   benchmarks, computer-use, and science agent benchmarks. Names, URLs, and the metric each reports.
2. METHODOLOGY of trustworthy agent evaluation: pass@k vs pass^k, number of repeats needed given
   run-to-run variance, cost/step accounting, partial credit, and how contamination is handled.
   What is the published guidance, and what is the common criticism of sloppy agent evals?
3. The reported VARIANCE of agent runs on identical tasks — actual numbers, so the reader can size
   how many repeats an honest comparison needs.
4. LLM-as-judge for scientific output: what is validated, where it fails, and what is recommended as
   a check.
5. Evaluating agents that call real, long-running scientific tools: any published protocol for
   distinguishing "the agent chose well" from "the underlying tool is good"? Ablations used to
   attribute credit between scaffold, model, and tool.
6. What reviewers of agent RESOURCE papers (application notes, GigaScience/NAR/Bioinformatics-style)
   ask for: case studies, reproducibility artefacts, comparison baselines. Any editorial guidance.

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
