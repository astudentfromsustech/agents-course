## Answer

Treat context as a bounded working set, not durable state. For a 3D-genomics agent, keep raw matrices, loop calls, QC tables, trajectories, and provenance externally; pass the main agent only task-relevant, query-shaped observations plus stable handles for drill-down. This is now the documented meaning of “context engineering”: repeatedly curate the full inference state—instructions, tools, history, and external data—into the smallest high-signal token set. [Anthropic, 2025](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

Use four complementary mechanisms:

1. **Shape every tool result before it enters the transcript.** Return a schema-level synopsis, counts/ranges/QC flags, top-k or paged rows, and artifact paths/IDs with parameter/provenance metadata; provide targeted follow-up tools for a locus, chromosome pair, loop ID, or trajectory frame. Anthropic explicitly recommends pagination, range selection, filtering and truncation, with a 25k-token default cap in Claude Code. [Anthropic tool guidance](https://www.anthropic.com/engineering/writing-tools-for-agents?s=08)
2. **Persist explicit state outside the window.** Maintain a structured run ledger: objectives, completed stages, assertions, derived artifact handles/checksums, unresolved anomalies, and next actions. Retrieve it selectively; do not treat vector similarity as authoritative truth for numerical or provenance claims. [Anthropic context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
3. **Compact completed phases, but preserve pointers and uncertainties.** Summaries retain continuity but are lossy; periodically reset into a fresh context using a verifiable handoff artifact when accumulated framing itself becomes harmful. [Anthropic long-running harness](https://www.anthropic.com/engineering/harness-design-long-running-apps)
4. **Delegate exploratory/verbose work to isolated subagents.** A worker can inspect large outputs and return a bounded report plus artifact references; this reduces main-context pollution, but only works for genuinely separable work with explicit contracts. [Anthropic multi-agent deployment report](https://www.anthropic.com/engineering/multi-agent-research-system)

“Context rot” is a 2025 Chroma technical-report label, not yet an established scholarly term; the underlying degradation is well evidenced by positional-bias and long-context benchmarks. [Chroma](https://www.trychroma.com/research/context-rot) [Lost in the Middle](https://aclanthology.org/anthology-files/anthology-files/anthology-files/pdf/tacl/2024.tacl-1.9.pdf)

## Key findings

- In 2025–26, context engineering covers curation of all tokens available at inference—not merely prompt wording—including message history, tool schemas/results, MCP/external data, compaction, memory, and delegation. [Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

- The best measured evidence is not a universal “performance versus fraction of advertised window” law. It is task/model/position dependent: **Lost in the Middle** varied document count and relevant-document position, finding a U-shaped curve; for GPT‑3.5‑Turbo, middle placement fell below its 56.1% closed-book multi-document-QA score. Increasing retrieved documents from 20 to 50 improved only ~1.5% for GPT‑3.5‑Turbo and ~1% for Claude 1.3 despite higher retriever recall. [Liu et al., TACL 2024](https://aclanthology.org/anthology-files/anthology-files/anthology-files/pdf/tacl/2024.tacl-1.9.pdf)

- **RULER** evaluates 17 long-context models at 4k–128k on retrieval, multi-hop tracing, aggregation, and QA; it is useful for length sweeps, but its score is a suite aggregate rather than a deployment-specific safe-used-context fraction. [Hsieh et al., COLM 2024](https://openreview.net/forum?id=kIoBbc76Sy)

- Chroma’s 2025 technical report tested 18 models. On a cleaned LongMemEval subset, it compared ~300-token focused prompts with full histories averaging ~113k tokens and reported consistently lower full-history performance; on a repeated-word task, it reports degradation across all model families as length rises. This is strong controlled evidence for the phenomenon, but it remains a vendor technical report with image-only headline scores, not a peer-reviewed universal law. [Chroma report](https://www.trychroma.com/research/context-rot)

- Transcript compaction works operationally when it keeps decisions, unresolved bugs, implementation details, and recent files while dropping redundant tool results; the documented failure is that aggressive compaction discards subtle facts whose future relevance was not predictable. [Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

- A peer-reviewed agent comparison supports **hierarchical, subgoal-scoped compaction**: HiAgent replaces finished subgoal traces with summaries and retains details for the current subgoal; across five long-horizon tasks it reports a twofold success-rate increase and 3.8 fewer average steps. [Hu et al., ACL 2025](https://aclanthology.org/2025.acl-long.1575/)

- Evidence for a general ranking of rolling summaries vs RAG vs file memory in action-heavy agents is still thin. LongMemEval measures conversational memory—extraction, temporal reasoning, updates, and abstention—not scientific tool-state fidelity. [LongMemEval](https://arxiv.org/abs/2410.10813)

- External file-based memory is a documented production pattern: write notes/state outside context and retrieve them later; Anthropic describes using structured artifacts for session handoffs and a file-based memory tool. [Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) [Long-running harness](https://www.anthropic.com/engineering/harness-design-long-running-apps)

- Subagents are documented as context isolation: Anthropic’s research workers explore in separate windows and return distilled findings. Its internal research evaluation reported a 90.2% improvement over a single Opus 4 agent, but the comparison also changed token budget/parallelism; token use alone explained 80% of BrowseComp performance variance, so this is not clean evidence that isolation alone caused the gain. [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)

- The reported delegation tax is real: agents use ~4× chat tokens and multi-agent systems ~15×; dependent tasks are a poor fit, and vague briefs caused duplication, gaps, and task misinterpretation. [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)

- For large scientific artifacts, expose `summary`, `filter`, `range`, `page`, and `detail` modes; return natural-language labels where possible; reserve identifiers needed for follow-up calls in detailed mode. Anthropic’s example reduces a detailed 206-token result to 72 tokens in concise mode. [Anthropic tool guidance](https://www.anthropic.com/engineering/writing-tools-for-agents?s=08)

- Tool-output flooding and opaque tool errors are documented failure sources: bounded outputs prevent agents being overwhelmed by logs, and actionable truncation/error messages steer the next call toward filters or pagination. [OpenAI](https://openai.com/index/equip-responses-api-computer-environment/) [Anthropic](https://www.anthropic.com/engineering/writing-tools-for-agents?s=08)

- Real deployment taxonomy: context pollution/rot; lost-in-the-middle positional failure; lossy-summary drift; premature wrap-up (“context anxiety”); duplicated or gap-ridden delegation; runaway search/overspending; synchronous-worker bottlenecks; asynchronous state-consistency and error-propagation problems; and compound error cascades where one failed step changes the trajectory. [Anthropic long-running harness](https://www.anthropic.com/engineering/harness-design-long-running-apps) [Anthropic research system](https://www.anthropic.com/engineering/multi-agent-research-system)

- Change in the last 12 months: model vendors now expose native/server-side compaction. OpenAI documents opaque compaction items, threshold-triggered compaction, and pruning items before the latest compaction item; Anthropic documents compaction, tool-result clearing, file memory, and clean-context resets. [OpenAI API docs](https://developers.openai.com/api/docs/guides/compaction) [Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

## Primary sources

| title | year | venue/type | URL | why it matters |
|---|---:|---|---|---|
| Effective context engineering for AI agents | 2025 | Anthropic engineering report | https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents | Defines the current practice; documents compaction, notes, tool clearing, and subagent isolation. |
| Writing effective tools for AI agents—using AI agents | 2025 | Anthropic engineering report | https://www.anthropic.com/engineering/writing-tools-for-agents?s=08 | Concrete result-shaping guidance: concise/detailed modes, 25k cap, pagination, filtering, truncation, helpful errors. |
| How we built our multi-agent research system | 2025 | Anthropic production engineering report | https://www.anthropic.com/engineering/multi-agent-research-system | Production evidence for isolated workers, cost, delegation failures, and cascading errors. |
| Harness design for long-running application development | 2026 | Anthropic engineering report | https://www.anthropic.com/engineering/harness-design-long-running-apps | Documents context anxiety and why structured handoff plus fresh-context reset may outperform compaction alone. |
| Compaction | 2026 | OpenAI API documentation | https://developers.openai.com/api/docs/guides/compaction | Current native compaction semantics and safe chaining/pruning behavior. |
| Context Rot: How Increasing Input Tokens Impacts LLM Performance | 2025 | Chroma technical report + reproducibility code | https://www.trychroma.com/research/context-rot | Source of the term; controlled 18-model tests including focused vs ~113k-token LongMemEval histories. |
| Lost in the Middle: How Language Models Use Long Contexts | 2024 | TACL | https://aclanthology.org/anthology-files/anthology-files/anthology-files/pdf/tacl/2024.tacl-1.9.pdf | Strong peer-reviewed evidence for positional degradation and diminishing benefit from more retrieved documents. |
| HiAgent: Hierarchical Working Memory Management for Solving Long-Horizon Agent Tasks with LLMs | 2025 | ACL long paper | https://aclanthology.org/2025.acl-long.1575/ | Published agent-task result for subgoal-scoped summaries rather than raw transcript retention. |
| RULER: What’s the Real Context Size of Your Long-Context Language Models? | 2024 | COLM | https://openreview.net/forum?id=kIoBbc76Sy | Standardized 4k–128k context-length stress benchmark. |
| LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory | 2024 | NeurIPS workshop paper | https://arxiv.org/abs/2410.10813 | Useful memory benchmark, but establishes the limits of evidence transfer to procedural scientific agents. |

## Caveats and open questions

- “Context rot” is a Chroma-coined label as far as I could verify; use it descriptively, not as settled terminology. The peer-reviewed underlying constructs are long-context degradation and lost-in-the-middle. [Chroma](https://www.trychroma.com/research/context-rot) [TACL paper](https://aclanthology.org/anthology-files/anthology-files/anthology-files/pdf/tacl/2024.tacl-1.9.pdf)

- I could not verify a public, peer-reviewed benchmark that compares rolling summary, structured file memory, retrieval, and subagent isolation on **large numerical/scientific artifacts** with provenance-sensitive correctness. [unverified]

- Vendor-reported multi-agent gains are confounded by increased total tokens, parallel search, model allocation, and task-specific orchestration; do not attribute the 90.2% result solely to context isolation. [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)

- Semantic retrieval is particularly risky for genomics state: near-duplicate loci, assemblies, parameters, replicate IDs, and derived versions require deterministic identifiers and provenance-aware lookup. This is an implementation inference from the evidence above, not a directly benchmarked claim. [unverified]

- For this module, the missing evaluation is clear: construct a replayable 3D-genomics trace benchmark that varies artifact size, relevant-result position, summary boundaries, retrieval miss rate, and handoff quality; score both biological answer correctness and exact artifact/provenance recovery. [unverified]