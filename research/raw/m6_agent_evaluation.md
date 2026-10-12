## Answer

A credible capability claim is a property of a fully specified agent–environment system, not of the base LLM alone: freeze the model/version, scaffold, prompts, tool registry and versions, data snapshot, hardware, budgets, and grader; then evaluate pre-registered tasks in resettable containers with executable checks wherever possible. For a 3D-genomics agent, score the analysis as a chain: input/QC validity, chosen workflow and parameters, successful execution, quantitative agreement with a blinded reference or biological truth, and evidence-grounded interpretation. This avoids giving full credit for a plausible narrative produced after an invalid pipeline. ScienceAgentBench makes the relevant precedent: it assesses generated program, execution result, and cost; BiomniBench argues for expert, task-specific trajectory rubrics in biomedical analysis. [ScienceAgentBench](https://arxiv.org/abs/2410.05080) [BiomniBench](https://www.biorxiv.org/content/10.64898/2026.05.12.724604v2)

Report single-rollout success, but do not stop there. `pass@k` means at least one success in \(k\) independent attempts—an oracle/best-of-\(k\) capability quantity; `pass^k` means all \(k\) attempts succeed and is the deployment-reliability quantity. τ-bench found state-of-the-art function-calling agents below 50% task success and retail `pass^8 <25%`, a concrete warning that a successful demo does not establish repeatability. [τ-bench](https://openreview.net/pdf?id=roNSXZpUDN)

Use paired, repeated rollouts for every compared condition, preserve raw trajectories, and calculate uncertainty by task-clustered bootstrap (not a naïve pooled binomial CI). There is no defensible universal repeat count: choose it from a pre-specified minimally important effect and pilot variance; 5–10 rollouts/task is a reasonable minimum screening design, while a Bernoulli rate near 0.5 requires about 96 independent observations for a ±10-point 95% margin and 384 for ±5 points. Include cost, wall time, tokens, tool calls, failures/timeouts, and cost per valid completed analysis—not only success rate.

## Key findings

- **Coding/SWE:** [SWE-bench](https://arxiv.org/abs/2310.06770) measures the percentage of real GitHub issues for which the agent’s repository patch passes the task test suite (“resolved rate”); it measures end-state repair, not necessarily maintainability or reasoning validity.

- **Tool-use:** [ToolBench](https://github.com/lchvelidze/toolbench) reports API/function-call execution success across heterogeneous tools; [τ-bench](https://arxiv.org/abs/2406.12045) instead checks whether an interactive agent reaches an annotated final database state while following policy, and reports `pass^k` reliability.

- **Computer use:** [WebArena](https://webarena.dev/og/) measures completion of realistic browser tasks in self-hosted sites; [OSWorld](https://os-world.github.io/) measures execution-based task success in real desktop/web applications and publishes success rate (and, on its leaderboard, mean ± SD where repeated). Neither is a scientific-validity benchmark.

- **Scientific analysis agents:** [ScienceAgentBench](https://github.com/OSU-NLP-Group/ScienceAgentBench) contains 102 tasks extracted from 44 papers in bioinformatics, chemoinformatics, cognitive neuroscience and geoscience; it evaluates self-contained Python programs, execution outputs, and cost. [BLADE](https://arxiv.org/abs/2408.09667) assesses data-analysis choices by computationally matching analysis representations to reference ground truth. [BixBench](https://arxiv.org/abs/2503.00096) provides >50 computational-biology scenarios and nearly 300 open-answer questions spanning analysis trajectories and interpretation.

- **For 3D genomics, the endpoint should be composite:** deterministic checks should verify input identity, genome build, valid pairs/QC, blacklist and normalization choices, expected files and successful tool exit/output; domain metrics should compare contact maps, compartments/TADs/loops or differential calls against fixed references; expert scoring should assess whether claims state uncertainty and are supported by the generated figures and logs. This is a proposed protocol, not a published 3D-genomics-specific standard. [unverified]

- **Partial credit is necessary but must be structured.** BiomniBench’s process-level, expert-designed task rubrics were motivated by the fact that outcome-only scoring can reward memorization/reward hacking and reject valid alternative analyses. Score independently auditable dimensions rather than a single holistic “correct” label. [BiomniBench](https://www.biorxiv.org/content/10.64898/2026.05.12.724604v2)

- **Separate capability from reliability.** `pass@k` is “at least one success”; the unbiased estimator used in code evaluation is \(1-\binom{n-c}{k}/\binom{n}{k}\), where \(n\) rollouts yield \(c\) correct ones. `pass^k` is success on every repeated rollout of the same task. [Chen et al., 2021](https://arxiv.org/abs/2107.03374) [τ-bench](https://openreview.net/pdf?id=roNSXZpUDN)

- **Published repeat evidence is still thin.** τ-bench explicitly ran >40 GPT-4-Turbo trials per task while validating ambiguous/low-success tasks, and reports `<50%` success for then-state-of-the-art function-calling agents but `<25% pass^8` in retail—an observed large reliability gap, not a small error bar. [τ-bench paper](https://openreview.net/pdf?id=roNSXZpUDN)

- **Do not report a winner from one stochastic run.** Compare matched task × seed rollouts; estimate the condition difference with a task-clustered paired bootstrap, include all failed infrastructure and timeout runs by a predeclared rule, and report task-level outcome matrices. This is statistical best practice extrapolated to agents; no primary source found that establishes a universal “N repeats” rule. [unverified]

- **Account for resources as outcomes.** ScienceAgentBench explicitly includes cost alongside program and execution evaluation; report input/output/reasoning tokens where available, model/API cost, tool CPU/GPU time, wall-clock time, number of steps/tool calls, retry count, and budget-exhaustion rate. [ScienceAgentBench](https://arxiv.org/abs/2410.05080)

- **Contamination is a live threat, especially for public-paper tasks and GitHub-derived benchmarks.** SWE-bench’s own Verified announcement notes that web-pretrained models are likely contaminated; an AAAI study found chronologically older datasets systematically easier and used training-data inspection, extraction, and membership inference as corroborating evidence. Prefer post-cutoff/withheld cases, newly generated perturbations, hidden test data, and disclose data dates and contamination audit. [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) [Li & Flanigan 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29808)

- **A recent cautionary example:** an OpenAI audit reports that SWE-bench Verified had design and contamination problems and that an audit of SWE-Bench Pro estimated ~30% broken tasks. The lesson is to publish task-validity audits and graders, not merely a leaderboard number. [OpenAI audit](https://openai.com/index/separating-signal-from-noise-coding-evaluations/)

- **LLM-as-judge is useful only after local validation.** MT-Bench/Chatbot Arena work found agreement with human preferences but documented position, verbosity, self-enhancement, and reasoning limitations; JUDGE-BENCH found substantial variation by judged property, human expertise level, and whether text was human- versus model-generated. [Zheng et al.](https://arxiv.org/abs/2306.05685) [JUDGE-BENCH](https://arxiv.org/abs/2406.18403)

- **For scientific interpretation, retain an LLM judge only for residual semantic criteria**—for example, whether a conclusion overclaims beyond the displayed result—after validating it on a blinded, double-expert-scored calibration subset. Report expert agreement and judge–expert agreement, randomize response order, run the judge repeatedly on borderline cases, and use deterministic/executable checks for anything that can be checked directly. This recommendation follows the documented judge biases and domain variability. [Zheng et al.](https://arxiv.org/abs/2306.05685) [JUDGE-BENCH](https://arxiv.org/abs/2406.18403)

- **Attribute agent versus tool performance with factorial ablations.** Hold the container, data, tools and budget fixed while varying backbone/scaffold; compare full agent, no specialized tools, direct code execution, no planner/retriever/reflection, and an oracle workflow plan executed by the same agent. Measure each scientific tool separately under a canonical human-authored workflow. Biomni compares a no-tool base LLM, direct-function coding agent, and an internal planning ablation; a single-cell study removes retrieval, planning, reflection and workflow control. [Biomni](https://pmc.ncbi.nlm.nih.gov/articles/PMC12157518/) [Genome Biology study](https://link.springer.com/article/10.1186/s13059-026-03998-z)

- **No universal published protocol was found for long-running scientific-tool attribution.** The defensible synthesis is a fixed/pinned tool environment plus the factorial ablations above, separate tool-quality baselines, and complete traces. Treat claims of “agent scientific superiority” without these controls as under-attributed. [unverified]

- **What resource-paper reviewers can reasonably require is unusually concrete.** Bioinformatics advises evaluation on standard datasets and comparison with previous methods on common datasets; GigaScience requires reviewer-testable code and test data, code/data availability, and open optimization/evaluation data for transparent ML benchmarking. [Bioinformatics author guidelines](https://academic.oup.com/BIOINFORMATICS/pages/author-guidelines) [GigaScience policies](https://academic.oup.com/gigascience/pages/editorial_policies_and_reporting_standards)

- **A submission-grade artefact set should include** frozen containers/lockfiles, data manifests/checksums and access instructions, task prompts and hidden/public split policy, all seeds and model identifiers, tool versions and parameters, raw trajectories/logs, graders, per-task scores, failures, and an archived DOI snapshot. These map directly to GigaScience’s reviewer-testability and archival requirements and to computational-benchmarking guidance on transparent scripts, parameters and versions. [GigaScience policies](https://academic.oup.com/gigascience/pages/editorial_policies_and_reporting_standards) [benchmarking guidelines](https://pmc.ncbi.nlm.nih.gov/articles/PMC6584985/)

## Primary sources

| title | year | venue/type | URL | why it matters |
|---|---:|---|---|---|
| SWE-bench: Can Language Models Resolve Real-World GitHub Issues? | 2024 | ICLR / benchmark paper | https://arxiv.org/abs/2310.06770 | Canonical executable coding-agent resolved-rate benchmark. |
| τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains | 2025 | ICLR / benchmark paper | https://openreview.net/pdf?id=roNSXZpUDN | Defines end-state evaluation and `pass^k`; supplies the clearest repeated-run numeric evidence. |
| OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments | 2024 | NeurIPS / benchmark | https://os-world.github.io/ | Execution-based real-computer task success. |
| ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery | 2025 | ICLR / benchmark paper | https://arxiv.org/abs/2410.05080 | Scientific program generation, execution-result, and cost evaluation. |
| BLADE: Benchmarking Language Model Agents for Data-Driven Science | 2024 | arXiv / benchmark | https://arxiv.org/abs/2408.09667 | Computational matching of data-analysis decisions to reference analyses. |
| BixBench: a Comprehensive Benchmark for LLM-based Agents in Computational Biology | 2025 | arXiv / benchmark | https://arxiv.org/abs/2503.00096 | Computational-biology analysis and interpretation cases. |
| BiomniBench: Process-level Evaluation of LLM Agents for Real-world Biomedical Research | 2026 | bioRxiv preprint | https://www.biorxiv.org/content/10.64898/2026.05.12.724604v2 | Expert task rubrics over trajectories; directly relevant to biomedical-analysis evaluation. |
| Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena | 2023 | NeurIPS / evaluation paper | https://arxiv.org/abs/2306.05685 | Human validation and known LLM-judge biases. |
| LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks | 2024 | arXiv / meta-evaluation | https://arxiv.org/abs/2406.18403 | Shows judge validity varies materially by task and expertise. |
| Task Contamination: Language Models May Not Be Few-Shot Anymore | 2024 | AAAI | https://ojs.aaai.org/index.php/AAAI/article/view/29808 | Empirical contamination detection and temporal evidence. |
| Systematic benchmarking of omics computational tools | 2019 | Nature Communications | https://www.nature.com/articles/s41467-019-09406-4 | Applicable norms for transparent, systematic computational-biology benchmarks. |

## Caveats and open questions

- Direct, peer-reviewed measurements of run-to-run variance for **scientific pipeline agents** remain sparse. τ-bench provides strong evidence of reliability collapse in tool-use, but it is not a bioinformatics workload; do not claim its percentages transfer to 3D genomics.

- BiomniBench, BioAgent Bench, and several biology-agent evaluations are recent preprints or emerging resources. They are useful design precedents, not settled standards.

- No source located establishes a field-wide repeat count. The correct number depends on the effect size worth detecting, task heterogeneity, rollout variance, and budget; a pilot plus task-clustered power analysis is more defensible than copying “three seeds.”

- A reference pipeline is not synonymous with biological truth. For difficult 3D-genomics calls, use multiple tiers: synthetic/spike-in or simulator cases with known truth, established public data with consensus expectations, and blinded expert review of real ambiguous cases.

- LLM judges may be operationally necessary for interpretation, but their validation must be task-local. Agreement on generic chat preferences does not validate an LLM judge for chromatin-contact interpretation.