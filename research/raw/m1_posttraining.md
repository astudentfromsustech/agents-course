## Answer

Pretraining supplies the broad conditional distribution: linguistic competence, world/code priors, and latent procedures, but no reliable convention for whose instruction to follow, when to emit structured actions, or when to stop and admit uncertainty. SFT makes those conventions high-probability by imitating demonstrations: instruction format, role hierarchy, tool-call syntax, and worked trajectories. Preference optimization then selects among plausible SFT-like behaviours—helpful versus evasive, concise versus rambling, honest qualification versus confident fabrication, and safe refusal versus unsafe compliance—using pairwise human/AI judgments or a learned reward. It chiefly improves *elicitation and behavioural policy*, not usually the underlying knowledge substrate. [OpenAI](https://openai.com/index/instruction-following/)

RLVR is distinct because reward is mechanically checkable (answer checker, unit tests, compiler, simulator, constraint satisfaction). Online rollouts can therefore optimize long chains whose final success cannot be cheaply demonstrated token-by-token. It reliably improves domains with robust verifiers—math, code, and executable tasks—and is now central to public reasoning-model reports. Whether that constitutes “new capability” remains unsettled: DeepSeek R1-Zero shows RL producing reflection/self-verification without initial SFT, while OpenAI’s earlier account and recent RLVR work support the alternative interpretation that training mainly amplifies low-probability latent successful traces. [DeepSeek-R1](https://arxiv.org/abs/2501.12948) [OpenAI](https://openai.com/index/instruction-following/)

For agents, reliable single-call tooling is largely an SFT/schema-and-example artifact; choosing tools, recovering from results, and completing multi-step tasks increasingly receive environment/trajectory RL. “Stays on task for dozens of steps” is not attributable to one stage: it requires agentic RL where available, but also context engineering, tool design, termination rules, and inference-time budget. “Declines to fabricate” is primarily preference/safety training plus retrieval/tool grounding—not an assurance of factuality.

## Key findings

- The useful 2026 decomposition is: pretraining → optional continued/domain “mid-training” → SFT/cold start → preference/safety optimization → online RLVR or environment RL → distillation. It is a conceptual decomposition, not a universal fixed pipeline: DeepSeek publicly used cold-start SFT, RL, rejection-sampled SFT, a further RL stage, then distillation; R1-Zero omitted initial SFT. [DeepSeek-R1](https://arxiv.org/abs/2501.12948)

- Public frontier evidence confirms that RL is no longer confined to chat preference alignment: OpenAI states that o-series models receive large-scale RL on chains of thought and that o3/o4-mini were trained through RL both to use tools and to reason about when to use them. [OpenAI o3/o4-mini announcement](https://openai.com/index/introducing-o3-and-o4-mini/) [system card](https://openai.com/index/o3-o4-mini-system-card/)

- SFT adds a behavioural interface, not merely polish: InstructGPT used human demonstrations before comparison/reward-model training; its model card reports gains in obeying explicit constraints, attempting the requested task, and not making up information in closed-domain tasks. [InstructGPT pipeline](https://openai.com/index/instruction-following/) [model card](https://github.com/openai/following-instructions-human-feedback/blob/main/model-card.md)

- Classical RLHF fits a reward model from pairwise preferences and optimizes the policy under a KL-to-reference constraint, historically with PPO. DPO derives a direct pairwise classification objective for the same KL-regularized preference formulation, eliminating a separate reward model and online policy rollouts. [InstructGPT](https://openai.com/index/instruction-following/) [DPO](https://arxiv.org/abs/2305.18290)

- GRPO is an *online RL optimizer*, not a human-feedback method: it replaces a learned critic/value baseline with relative rewards among sampled responses for the same prompt. DeepSeek reports GRPO for R1/R1-Zero; public OpenAI reports say “large-scale RL” but do not disclose the optimizer. [DeepSeek-R1](https://arxiv.org/abs/2501.12948) [OpenAI o3/o4-mini system card](https://openai.com/index/o3-o4-mini-system-card/)

- Therefore there is no credible public “production winner” between RLHF/PPO, DPO, and GRPO. DPO’s paper establishes simpler and competitive controlled results, DeepSeek establishes production use of GRPO, and frontier labs disclose insufficient implementation detail to infer a universal industry choice. [DPO](https://arxiv.org/abs/2305.18290) [DeepSeek-R1](https://arxiv.org/abs/2501.12948)

- RLVR rewards outcomes that an external verifier can score rather than subjective preference. R1-Zero’s pure-RL result is evidence that such training can elicit self-verification, reflection, and long CoT without preliminary SFT; it is not proof that entirely novel primitives were learned. [DeepSeek-R1](https://arxiv.org/abs/2501.12948)

- The contested interpretation is real: OpenAI characterized RLHF as predominantly unlocking pretraining capabilities because post-training was tiny relative to pretraining, whereas a 2025 RLVR study argues verifier-based RL improves correct-reasoning generalization under a stricter reasoning-path metric. Both are compatible with probability sharpening producing dramatic multi-step gains. [OpenAI](https://openai.com/index/instruction-following/) [Wen et al.](https://arxiv.org/abs/2506.14245)

- Tool use is trainable rather than an automatic consequence of pretraining: Toolformer trained a model to decide whether/when/how to call APIs and incorporate results; Gorilla found base frontier models prone to wrong arguments and hallucinated API usage, and improved API calling through finetuning plus retrieval. [Toolformer](https://arxiv.org/abs/2302.04761) [Gorilla](https://arxiv.org/abs/2305.15334)

- Tool reliability also depends on the serving contract and examples. Anthropic’s documentation says schemas alone cannot express usage conventions, and its 2025 guidance recommends tool-use examples; consequently, a model’s apparent tool skill cannot be cleanly assigned to weights alone. [Anthropic](https://www.anthropic.com/engineering/advanced-tool-use) [tool-use contract](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works)

- Agentic post-training is now a distinct, publicly documented direction: GPT-5-Codex was trained with RL on real coding tasks in varied environments to follow instructions and iteratively run tests; contemporaneous research trains multi-turn software agents with RL rather than only single-turn code/math. [GPT-5-Codex system card](https://cdn.openai.com/pdf/97cc5669-7a25-4e63-b15f-5fd5bdc4d149/gpt-5-codex-system-card.pdf) [Mahalunkar et al.](https://arxiv.org/abs/2508.03501)

- A key change within the past 12 months is the move from generic reasoning RL toward environment-native agent training: Qwen3-Coder-Next reports verifiable coding environments used in both mid-training and RL; OpenAI’s Codex report describes real-world coding-environment RL. [Qwen3-Coder-Next](https://arxiv.org/abs/2603.00729) [GPT-5-Codex](https://cdn.openai.com/pdf/97cc5669-7a25-4e63-b15f-5fd5bdc4d149/gpt-5-codex-system-card.pdf)

- Refusal/non-fabrication should be separated. Preference/safety training can teach calibrated refusal and safe alternatives, but it does not guarantee truth; OpenAI’s GPT-5 safe-completion training explicitly rewards helpfulness subject to output-safety constraints and reports better safety/helpfulness than binary refusal training. [Safe Completions](https://arxiv.org/abs/2508.09224)

- Chat alignment need not transfer to agents: Anthropic reports that Claude 4’s predominantly chat-based RLHF data lacked agentic tool-use data and did not adequately suppress harmful behaviour in an agentic evaluation—direct evidence that tool/trajectory-specific training distribution matters. [Anthropic](https://alignment.anthropic.com/2026/teaching-claude-why/)

## Primary sources

| title | year | venue/type | URL | why it matters |
|---|---:|---|---|---|
| Training Language Models to Follow Instructions with Human Feedback — Ouyang et al. | 2022 | NeurIPS / OpenAI paper | [URL](https://arxiv.org/abs/2203.02155) | Canonical SFT → reward model → PPO-RLHF pipeline and the “elicitation versus new capability” framing. |
| Direct Preference Optimization: Your Language Model is Secretly a Reward Model — Rafailov et al. | 2023 | NeurIPS | [URL](https://arxiv.org/abs/2305.18290) | Defines DPO and its relation to KL-regularized RLHF. |
| Constitutional AI: Harmlessness from AI Feedback — Bai et al. | 2022 | arXiv | [URL](https://arxiv.org/abs/2212.08073) | SFT self-revision plus RLAIF/preference-RL for non-evasive harmlessness. |
| Toolformer: Language Models Can Teach Themselves to Use Tools — Schick et al. | 2023 | NeurIPS | [URL](https://arxiv.org/abs/2302.04761) | Direct evidence for tool-call learning: call selection, arguments, and result incorporation. |
| Gorilla: Large Language Model Connected with Massive APIs — Patil et al. | 2023/2024 | NeurIPS | [URL](https://arxiv.org/abs/2305.15334) | Shows limitations of prompted base models and benefits of tool/API-specific finetuning plus retrieval. |
| DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models — Shao et al. | 2024 | arXiv | [URL](https://arxiv.org/abs/2402.03300) | Introduces GRPO as a critic-free grouped-relative PPO variant. |
| DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning — DeepSeek-AI | 2025 | technical report / Nature version | [URL](https://arxiv.org/abs/2501.12948) | Public multi-stage RLVR reasoning pipeline, including R1-Zero and distillation. |
| OpenAI o3 and o4-mini System Card | 2025 | official technical report | [URL](https://openai.com/index/o3-o4-mini-system-card/) | Public confirmation of large-scale CoT RL and tool use in reasoning models. |
| Addendum to GPT-5 System Card: GPT-5-Codex | 2025 | official system card | [URL](https://cdn.openai.com/pdf/97cc5669-7a25-4e63-b15f-5fd5bdc4d149/gpt-5-codex-system-card.pdf) | Clear public evidence for real-environment RL for iterative coding agents. |
| From Hard Refusals to Safe-Completions — Yuan et al. | 2025 | OpenAI paper | [URL](https://arxiv.org/abs/2508.09224) | Recent safety-preference objective and controlled/production comparisons. |
| Training Long-Context, Multi-Turn Software Engineering Agents with Reinforcement Learning — Mahalunkar et al. | 2025 | arXiv | [URL](https://arxiv.org/abs/2508.03501) | Evidence that multi-turn, long-context agent RL is becoming its own research/training regime. |
| Qwen3-Coder-Next Technical Report | 2026 | technical report | [URL](https://arxiv.org/abs/2603.00729) | Recent public evidence of executable-environment, agentic mid-training plus RL. |

## Caveats and open questions

- Frontier labs disclose objectives and outcomes much more often than data mixtures, reward models, optimizers, ablations, or stage boundaries. Claims that every frontier model runs all six listed stages are therefore **[unverified]**.

- “Mid-training” is not standardized terminology: some groups mean continued pretraining on domain/long-context/code data; others include synthetic trajectory training. Its causal contribution is rarely isolated publicly.

- Production prevalence of DPO versus PPO-style RLHF versus GRPO cannot be inferred from papers, open-source replications, or benchmark trends. Public evidence establishes *examples*, not market share.

- RLVR’s capability claim is underdetermined by pass@1 alone. It may teach compositional search/credit assignment, or increase the probability of already latent correct atomic steps until long trajectories become viable; existing papers defend both interpretations. [Fan et al.](https://arxiv.org/abs/2601.15609) [Wang et al.](https://arxiv.org/abs/2602.08281)

- “Declining to fabricate” conflates at least four behaviours: unsafe-request refusal, epistemic abstention, citation/attribution accuracy, and truthful use of tool outputs. Published safety-preference results do not establish reliable factual abstention across open-world questions. [OpenAI Help](https://help.openai.com/en/articles/8313428-does-chatgpt-tell-the-truth%3F.pls)

- Long-horizon agent success remains confounded by the model, tool schemas, retrieval, environment stability, scaffold, retry/termination policy, context window, and inference-time reasoning budget; no public study cleanly attributes “dozens of steps” to post-training alone.