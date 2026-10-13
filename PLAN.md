# Study plan: developing agents — theory and practice

Scope: **orientation first**, then build · Purpose: **learn to develop agents**, well enough to build a
real one · Stack: **raw Messages-API loop first, Claude Agent SDK second** ·
Learner: ML-literate, years of bioinformatics algorithm development · Written: 2026-09-30

**Structure:** this course is self-contained and general — theory chapters plus runnable practice, all
under this directory. It is not *about* any one system. Then it is applied: **FoldClaw**
(`~/Desktop/projects/projects_3dgenomics_agent/`) is **a real project, not a teaching example** — two
papers are attached to it — so the capstone is a deliverable, and it gets its own project plan rather
than a chapter. The course exists to make that project succeed; it is not a prerequisite ritual.

```
agents/
├── 00_overview/   ✅ written      Transformer → LLM → Agent: the map
├── 01_transformer/✅ written      self-attention · the architecture
├── 02_LLM/                        what training added
├── 03_agent/                      the loop, and why tools are just text
├── 04_context/                    the context window is the whole state
├── 05_knowledge/                  giving an agent knowledge it was never trained on
├── 06_multi_agent/                one agent or many
├── 07_evaluation/                 what counts as "it worked"
├── practice/      p1…p7           runnable exercises, one per mechanism
├── 08_capstone/                   applying all of it to FoldClaw
├── research/      briefs · raw · logs
└── resources/     style.css · figures/   the course's OWN assets (chapters link ../resources/figures/)
```

### Where the papers go — no vault here

This course keeps no corpus of its own, matching its siblings (`comparative_genomics`,
`superenhancer_differential`), which carry only `resources/` + `research/`. A course accumulates
chapters; papers belong to a project vault:

- **Agent-methodology papers** (ReAct, Toolformer, the benchmark and long-horizon-eval literature) go to
  FoldClaw's `knowledge_base/papers/05_agents/`, which already holds seven of them, with repos in
  `tools/05_agents/`. Widen that module's scope note from "competitors" to cover method papers too.
- **PDFs** come from the central corpus `~/Desktop/projects/papers/` as hard links, never re-downloaded.
- **Each chapter carries its own primary-source table, and that table is the download list** — a source
  good enough to cite is worth fetching; one that isn't, isn't.
- `resources/` stays for the course's own assets (`style.css`, `figures/` — the written chapters link
  `../resources/figures/`). `research/` keeps the briefs, raw Codex output, and logs.

**Navigation is generated, not hand-written.** `agents.html` is the contents page; every chapter also
carries a folding sidebar (collapsed by default, `c` toggles, state remembered across pages) that lists
all chapters and the current chapter's sections — and opens a closed `<details>` when you jump into one.
Both come from `resources/_build_toc.py` + `resources/toc.js`:

```
python3 resources/_build_toc.py            # re-inject into every page
python3 resources/_build_toc.py --check     # report drift, change nothing
```

Re-run it after writing a chapter, and add the new chapter to the `CHAPTERS` registry at the top of the
script. It is idempotent and rewrites only the block between the `TOC:START`/`TOC:END` markers — never
hand-edit that block.

It also renders a **practice** section in the sidebar from its `PRACTICE` registry, so every
chapter can jump straight into a runnable notebook (Colab) or a lab on GitHub — add a row there
when a new exercise lands. And it makes **every external link open in a new tab** (`target="_blank"` +
`rel="noopener noreferrer"`), leaving internal chapter navigation in place, so a reader never loses the
chapter by following a reference. Write plain `<a href="https://…">` in a chapter; the build adds the
attributes. ⚠️ It rewrites the `<a>` tag, so a later script that string-matches on `<a href="https…"`
will miss — match the href alone.

---

## Phase A — theory (6 chapters, HTML in the existing style)

**Sizing, so this cannot expand indefinitely:** one session per chapter at orientation depth (M3 and M4
are the two worth two apiece). Read the chapter, take its acceptance test, build its paired exercise, move on. If a
chapter runs to a third session it has stopped being orientation — note the open question and continue.

Order is strict: each unlocks the next. Every chapter pairs with a practice exercise — read, then build.

### M1 · `02_LLM` — what training actually added
- **Question:** pretraining produces a next-token text simulator. What do SFT, preference optimization
  (RLHF / DPO) and RL-with-verifiable-rewards each *add* — and which agent behaviours (reliable
  tool-calling, multi-step instruction following, staying on task over dozens of steps, declining to
  fabricate) are attributable to which stage?
- **Why here:** every agent misbehaviour is either a capability gap (pretraining) or a behaviour gap
  (post-training), and the fix differs — retrain, re-prompt, or re-scaffold. Plus the fact chapter 03
  rests on: **tool-calling is a trained behaviour, not an architectural feature.**
- **Anchor:** read DPO as a loss function — a closed-form reparameterization of the RLHF objective —
  not as "RL magic".
- **Understood when:** given a misbehaviour, you name the responsible stage and say whether prompting
  can repair it.
- **Research:** returned + cross-checked · **Status:** ✅ **written** — `02_LLM/02_LLM.html`

### M2 · `03_agent` — the loop, and why tools are just text
- **Question:** mechanically, what crosses the wire when a model "calls a tool"? Where does the loop
  live, who owns the state, what is the stop condition — and what makes this different from a shell script?
- **Why here:** you cannot debug an agent you think is magic. This is the chapter that makes everything
  after it concrete.
- **Anchor:** it is the `END`-token argument from chapter 00 again. The model only *emits a request*; an
  ordinary `while` loop in your code does the calling and appends the result. The model is a
  `choose_next_step()` with a very good prior; the agent is your driver script.
- **Also settles:** the four build shapes above, on the harness × deployment grid; and **who is allowed
  to do what** — permissions, approval gates, and why an agent acting on a shared multi-user box is a
  different risk class from one editing a local file. The capstone runs on boxes other people depend on,
  so this is not an appendix.
- **No web search.** Claude-API territory — written from the `claude-api` skill and official docs,
  because search returns stale wire formats and dead model IDs.
- **Understood when:** you can whiteboard messages array → `tool_use` block → `tool_result` block →
  stop reason, and say what breaks if any one part is dropped.
- **Pairs with:** p1, p2 · **Status:** ✅ **written** — `03_agent/03_agent.html`

### M3 · `04_context` — the context window is the whole state
- **Question:** what are the real techniques for managing a long run — compaction, external memory,
  retrieval, tool-result shaping — and what are the documented failure modes (context rot, drift,
  tool-output flooding, lost-in-the-middle, error cascades)?
- **Why here:** this is what decides whether an agent survives 30 steps rather than 5. It is also the
  chapter with the highest ratio of practice to prose — the effects are measurable.
- **Anchor:** it is feature engineering, moved to run time. You already know the representation decides
  whether the algorithm can work; same problem, new substrate.
- **Understood when:** for any agent you write, you can state what enters the context at step *n* and
  what is deliberately excluded, with a reason per item.
- **Research:** returned + cross-checked · **Status:** ✅ **written** — `04_context/04_context.html`

### M4 · `05_knowledge` — building a knowledge base for an agent
- **Question:** the weights are frozen and the corpus is fixed at training time. How do you give an agent
  knowledge it never trained on — and, given a real corpus of a few hundred scientific PDFs, what actually
  works: embedding-based RAG, plain search over a structured index, an agent that navigates files itself,
  a knowledge graph, or some combination?
- **Why here:** this is the *other* of the two routes information can take into a prediction — chapter 00
  named them (weights and context) and this chapter is the second one, engineered. It follows `04_context`
  because retrieval is how you *choose* what occupies the state you just learned to manage. And it is the
  chapter the capstone leans on hardest: FoldClaw's component 1 is a 201-paper corpus, which is worthless
  unless an agent can find the right page in it.
- **Must settle, because the field disagrees loudly:** whether embeddings are needed at all. An agent with
  `grep` over a well-structured index is a real baseline that frequently beats a vector store, and "we
  built RAG" is the most common unexamined default in the field. The chapter's job is to say *when* each
  wins, on evidence.
- **Anchor:** it is indexing and search, which you have built before — plus one genuinely new failure mode:
  the retriever and the reader are the same probabilistic system, so a confident answer can be grounded in
  the wrong chunk. Provenance is not decoration; it is the only way to catch that.
- **Covers:** chunking and what it destroys (tables, figure captions, methods spanning pages), embeddings
  vs BM25 vs hybrid, reranking, contextual retrieval, long-context-instead-of-retrieval, memory vs
  retrieval, citation and staleness, and how to *evaluate* retrieval separately from the agent.
- **Understood when:** for a given corpus you can say which retrieval design you would build and what
  measurement would prove it better than `grep` over a good index.
- **Research:** returned; the central DCI claim **verified against the paper** · **Status:** ✅ **written** — `05_knowledge/05_knowledge.html`
- **Pairs with:** p4

### M5 · `06_multi_agent` — one agent or many
- **Question:** when does splitting work across a planner + workers + a critic actually help, when does
  it just multiply cost and coordination failure, and what does the evidence say rather than the marketing?
- **Why here:** it is the most over-recommended pattern in the field and the easiest way to make an agent
  worse. Deciding it on evidence is the point.
- **Anchor:** the honest version of the question is amortization — does delegation buy enough isolated
  context to pay for the lost shared state? Same trade as any divide-and-conquer.
- **Understood when:** you can state the condition under which you would *not* use subagents, and the
  one measurement that would settle it for a given task.
- **Research:** partially covered by the context brief (delegation section); may need its own pass.
- **Pairs with:** p4 · **Status:** planned

### M6 · `07_evaluation` — what counts as "it worked"
- **Question:** how is an agent's capability measured credibly — benchmark design, pass@k vs pass^k,
  how many repeats given run-to-run variance, cost per completed task, contamination — and how do you
  attribute credit between the model, the scaffold, and the tool it called?
- **Why here:** last theory chapter, because the eval is what you build *against*. An agent without one
  is a demo.
- **Anchor:** treat it exactly as a benchmarking paper — fixed baselines, fixed seeds, paired cases,
  reported variance, and a named result that would falsify the approach. You already do this for assemblers.
- **Understood when:** you can write an eval protocol for an agent you just built, including the result
  that would make you abandon it.
- **Research:** returned + cross-checked · **Status:** ✅ **written** — `07_evaluation/07_evaluation.html`

---

## Phase B — practice (runnable, here, one exercise per mechanism)

Small and self-contained on purpose: each isolates one thing, runs on the Mac, needs no server, and is
finished in a sitting. `practice/p<N>_<name>/`. Python, Anthropic SDK, `claude-opus-5`.

**Because FoldClaw is real, the exercises use its corpus wherever it costs nothing.** p1–p2 and p4 build
their tool surface over FoldClaw's `knowledge_base/` — 201 PDFs, 50 cloned repos and an `INDEX.md`
already on disk, no server and no pipeline needed. So the first working tool (`kb_query`) is a component
the capstone keeps, and the throwaway-toy stage is skipped entirely.

| | Exercise | Teaches | Done when |
|---|---|---|---|
| **p1** | the 80-line agent | the loop is yours, not the model's | it calls one real tool instead of guessing, and you can point at the line that halts it |
| **p2** | tool surface design | schemas, validation, errors as data, parallel calls; then the same agent on the Tool Runner | a malformed call returns a useful `is_error` result and the agent recovers instead of looping |
| **p3** | context engineering | compaction, tool-result shaping, what a huge observation does to a run | you *measure* a long run failing, fix it by shaping observations, and measure the difference |
| **p4** | a knowledge base the agent can use | chunking, indexing, retrieval, provenance | retrieval is measured against a `grep`-over-index baseline on real questions, and you know which won |
| **p5** | subagents | delegation, isolated context, coordination cost | you have a task where delegation wins and one where it loses — with numbers |
| **p6** | the Claude Agent SDK | hooks, permissions, built-in tools, sessions | a hook provably blocks an action you decided is forbidden |
| **p7** | an eval harness | repeats, variance, baselines, cost per task | one table comparing two agent variants on the same cases, with variance reported |

The order matters: p3, p4 and p7 are the three that change how you build everything afterwards, and all
depend on having something that already works (p1–p2). **p4 runs against a real corpus** — this course's
vault or FoldClaw's 201 papers — so its output is a working component, not an exercise.

---

## Phase C — `08_capstone`: FoldClaw, a real project

Applied **after** the course, using everything above — and treated as real work with a paper attached,
which means it needs its own project plan (repo scaffold, which assay first, which simulator, which case
studies) beyond anything a chapter can carry. `~/Desktop/projects/projects_3dgenomics_agent/`
is a 3D-genomics agent with strategy locked 2026-06-10 — Hi-C/Micro-C/ChIA-PET/HiChIP pipelines plus
phase-separation simulation, whose intended differentiator is the **data ↔ simulation loop**.

Its state today, which is what makes it a real example rather than an exercise:

| Component | Status |
|---|---|
| `knowledge_base` | **built** — 201 paper PDFs, 33 DOI-only stubs, 50 cloned tool repos, 7 modules, `INDEX.md` |
| Paper A manuscript scaffold | **built** (sections stubbed) |
| pipelines · simulation · **the agent itself** | **not built** — there is no `FoldClaw/` code repo yet |
| Paper B flagship idea | `manuscript/ideas/flagship.md` still reads `TODO: capture verbatim` (since 2026-06-10) |

### Verified competitor finding (2026-09-30) — it narrows the claim

Two briefs already ran against the capstone (`research/raw/m4_bioagent_landscape.md`,
`research/raw/m5_data_simulation_spine.md`). One result is decision-grade and **verified against primary
sources**, not taken on the researcher's word:

- ✅ **BioMaster (Patterns 2026, `10.1016/j.patter.2026.101611`, code released by `ai4nucleome`) is a
  multi-agent bioinformatics workflow system — Plan / Execute / Debug / Check — and Hi-C is among the
  assays it handles.** Confirmed at `github.com/ai4nucleome/BioMaster`. Note the lab: `ai4nucleome`
  works in exactly this field. ⚠️ The brief's stronger specifics (five GM12878 trials, HiCRep SCC > 0.99,
  identical Mustache loop counts) are paper-internal and **not independently verified** — read the paper
  before relying on them.
- ✅ **HiCFoundation is a foundation model, not an agent** — self-supervised on Hi-C, fine-tuned for loop
  detection, resolution enhancement, reproducibility, epigenomic prediction, scHi-C, 316 species
  (`github.com/Noble-Lab/HiCFoundation`; Wang … Noble, Wang). The repo still cites the 2024 bioRxiv
  preprint; the KB's `2026_NatMethods_wang-hi-c-foundation-model` stub is the journal version. It plans
  nothing and executes nothing, so it is a comparator, not a competitor.
- ⚠️ **No agent found that closes the contact-data ↔ physical-simulation loop.** An absence claim from web
  search, so treat it as "not found", not "does not exist" — but it is the gap FoldClaw's strategy note
  already bet on.

**Consequence for Paper A: FoldClaw cannot claim to be the first 3D-genomics agent — BioMaster got
there.** The defensible claim is narrower and better, and it is the one `strategy.md` already chose: the
first agent to close the loop between contact data and a physical model of the folding. Which means the
pipelines and the KB are explicitly *not* the contribution, and the capstone should spend its effort on
the spine. Re-run this search at submission time.

Two findings also pre-load `07_evaluation`, and they are warnings: an artifact-and-perturbation benchmark
reports that superficially completed bioinformatics pipelines break under controlled perturbation (so
"it ran" ≠ "it reasoned"), and an external audit of a widely used bio-agent benchmark argues flawed tasks
and LLM-judge errors explain most apparent failure. Both unverified by me; both exactly the traps that
chapter exists to avoid.

One constraint the course should surface before the capstone, since it eliminates an option most guides
recommend: **the hosted Managed-Agents route cannot work for FoldClaw** — its sandbox cannot reach
`512gb_server1` or `server_sustech`, where the data and GPUs live behind ssh. The agent must run where
the data is, which leaves options 1, 2 and 4.

---

## How the course gets finished — fill it while studying it

**Set 2026-10-07.** The chapters are written; `practice/` and `08_capstone/` are not, and the written
chapters are thinner than the two that preceded them (see the audit below). These gaps are **not**
filled in bulk up front. They are filled **during study, at the point where the gap is actually felt**
— because a reader hitting a thin passage locates the real gap in one sentence, where guessing at it
from the outside adds bulk in the wrong place.

So the loop per chapter is:

1. **Read it.** Stop at the first place the text asserts something it does not earn.
2. **Say so.** "This table claims X and I don't believe it" / "I can't picture this" / "this needs a
   number". That sentence is the spec.
3. **Fill it together, then** — a worked example, a diagram, a code block, a measurement.
4. **Take the acceptance test**, then build the paired exercise. The exercise is written when you
   reach it, not before.
5. **Re-run `python3 resources/_build_toc.py`** if headings changed.

What is known to be missing, so it is not rediscovered each time:

| Gap | Where |
|---|---|
| All seven exercises | `practice/` is empty — `p1` is the keystone, write it first |
| The capstone | `08_capstone/` is empty; it needs FoldClaw's project plan, not a chapter |
| Depth: ~1 400 words vs ~4 000–5 400 in ch 00–01 | `02_LLM` most of all — six topics, one code block, no diagram |
| Code and diagrams | `05_knowledge` and `06` have **zero** of both; `07` has no diagram |
| Stale navigation | `00_overview`'s `.docnav` still shows the old four-chapter arc |

`06_multi_agent` is short **on purpose** — the evidence is thin and the page says so. Leave it.

## Working rules

- **`rc=0` is not understanding.** Every chapter ends with its acceptance test actually taken, and every
  exercise with something that runs.
- **Codex searches, Claude verifies.** `[unverified]` flags stay visible in the notes until checked.
- Nothing about the Claude API goes to Codex (stale model IDs, wrong wire formats) — that comes from the
  `claude-api` skill.
- Practice code stays here in `practice/`. Capstone code lands in the FoldClaw project, under its own
  layout and its own server-first rules.
