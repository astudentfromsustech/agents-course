# Developing agents — a course

Seven chapters and seven exercises on building LLM agents, written around one sentence:

> An agent is a pipeline driver whose next step is chosen by a model at run time
> instead of by a DAG you wrote in advance.

Every hard problem in the field is a consequence of that trade, and every chapter is one of
those consequences.

**Read it:** https://astudentfromsustech.github.io/REPO_NAME/

## Chapters

| | | |
|---|---|---|
| 00 | [Overview](00_overview/00_overview.html) | Transformer → LLM → Agent: the map |
| 01 | [Transformer](01_transformer/01_self_attention.html) · [architecture](01_transformer/02_transformer.html) | Self-attention, then the stack it lives in |
| 02 | [LLM](02_LLM/02_LLM.html) | What training added — capability vs behaviour |
| 03 | [Agent](03_agent/03_agent.html) | The loop, and why tools are just text |
| 04 | [Context](04_context/04_context.html) | The context window is the whole state |
| 05 | [Knowledge](05_knowledge/05_knowledge.html) | Giving an agent what it never trained on |
| 06 | [Multi-agent](06_multi_agent/06_multi_agent.html) | One agent or many |
| 07 | [Evaluation](07_evaluation/07_evaluation.html) | What counts as "it worked" |

Each chapter ends with an acceptance test and a paired exercise. `practice/` and the capstone
are deliberately unwritten — they get filled in while the course is studied, at the point where
a gap is actually felt. See `PLAN.md` for the syllabus and the known gaps.

## Building

Pages are self-contained HTML sharing one stylesheet. After editing a chapter:

```bash
python3 resources/_build_toc.py          # rebuild the sidebar + external-link handling
python3 resources/_build_toc.py --check  # report drift, change nothing
```

The script owns the contents sidebar (between the `TOC:START`/`TOC:END` markers — never hand-edit
that block) and makes every external link open in a new tab. Add new chapters to its `CHAPTERS`
registry.

## Sources and credit

These are study notes, and they lean on other people's teaching throughout:

- **Figures are slides from [Hung-yi Lee's Machine Learning course](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.php)** (National Taiwan University), reproduced here for educational commentary and credited individually in each caption. The course is excellent and freely taught — watch it rather than only reading about it. If the rights holder would prefer these images not be mirrored, open an issue and they will be removed.
- The full transformer architecture diagram is by dvgodoy via Wikimedia Commons, CC BY 4.0.
- Claims about the Claude API follow the official documentation; everything else carries its primary source inline, and claims that could not be verified are marked as such in the text.

Written with [Claude Code](https://claude.com/claude-code).
