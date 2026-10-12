You are the research leg of a study plan. Do WEB RESEARCH and report findings.

TOPIC: Closing the loop between contact data and chromatin polymer / phase-separation simulation
QUESTION TO ANSWER: Concretely, what is the state of the art in going from measured contact data
(Hi-C / Micro-C) to a PHYSICAL model (loop extrusion polymer models, LLPS/condensate models) and back
to a predicted contact map — i.e. FITTING or INFERRING model parameters from data rather than
hand-tuning them — and what metric establishes that a simulation reproduced the data?
CONTEXT: module 5 of a plan to build a 3D-genomics agent whose differentiator is automating exactly
this loop. Reader is an expert computational biologist, comfortable with inverse problems and
likelihood-based inference, but NOT a polymer physicist.
DEPTH: orientation, technically precise.

FIND AND REPORT:
1. The main model families and their canonical implementations: loop-extrusion / polychrom-style
   simulations, MiChroM / OpenMiChroM (maximum-entropy), strings-and-binders, HiP-HoP, Chrom3D,
   and restraint-based 3D inference (PASTIS, miniMDS, 3DMax). For each: what is fitted vs assumed,
   and what data it consumes.
2. PARAMETER INFERENCE specifically: which published methods actually FIT parameters to a contact map
   (maximum entropy, Bayesian/ABC, simulation-based inference, differentiable simulators, ML
   surrogates)? This is the crux — be thorough and cite code where it exists.
3. Condensate / LLPS simulation coupled to genome organisation: CALVADOS, Mpipi, OpenABC and any
   work that couples condensate models to Hi-C-scale chromatin models. Is that coupling published,
   or still open?
4. GOODNESS-OF-FIT: what metrics are used to say a simulated contact map matches an observed one
   (stratum-adjusted correlation / HiCRep, P(s) curves, insulation, compartment eigenvector,
   loop recovery)? Which are considered rigorous and which are criticised?
5. THE CONTESTED PART: the state of the argument over whether compartmentalisation is driven by
   phase separation vs loop extrusion vs both, and whether LLPS claims in chromatin are
   over-applied. Report the disagreement with sources on both sides — do not flatten it.
6. Computational cost: rough GPU/CPU cost of running these simulations at a realistic genome scale,
   since the agent must schedule them.

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
