# Citations, verified

Every entry below was fetched from arxiv.org on **2026-08-29** and the title, authors
and claim were confirmed against the abstract. The 2026 IDs originally came from
agents with web access and had **not** been independently checked; the fixer that
applied them said so explicitly. They have now been checked.

Do not cite anything from this project that is not on this list without fetching it
first. A fabricated reference would be worse than no reference.

---

### Mishra, Khashabi, Liu. *Steered LLM Activations are Non-Surjective.*
`arXiv:2604.09839` · **verified 2026-08-29**

Frames prompt-vs-steering as a surjectivity problem and proves that under practical
assumptions steering pushes the residual stream beyond the manifold reachable from
discrete prompts. Demonstrated on three LLMs: no prompt typically reproduces the
internal state steering creates.

**What it settles for us.** "Compile a steering vector into tokens" is ruled out at the
activation level, as a theorem rather than an empirical gap. Our `compile_check`
(`cos(Δ,d) = 0.038`, `Δ∥` = 3.1% of α) is an instance of it, not a discovery. It is
also why the project's goal has to be stated behaviourally: match the *effect*, not
the activations.

### Wu, Arora, Geiger, Wang, Huang, Jurafsky, Manning, Potts. *AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders.*
`arXiv:2501.17148` · **verified 2026-08-29**

Benchmarks steering and concept detection on Gemma-2-2B and 9B across prompting,
finetuning, SAEs, linear probes and others. Finding: "prompting outperforms all
existing methods, followed by finetuning"; SAEs are not competitive.

**What it settles for us.** "A prompt beats a steering vector" is already established,
so our ~2x is a replication rather than a novelty claim. More usefully, AxBench tunes
the steering factor per concept to its own optimum and prompting still wins, which
closes the obvious objection to our result: that we simply picked a bad α.

### Panickssery, Gabrieli, Schulz, Tong, Hubinger, Turner. *Steering Llama 2 via Contrastive Activation Addition.*
`arXiv:2312.06681` · **verified 2026-08-29**

CAA. Steering vectors from contrastive pairs, added "at all token positions after the
user's prompt" during inference with a positive or negative coefficient.

**What it settles for us.** It is the reference point our intervention is *not*. Ours
is prefill-only, so the edit lands on prompt positions and does not compound across
decoded tokens. That difference is why we can run α = 1.0·‖R‖ where per-token practice
reports far smaller maxima, and it has to be stated or the coefficient looks reckless.

### Luo, Liang, Xuan. *SteerCheck: Attribution Specificity and Alignment Leakage in Activation-Steering Audits.*
`arXiv:2608.24335` · **verified 2026-08-29**

A preregistered audit framework for activation steering. Isotropic random directions
cluster in a narrow region; sign-randomised controls "often retain substantial target
alignment" (ρ = .94 with signed cosine, 25.3% of draws exceeding cosine .5). Also
reports their automatic judge failing calibration at macro-F1 .562.

**What it settles for us.** Our criticism of isotropic random-direction nulls is not
novel; this says it two days before we wrote it down, and supplies better controls
(PCA-subspace, sign-randomised under a matched KL budget). Their judge-calibration
failure is independent support for our own finding that LLM-judge reliability is
conditional.

### Mody, Agarwal, Mittal, Mahato. *Minimizing Targeted Activations: Input-Only Suppression of Evaluation-Awareness Latents in Large Language Models.*
`arXiv:2607.25907` · **verified 2026-08-29**

GCG-style token optimisation plus a fluency regulariser, driving a target latent toward
zero across five latent constructions. Critically: **"a placebo random direction is
suppressed just as hard and shifts behavior just as far."**

**What it settles for us.** This is the most dangerous paper for our result and belongs
in Limitations, not buried. We have no random-direction control for the *token-search*
arm: `control_junk` and `control_text` are hand-written, not optimised. Until a string
is searched against a random direction to a matched board score and judged identically,
"the behaviour comes from optimising against `d`" is not established against the rival
"optimising a token string against any direction does this."

### Ibrahim, Hafner, Rocher. *Training language models to be warm and empathetic makes them less reliable and more sycophantic.*
`arXiv:2507.21919` · **verified 2026-08-29**

Five models, controlled experiments: warm variants showed "substantially higher error
rates (+10 to +30 percentage points)", promoted conspiracy theories, and validated
incorrect user beliefs, especially when users expressed sadness.

**What it settles for us.** A warmth/reliability trade-off is documented, so our honesty
re-judge finding no cost (`pro_top` −0.108, p=0.55) is the *expected* null on a corpus
where 45 of 50 prompts put nothing honesty-related at stake, not evidence that no cost
exists. Prevents over-claiming the null in either direction.

---

## Not yet verified

- **Hewitt and Liang (2019), control tasks.** Older precedent for the random-control
  critique. Real and well known, but the exact claim attributed to it here has not been
  re-read against the paper.
- `arXiv:2606.06735` (Aparin and Gaintseva, angular/radial decomposition). Mentioned
  once by an agent as supplying field vocabulary. Not fetched. Do not cite as-is.
- The "Anthropic used 0.1x normalised residual norm" and "GLM-5 needed 0.025x" figures
  are **second-hand via a LessWrong reproduction** and were flagged unverified at the
  time. Either chase the primary source or drop the specific numbers and keep the
  qualitative point.
