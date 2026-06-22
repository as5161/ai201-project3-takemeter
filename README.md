# TakeMeter — Classifying Discourse Quality on r/wallstreetbets

A fine-tuned DistilBERT classifier that sorts r/wallstreetbets posts into three
discourse-quality categories — `dd`, `speculation`, and `reaction` — benchmarked
against a zero-shot Groq (llama-3.3-70b) baseline on the same test set.

**TL;DR result:** on a 54-post test set, the zero-shot 70B baseline (macro-F1 **0.77**)
outperformed the fine-tuned DistilBERT (macro-F1 **0.65**). The interesting finding is
*where* the fine-tuned model fails: it cannot hold the `speculation` class, collapsing it
into `reaction`. That failure maps directly onto the hardest label boundary I identified
before training.

---

## Community

**r/wallstreetbets** — a high-volume retail-investing subreddit whose discourse ranges
from genuine due-diligence write-ups to pure emotional reactions to market moves. It's a
good fit for a discourse-quality classifier because the quality of "takes" varies
enormously and *visibly*, and the community has native vocabulary for that variation
("DD," "YOLO," "FUD," "bagholder," "hype"). Members actively police the line between real
analysis and hype dressed up as analysis — "great DD bro 🤡" is a recognized form of
mockery — which means the distinctions I'm classifying are ones real participants already
care about. The catch is that the sub skews toward emotional/hype content, so I sampled
deliberately to keep the three classes balanced.

---

## Label Taxonomy

Three mutually exclusive labels. **Operative test:** if removing the opinion framing still
leaves *specific, checkable evidence*, it's `dd`; if the "evidence" is vague, decorative,
or absent, it's `speculation`; if there's no claim being argued at all — only feeling —
it's `reaction`.

**`dd`** — structured reasoning backed by specific, verifiable evidence (financials,
valuation multiples, catalysts, guidance, option mechanics). You could argue with it on
its merits.
- *"Supreme Court is going to decide West Virginia v. EPA tomorrow… I'm pretty confident they're going to limit the EPA's power… a big boon for utility companies that still run coal plants like Duke Energy."*
- *"COST Short Thesis — Costco trades at ~42x forward earnings, priced for perfection. Core merchandise margins contracted three quarters in a row, membership income faces comp headwinds…"*

**`speculation`** — a confident directional call or conviction play asserted *without*
real supporting reasoning, including posts that wear the *costume* of analysis (asserted
mechanisms, buzzwords) but contain no checkable evidence.
- *"…the rate hike was 75 bps not 100 bps… In the next 2 to 3 weeks we're blasting higher… I'm feeling an early santa-rally HO HO HOLD!!!"*
- *"Once the US 10yr hits 4.20% the play is TSLA 400c 3 months out. You heard it here first."*

**`reaction`** — emotional or momentum expression about a move or position (hype, panic,
celebration, loss/gain posts). No claim is being argued.
- *"Look, I don't give a shit about your holidays. I want to watch stocks go up and down… Stop taking this away from me."*
- *"Lmfao so many puts got bent over 🤡🤡🤡"*

---

## Dataset

**Source.** I originally planned to scrape r/wallstreetbets directly via the Reddit API
(PRAW). Reddit's 2025 Responsible Builder Policy now gates new Data API apps behind an
approval process I couldn't satisfy for a class project, so I pivoted to a public,
pre-collected dataset: **`StephanAkkerman/wallstreetbets-ner` on Hugging Face**. I used the
post **text only** and ignored the dataset's named-entity (ticker/company) labels, which
are irrelevant to my task — I applied my own `dd`/`speculation`/`reaction` labels from
scratch.

**Labeling process.** An LLM pre-labeled all 453 candidate rows from my definitions; I then
reviewed and corrected **every row**. During review I dropped ~87 rows that were pure
questions, bare news headlines, or off-topic/meta posts (these don't fit a discourse-quality
taxonomy), and re-labeled ~14 posts from `dd` to `speculation` after tightening my evidence
bar. (LLM annotation assistance is disclosed in the AI Usage section.)

**Label distribution (360 examples, single CSV: `text`, `label`):**

| Label | Count | Share |
|---|---|---|
| reaction | 163 | 45% |
| dd | 102 | 28% |
| speculation | 95 | 26% |

No class exceeds the 70% ceiling, so no rebalancing was needed. The notebook splits this
70/15/15 (stratified) → 252 train / 54 validation / 54 test.

**Three genuinely difficult posts and how I ruled on them:**

1. **Rigorous-looking technical analysis (`dd` vs. `speculation`).**
   *"I've just purchased $VET as it breaks through the daily and weekly zone… CVX and OXY
   yanked back when oil futures pulled in but $VET kept moving. This is the relative
   strength you want to see…"* — It names tickers, makes a comparison, includes risk
   management; it *reads* like analysis. **→ `speculation`**: chart-zone reading and
   "relative strength" are pattern interpretation, not verifiable evidence.

2. **Bare earnings/news drop (`dd` vs. drop).**
   *"Seagate +16% after-hours on Q3 2026 earnings: EPS $5 vs $3.97 est, revenue $3.45B vs
   $3.16B est."* — Just a headline, no explicit argument. **→ `dd`**: the content *is*
   specific, checkable financial evidence (beats vs. estimates). My rule: a news drop with
   hard figures counts as `dd`; a bare headline with no numbers and no take gets dropped.

3. **Satirical pseudo-DD (`speculation` vs. `dd`).**
   *"THE DEFLATION GLITCH: Why $2 Bills are the ultimate hedge… (The 'Due Diligence') only
   1.7 billion $2 bills vs 14.9 billion $1 bills…"* — Full DD costume: headers, a "Due
   Diligence" section, real Fed figures. **→ `speculation`**: function over surface —
   structure plus real numbers ≠ `dd` when the reasoning doesn't support a genuine,
   checkable investment claim.

---

## Fine-Tuning Approach

- **Base model:** `distilbert-base-uncased` (Hugging Face) with a 3-class classification head.
- **Platform:** Google Colab, free T4 GPU.
- **Setup:** 6 epochs, learning rate 2e-5, batch size 16, max sequence length 256,
  weight decay 0.01, `warmup_ratio=0.1`, best checkpoint selected by validation macro-F1,
  class-weighted cross-entropy loss.

**Key training decision #1 — fixing undertraining (warmup vs. total steps).**
The notebook's default `warmup_steps=50` with 3 epochs was *longer than the entire training
run*: 252 train examples ÷ batch 16 ≈ 16 steps/epoch × 3 = ~48 total steps. With a 50-step
linear warmup, the learning rate never finished ramping up, so the model trained at a
fraction of its intended LR and badly undertrained — it collapsed toward the majority class.
I switched to `warmup_ratio=0.1` and raised epochs to 6. **Observation:** this moved the
model from collapse to actually learning the `dd` and `reaction` classes.

**Key training decision #2 — class weighting to recover the scarce class.**
Even after the warmup fix, the model barely predicted `speculation`: it has no distinctive
surface signal, so the model defaulted those posts to a neighbor. I added balanced class
weights to the loss (`speculation` 1.25, `dd` 1.18, `reaction` 0.74) and selected the best
checkpoint by macro-F1 rather than accuracy. **Observation:** this lifted `speculation` off
the floor (recall ≈ 0.21 on the reported run) but did not solve the class — the model still
collapses most `speculation` posts into `reaction`. The weighting buys a little recall and
costs some precision elsewhere, a deliberate trade of raw accuracy for balance, consistent
with my stated metric (macro-F1). The reported model below is this class-weighted version.

---

## Baseline (Zero-Shot Groq)

**Approach:** each test post is classified by `llama-3.3-70b-versatile` via the Groq API,
zero-shot (no task-specific training), `temperature=0`. The system prompt names the
community and task, defines all three labels in plain language with one example each, and
instructs the model to output only the label name. Responses are matched to a label string;
unparseable responses are excluded (all 54/54 parsed in the reported run). Results are
computed on the **same locked test set** as the fine-tuned model. The full prompt is in the
notebook (Section 5).

---

## Evaluation Report

All metrics are on the same 54-post locked test set (dd 16, speculation 14, reaction 24).

### Headline

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Zero-shot baseline (Groq llama-3.3-70b) | **0.778** | **0.768** |
| Fine-tuned DistilBERT (class-weighted) | 0.722 | 0.650 |

Fine-tuning **did not beat the baseline** — macro-F1 is **−0.118** lower. A 70B
general-purpose model is a strong zero-shot baseline for a subjective 3-class task, and a
DistilBERT fine-tuned on 252 examples did not surpass it.

### Per-class (precision / recall / F1)

| Class | Baseline | Fine-tuned |
|---|---|---|
| dd | 0.93 / 0.81 / **0.87** | 0.79 / 0.94 / **0.86** |
| speculation | 0.60 / 0.64 / **0.62** | 0.50 / 0.21 / **0.30** |
| reaction | 0.80 / 0.83 / **0.82** | 0.72 / 0.88 / **0.79** |
| **macro avg** | 0.78 / 0.76 / **0.77** | 0.67 / 0.68 / **0.65** |

The two models are close on `dd` (0.87 vs 0.86) and `reaction` (0.82 vs 0.79). The entire
gap is `speculation`: baseline 0.62 F1 vs fine-tuned 0.30. `speculation` is the hardest
class for *both* models, but the baseline handles it far better.

### Confusion matrix (fine-tuned model)

Rows = true label, columns = predicted label.

| true ↓ / pred → | dd | speculation | reaction |
|---|---|---|---|
| **dd** | 15 | 1 | 0 |
| **speculation** | 3 | 3 | 8 |
| **reaction** | 1 | 2 | 21 |

The dominant error is **`speculation` → `reaction`** (8 posts) — more than half of all 15
errors come from this single cell. Together with the smaller `speculation` → `dd` (3) and
`reaction` → `speculation` (2) cells, the model's whole problem lives on the boundaries
around `speculation`. `dd`, by contrast, is learned cleanly: it is never confused with
`reaction` (0 cases) and only once with `speculation`, so 15 of 16 `dd` posts are caught.

### Did I hit my success criteria? (set in planning.md, before results)

| Criterion | Target | Result | Met? |
|---|---|---|---|
| Beat baseline on macro-F1 | ≥ +10 pts | −11.8 pts | No |
| Every class F1 | ≥ 0.60 | speculation = 0.30 | No |
| Errors concentrate on `dd`↔`speculation` | — | concentrate on `speculation`↔`reaction` | No |
| Deployment bar: `dd` precision | ≥ 0.70 | 0.79 | Yes |

I missed all three primary criteria. My hypothesis that the hard boundary would be
`dd`↔`speculation` was **wrong** — the real confusion is `speculation`↔`reaction`. The one
bar I cleared is the deployment sub-criterion: `dd` precision is 0.79, so the model *is*
reliable enough at flagging genuine analysis to be useful as a triage tool, even though it
can't isolate speculation.

### Three wrong predictions, analyzed

1. **Short hype-toned call reads as pure emotion (`speculation` → `reaction`, conf 0.65).**
   *"What do you guys think? I Think Bullish"* — True `speculation` (a directional claim).
   It's ultra-short and casual, with the claim buried in a question frame, so the model saw
   only the reaction-like surface. This is the `speculation`↔`reaction` collapse in
   miniature, and it is the single biggest error cell in the matrix (8 of 15 errors):
   speculation with no analytical surface is indistinguishable from reaction to a
   surface-driven model.

2. **Quant `dd` misread as emotion (`dd` → `reaction`, conf 0.40).**
   *"SOFI 5/15 calls: macro lotto or theta donation? Positions: 125x SOFI $16C 5/15 @ $0.09…"*
   — True `dd` (specific option positions, strikes, expiries). The model predicted
   `reaction` at very low confidence (0.40). The casual, self-deprecating framing ("macro
   lotto or theta donation?") gives it a reaction-like *tone* that overrode the concrete
   position detail. This is the rare `dd` miss (only 1 of 16), and the low confidence shows
   the model was genuinely torn — the substantive surface was there but the flippant voice
   pulled it toward `reaction`.

3. **Costume of analysis fools the model (`speculation` → `dd`).**
   A post formatted with a bold "Fundamental Analysis" header and macro vocabulary but
   carrying no checkable evidence — hype wearing the costume of analysis. Three `speculation`
   posts were pulled into `dd` this way (the `speculation` → `dd` cell). The model keys on
   *structure* (headers, length, finance words) rather than whether any claim is actually
   supported. This is exactly my planning "fake DD" edge case: structure ≠ evidence, but the
   model can't tell.

### Sample classifications (fine-tuned model)

| Post (truncated) | True | Predicted | Confidence | Correct? |
|---|---|---|---|---|
| "Costco trades at ~42x forward earnings, priced for perfection. Margins contracted three qu…" | dd | dd | 0.42 | Yes |
| "Thank God I had class. I was going to buy PUTS on $DOCU for ER" | reaction | reaction | 0.63 | Yes |
| "Upcoming Stock Market Drop Will Be Epic Fury. Fundamental Analysis: he wants oil high and…" | speculation | speculation | 0.47 | Yes |
| "SOFI 5/15 calls: macro lotto or theta donation? Positions: 125x SOFI $16C 5/15 @ $0.09…" | dd | reaction | 0.40 | No |
| "What do you guys think? I Think Bullish" | speculation | reaction | 0.65 | No |

*Why a correct one is reasonable:* the Costco post is classified `dd` and that is exactly
right. The post lays out concrete, checkable financial evidence — a ~42x forward multiple,
sequential margin compression, membership-income headwinds — which is precisely the surface
that genuinely correlates with `dd`. `dd` is the model's strongest class (15 of 16 correct),
because that substantive, number-heavy surface is learnable and really does mark genuine
analysis. The relatively low 0.42 confidence is honest: the post is terse, so the model is
right but not emphatic.

---

## Reflection — what the model learned vs. what I intended

I intended the model to learn a **claim-vs-evidence logic**: does the post argue from
checkable evidence (`dd`), assert a directional claim without it (`speculation`), or express
feeling with no claim (`reaction`)? What it actually learned was **surface correlates**:
numbers/length/structure → `dd`, and short/emotional → `reaction`. Those two heuristics work
well, which is why `dd` (F1 0.86) and `reaction` (0.79) are strong.

The class it could not learn is the *abstract middle*. `speculation` is defined by an
*absence* (no real evidence) plus a *subtle* directional claim — it has no distinctive
surface, so a surface-driven model has nothing to grab. Short speculation looks like
reaction (the 8-post `speculation`→`reaction` cell), and costume-speculation looks like `dd`
(the 3-post `speculation`→`dd` cell). Class weighting pushed the model to predict
`speculation` a little more, but it didn't teach the real boundary — the class still
collapses, mostly into `reaction`. The model captured the surface-correlated classes and
missed the one that requires reasoning about *what kind of statement* a post is — which is,
fittingly, the exact boundary I flagged as hardest before I started, even though the specific
confusion (`speculation`↔`reaction`) differed from the `dd`↔`speculation` confusion I
predicted.

---

## Spec Reflection

**One way the spec helped:** its insistence on designing the label taxonomy *first*
(Milestone 1) and explicitly naming the hardest edge case forced me to write precise,
function-over-surface decision rules before touching any code. That paid off at evaluation
time: the model's failures mapped cleanly onto edge cases I'd already named (the "fake DD"
costume case, the surface-fights-structure case), so the error analysis was legible instead
of a guessing game.

**One way my implementation diverged:** the spec assumed I'd collect data directly from my
community, but Reddit's API gating made live scraping impossible for a student project, so I
pivoted to a public pre-collected dataset (text only, my own labels). I also diverged from
the notebook's "use the defaults" framing — the default warmup/epochs undertrained on my
small dataset, so I had to diagnose and fix the warmup-vs-total-steps bug and add class
weighting to get the model to learn at all.

---

## AI Usage

1. **Annotation (disclosed).** I directed an LLM to pre-label all 453 candidate rows using my
   planning.md definitions. I then reviewed and corrected **every** row myself: I dropped ~87
   rows that didn't fit the taxonomy (questions, bare news, off-topic) and overrode ~14
   `dd` labels down to `speculation` after deciding the model had been too generous with what
   counts as evidence.
2. **Label stress-testing.** Before annotating, I had an LLM generate posts that deliberately
   sat on the `dd`↔`speculation` and `speculation`↔`reaction` boundaries, then classified them
   myself. Where I couldn't classify one cleanly, I tightened the definition — this is where
   the "surface fights structure" edge case came from.
3. **Debugging / iteration.** When fine-tuning underperformed, I used an AI assistant to help
   diagnose the cause. It identified that `warmup_steps` exceeded total training steps and
   proposed class weighting; I implemented both, ran them, and verified the results myself. I
   also **overrode** the instinct to keep retraining toward beating the baseline once I saw
   that per-class results swing several points run-to-run on a 54-post test set (~14–24 posts
   per class) — chasing a single lucky run wasn't meaningful, so I locked one trained model as
   the source of truth and reported it.
4. **Drafting.** AI assistance helped structure planning.md and this README, which I edited
   and own.

---

## Repository Contents

- `planning.md` — design doc (labels, edge-case rules, data plan, metrics, success criteria, AI tool plan)
- `takemeter_data.csv` — 360 labeled examples (`text`, `label`)
- `README.md` — this file
- `evaluation_results.json` — full metrics for both models
- `confusion_matrix.png` — fine-tuned model confusion matrix
- `ai201_project3_takemeter.ipynb` — the Colab notebook (training + baseline + evaluation)
- `Demo Video` — Watch the 3-5 minute demo here(https://drive.google.com/file/d/140NphXA7qJ4DWsse0apLZ0HNGR60BhAo/view?usp=sharing)

### Reproduce
1. Open the notebook in Colab, set runtime to T4 GPU.
2. Sections 1–2: define the label map, upload `takemeter_data.csv`, split + tokenize.
3. Section 3: fine-tune (6 epochs, class-weighted loss, macro-F1 checkpoint selection).
4. Section 4: evaluate on the test set; export the confusion matrix.
5. Section 5: add a Groq API key (Colab Secrets), run the zero-shot baseline.
6. Section 6: export `evaluation_results.json`.