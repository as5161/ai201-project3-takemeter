# TakeMeter — Planning

Project: a fine-tuned classifier that scores discourse quality on r/wallstreetbets.

---

## 1. Community

**Community:** r/wallstreetbets

r/wallstreetbets is a high-volume retail-investing community whose discourse ranges
from genuine due-diligence write-ups to pure emotional reactions to market moves.
I'm classifying posts as `dd`, `speculation`, or `reaction` (defined below). This
distinction matters to participants because the community itself constantly polices
the line between real "DD" and hype dressed up as analysis — "great DD bro 🤡" is a
recognized form of mockery there.

**Why it's a good fit for a classification task:** the quality of takes varies
enormously and visibly, the community has native vocabulary for that variation
(DD, YOLO, FUD, bagholder, hype), and the volume means plenty of public examples.
The catch I'm managing: it skews toward emotional/hype content, so I sampled
deliberately to keep the classes balanced (see §4).

---

## 2. Labels (3, mutually exclusive)

**Operative test for the analysis boundary:** if removing the opinion framing still
leaves *specific, checkable evidence* that supports the claim, it's `dd`. If the
"evidence" is vague, decorative, or absent, it's `speculation`. If there's no claim
being argued at all — only feeling — it's `reaction`.

### `dd`
Structured reasoning backed by specific, verifiable evidence (financials, valuation
multiples, catalysts, guidance, option mechanics). You could argue with it on its merits.

- Example 1: Supreme Court is going to decide on the case of West Virginia v. EPA tomorrow and based on previous opinions from the court, Im pretty confident theyre going to limit the EPAs power. This will make it so that Congress will have to decide on laws for the regulation of power plants... This will be a big boon for the utility companies, especially those that still have a lot of coal plants in operation like Duke Energy.

- Example 2: COST Short Thesis — Costco trades at ~42x forward earnings, priced for perfection. My bearish tilt rests on three quantitative flags: gross margin compression (core merchandise margins contracted sequentially for three quarters), membership income facing comp headwinds, and durable-goods deflation stalling average transaction value growth.

### `speculation`
A confident directional call or conviction play asserted *without* real supporting
reasoning. The claim may be right, but the post asserts rather than argues — including
posts that *wear the costume* of analysis (asserted mechanisms, buzzwords) but contain
no checkable evidence.

- Example 1: We got 2 gaps to be closed, market is oversold at this point and the rate hike was 75 bps not 100 bps... In the next 2 to 3 weeks we're blasting higher and if the bulls can close the gap at 4080 to 4108... the bears are going to feel the squeeze... I'm feeling a early santa-rally in sept HO HO HOLD !!!

- Example 2: JPow and Musk are going to send us to Nirvana, they will dump the stock market by raising rates. Once the US 10yr hits 4.20% the play is TSLA 400c 3 months out. You heard it here first.

### `reaction`
Emotional or momentum expression about a move or position — hype, panic, celebration,
loss/gain posts. No claim is being argued.

- Example 1: Look, I don't give a shit about your holidays. I want to watch stocks go up and down. I want to look at charts moving and convince myself I can see patterns in it that will make me money. Stop taking this away from me.

- Example 2: Lmfao so many puts got bent over 🤡🤡🤡

---

## 3. Hard Edge Cases

**Edge case A — fake DD / sarcasm (the signature WSB problem).**
WSB is drenched in irony: sarcastic praise ("great DD bro 🤡") and posts that mimic
analysis structure without substance.
*Decision rule:* label by **function, not surface**. Sarcastic praise is `reaction`.
A post that mimics DD structure but carries no checkable evidence is `speculation` or
`reaction`, never `dd` — the evidence test fails.

**Edge case B — asserted-mechanism hype (speculation vs. reaction).**
e.g. "AMC is the play. Shorts are trapped. They HAVE to cover. It's basic math."
A meme-stock rallying cry with real emotional charge, but it advances a directional
claim with an (unsupported) mechanism.
*Decision rule:* if a directional claim is present, it's `speculation` even when the
support is bogus; pure emotion with no claim is `reaction`.

**Edge case C — surface fights structure (a model risk, not a labeling problem).**
"🚀🚀 PLTR to $100 by EOY 🚀🚀" carries an explicit target (→ `speculation`) but a
surface (rocket emoji) that screams `reaction`. A rule-following human gets these right;
the model likely will not. I deliberately kept such posts in the dataset.

### Three genuinely difficult posts I had to rule on during annotation

1. **Rigorous-looking technical analysis (dd vs. speculation).**
   *Post:* "I've just purchased $VET as it breaks through the daily and weekly zone...
   Notice how CVX and OXY yanked back when oil futures pulled in but $VET kept moving.
   This is the relative strength you want to see... clear skies ahead if it can hold."
   *Why hard:* it names specific tickers, makes a comparative observation, and includes
   risk management — it *reads* like analysis.
   *Decision → speculation.* Chart-zone reading and "relative strength" are pattern
   interpretation, not verifiable evidence. Under my operative test, TA is decorative,
   so the post asserts a directional trade without checkable support.

2. **Bare earnings/news drop (dd vs. drop).**
   *Post:* "Seagate +16% after-hours on Q3 2026 earnings: EPS $5 vs $3.97 est, revenue
   $3.45B vs $3.16B est."
   *Why hard:* it's just a headline with no explicit argument — a candidate for `drop`
   (doesn't fit a discourse-quality bucket).
   *Decision → dd.* The content *is* specific, checkable financial evidence (beats vs.
   estimates). My rule: a news drop carrying hard figures/beats counts as `dd`; a bare
   headline with no numbers and no take gets dropped.

3. **Satirical pseudo-DD (speculation vs. dd).**
   *Post:* "THE DEFLATION GLITCH: Why $2 Bills are the ultimate hedge... (The 'Due
   Diligence') only 1.7 billion $2 bills vs 14.9 billion $1 bills... psychological arbitrage."
   *Why hard:* full DD costume — headers, a "Due Diligence" section, real Fed figures.
   *Decision → speculation.* Per edge-case A, function over surface: structure plus real
   numbers ≠ `dd` when the reasoning doesn't support a genuine, checkable investment claim.

---

## 4. Data Collection (as executed)

**Planned vs. actual:** I originally planned to scrape r/wallstreetbets directly via the
Reddit API (PRAW), sampling deliberately across the DD flair, YOLO/positions posts, and
the Daily Discussion thread to balance the classes. Reddit's 2025 Responsible Builder
Policy now gates new Data API apps behind an approval process I couldn't satisfy for a
class project, so I pivoted to a public, pre-collected dataset.

**Source:** `StephanAkkerman/wallstreetbets-ner` on Hugging Face (public, real WSB posts).
I used the post **text only** and ignored the dataset's named-entity (ticker/company)
labels, which are irrelevant to my task. I applied my own `dd`/`speculation`/`reaction`
labels from scratch.

**Annotation process:** an LLM pre-labeled all 453 candidate rows from my definitions
(§2); I then reviewed and corrected **every** row. During review I dropped ~87 rows that
were pure questions, bare news headlines, or off-topic/meta posts (these don't fit a
discourse-quality taxonomy), and re-labeled ~14 posts from `dd` to `speculation` after
tightening my evidence bar.

**Final dataset:** 360 labeled examples in one CSV (`text`, `label`).
Distribution — reaction 163 (45%), dd 102 (28%), speculation 95 (26%). No class above
the 70% ceiling, so no rebalancing was needed.

---

## 5. Evaluation Metrics

I lead with **macro-averaged F1**, supported by **per-class precision/recall/F1** and a
**confusion matrix**. Overall accuracy is reported but not relied on.

Accuracy alone is misleading here because the classes are imbalanced — `reaction` is 45%
of the data, so a model that simply predicted `reaction` every time would score ~45%
accuracy while learning none of the boundaries I care about. Macro-F1 averages the three
per-class F1 scores equally regardless of class frequency, so it directly penalizes a
model that neglects the scarcer `dd` and `speculation` classes — exactly the failure mode
I'm worried about. Per-class F1 tells me *which* boundary the model did or didn't learn,
and the confusion matrix shows the *direction* of errors (e.g. whether `dd` is being read
as `speculation`), which is the most diagnostic output for a task whose entire difficulty
is the `dd`↔`speculation` line.

---

## 6. Definition of Success

Criteria set before viewing the fine-tuned model's results:

1. **Beats the zero-shot Groq baseline on macro-F1 by ≥ 10 points.** Fine-tuning must
   demonstrably add value over a general model with no task-specific training.
2. **Every class F1 ≥ 0.60, including `speculation`** (my hardest, scarcest class). A high
   overall score that hides a near-zero class is not a success.
3. **Errors concentrate on the `dd`↔`speculation` boundary, not at random.** If `reaction`
   is being confused with `dd`, the model failed to learn the structure I intended.

**Deployment bar:** useful as a community tool if it can reliably auto-flag likely `dd`
posts for human attention — i.e. `dd` precision ≥ 0.70 (flagged posts are usually genuinely
substantive), even if recall is lower.

**Noise caveat:** with ~360 examples the 15% test split is only ~54 posts (~14–24 per
class), so per-class F1 will be noisy. I'll treat per-class differences under a few points
as within noise rather than meaningful.

---

## AI Tool Plan

**Label stress-testing — DONE before annotation.** I gave an assistant my label definitions
and edge-case rules and had it generate boundary-sitting posts; I classified them and
tightened my rules where they leaked. (Result: the surface-vs-structure pattern in §3-C.)

**Annotation assistance — DONE (disclosed).** I used an LLM to pre-label all 453 candidate
rows from my definitions, then personally reviewed and corrected every row before training
(re-labeling ~14 `dd`→`speculation`, dropping ~87 non-fitting rows). Disclosed in the
README's AI usage section.

**Failure analysis — planned.** After fine-tuning, I'll give my list of misclassified
examples to an AI tool and ask it to surface patterns (label pair confused, post length,
sarcasm), then verify each pattern myself by re-reading the examples before writing it up.