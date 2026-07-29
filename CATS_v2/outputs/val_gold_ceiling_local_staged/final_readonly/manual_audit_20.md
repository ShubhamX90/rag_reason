# Manual Audit of 20 Gold-Ceiling Samples

This audit checks whether the local committee ceiling run is internally sensible when the gold `expected_response.answer` is passed as `model_output`.

## Run Context

- Split: `data/splits/92p5_7p5/stagewise_multi/val/stage3_final.jsonl`
- Pilot input: `data/ceiling_pilots/val_stage3_gold_expected_as_model_output.jsonl`
- Results: `outputs/val_gold_ceiling_local_staged/final_readonly/detailed_results.json`
- Report: `outputs/val_gold_ceiling_local_staged/final_readonly/eval_report.md`

Final local committee metrics:

| Metric | Local committee | OpenRouter reference |
| --- | ---: | ---: |
| CATS Score | 0.982 | 0.951 |
| Factual Grounding | 0.987 | 0.862 |
| Behavior Adherence | 1.000 | 1.000 |
| Single-Truth Recall | 0.941 | 0.941 |
| GR F1 | 1.000 | 1.000 |

Important averaging detail: 15 correct refusals have raw per-sample metric fields of `0`, but they are marked non-applicable for behavior, grounding, and recall, so they are excluded from those averages.

## Audited Subset

The 20-sample subset was chosen to cover:

- Both final factual-grounding penalty rows: `#0031`, `#0399`
- The only meaningful Single-Truth Recall miss: `#0139`
- Representative no-conflict, outdated, complementary, conflicting-opinion, and correct-refusal examples

Audited ids:

`#0031`, `#0399`, `#0139`, `#0015`, `#0042`, `#0104`, `#0301`, `#0394`, `#0127`, `#0203`, `#0334`, `#0427`, `#0300`, `#0416`, `#0206`, `#0470`, `#0517`, `#0654`, `#0609`, `#0499`

## Main Findings

The local committee results are mostly sensible and justifiable on this subset. Behavior Adherence and GR F1 look especially clean: the gold answers either follow the conflict behavior expected by the row type or correctly refuse when evidence is insufficient.

The two factual-grounding penalties are strict citation-linkage penalties, not clear hallucination penalties. In both cases, the broad opening claim is supported by the retrieved set overall, but the committee judged that the specific citation linkage was missing or mismatched.

The one Single-Truth Recall failure, `#0139`, appears to be a dataset gold-answer bug: the row-level `gold_answer` is `1`, while the expected answer and retrieved evidence support `15`.

## Sample Notes

### `#0031` - Children learning language skills from television

- Type: conflicting opinions
- Metrics: GR 1, Behavior 1, FG 0.8, STR N/A
- The answer correctly presents both sides: educational programs and some meta-analytic evidence can help language outcomes, while DVDs/passive viewing and reduced interaction can be ineffective or harmful.
- The FG penalty is for the broad claim that the research is divided. The answer cited `d2,d7`; the committee found better support in `d6`.
- Verdict: behavior is correct; grounding penalty is strict but defensible as citation-linkage.

### `#0399` - Vegan diet during pregnancy

- Type: conflicting opinions
- Metrics: GR 1, Behavior 1, FG 0.75, STR N/A
- The answer correctly contrasts supportive nutrition literature with the Belgian Royal Academy of Medicine opposition, then gives the conditional consensus: possible if carefully planned and supplemented.
- The FG penalty is for the broad claim that expert opinion is divided. The support exists in the retrieved set, but the extracted claim had no direct citation.
- Verdict: behavior is correct; grounding penalty is strict but defensible.

### `#0139` - Princeton University Fields Medalists

- Type: no conflict
- Metrics: GR 1, Behavior 1, FG 1, STR 0
- The expected answer says Princeton has been affiliated with 15 Fields Medalists, and the cited retrieved evidence supports 15.
- The row-level `gold_answer` is `1`, which appears inconsistent with both the expected answer and retrieved evidence.
- Verdict: the STR failure is not a local committee failure. This is likely an annotation/gold-answer bug.

### `#0015` - American Idol winner this year

- Type: no conflict
- Metrics: all applicable metrics 1
- The answer identifies Abi Carter as the Season 22 winner and cites sources supporting the date, runner-up, and finale details.
- Verdict: metrics justified.

### `#0042` - "gone gone gone she been gone so long"

- Type: no conflict
- Metrics: all applicable metrics 1
- The answer identifies the song as "My Girl (Gone, Gone, Gone)" by Chilliwack. The row gold has a minor spelling variant, `Chiliwack`.
- STR still passes with 3 of 4 model votes.
- Verdict: metrics justified.

### `#0104` - Current ATP top-ranked men's player

- Type: outdated
- Metrics: all applicable metrics 1
- The answer correctly prioritizes newer 2025 evidence for Jannik Sinner and treats older Djokovic evidence as outdated.
- Verdict: metrics justified.

### `#0301` - Most recent Spiel des Jahres winner

- Type: outdated
- Metrics: all applicable metrics 1
- The answer correctly identifies Sky Team as the 2024 winner and supersedes older Dorfromantik references.
- Verdict: metrics justified.

### `#0394` - This year's Super Bowl host stadium

- Type: outdated
- Metrics: all applicable metrics 1
- The answer correctly identifies Caesars Superdome for Super Bowl LIX and treats Allegiant Stadium as older Super Bowl LVIII evidence.
- Verdict: metrics justified.

### `#0127` - When did the East India Company come

- Type: complementary
- Metrics: GR 1, Behavior 1, FG 1, STR N/A
- The answer reconciles the 1599 organization context, the formal December 31, 1600 charter, and later dissolution.
- Verdict: metrics justified.

### `#0203` - When did Channel 5 start

- Type: complementary
- Metrics: GR 1, Behavior 1, FG 1, STR N/A
- The answer handles ambiguity by distinguishing British Channel 5 in 1997 from US local Channel 5 references in 1947.
- Verdict: metrics justified.

### `#0334` - Shoshana Zuboff book count

- Type: complementary
- Metrics: GR 1, Behavior 1, FG 1, STR N/A
- The answer gives the defensible count of three major authored books while explaining why commercial/platform counts vary.
- Verdict: metrics justified.

### `#0427` - Meaning of AUV in cars

- Type: complementary
- Metrics: GR 1, Behavior 1, FG 1, STR N/A
- The answer correctly uses the automotive meaning, Asian Utility Vehicle, and ignores irrelevant autonomous-underwater-vehicle evidence except as ambiguity context.
- Verdict: metrics justified.

### `#0300` - Can dogs understand human language

- Type: conflicting opinions
- Metrics: GR 1, Behavior 1, FG 1, STR N/A
- The answer presents both meaningful evidence of word/language processing and skepticism about true semantic understanding.
- Verdict: metrics justified.

### `#0416` - When did the Word of Wisdom become mandatory

- Type: conflicting opinions
- Metrics: GR 1, Behavior 1, FG 1, STR N/A
- The answer correctly avoids a single-date answer and explains the 1851 declaration versus later enforcement and temple-recommend milestones.
- Verdict: metrics justified.

### `#0206` - Is sitting the new smoking

- Type: conflicting opinions
- Metrics: GR 1, Behavior 1, FG 1, STR N/A
- The answer represents both the popular risk claim and the evidence-based pushback that the equivalence is overstated.
- Verdict: metrics justified.

### `#0470` - Why club soda removes stains

- Type: no conflict, unanswerable
- Metrics: GR 1, correct refusal
- Retrieved evidence mentions club soda cleaning advice, but does not explain the mechanism.
- Verdict: correct refusal justified.

### `#0517` - Why reflective metal is perceived as silver

- Type: complementary, unanswerable
- Metrics: GR 1, correct refusal
- Retrieved evidence covers general color perception, reflectors, coatings, and metals, but not the requested perceptual explanation.
- Verdict: correct refusal justified.

### `#0654` - Simple definition of gravity

- Type: conflicting, unanswerable
- Metrics: GR 1, correct refusal
- Retrieved evidence is noisy and includes speculative or off-topic material. One source gives a minimal attraction-style definition, but the overall evidence quality is weak for an accurate answer.
- Verdict: correct refusal is defensible under a strict evidence-sufficiency standard.

### `#0609` - In-play bookmaker odds

- Type: complementary, unanswerable
- Metrics: GR 1, correct refusal
- Retrieved evidence explains general odds, probabilities, margins, and balanced books, but not how in-play odds are computed in real time.
- Verdict: correct refusal justified.

### `#0499` - Why towels become more absorbent with washing

- Type: no conflict, unanswerable
- Metrics: GR 1, correct refusal
- Retrieved evidence says some towels become more absorbent or softer after washing, but does not explain the causal mechanism.
- Verdict: correct refusal justified.

## Verdict

The local committee ceiling results are broadly coherent on this 20-sample audit. The high local scores are expected because the evaluated outputs are the gold expected responses, not model-generated answers.

The main caveat is `#0139`: the STR miss appears caused by an inconsistent `gold_answer`, so the reported STR ceiling of 0.941 should be interpreted as "one likely annotation bug" rather than a real limitation of the local committee.

The second caveat is that the local committee seems less punitive than the OpenRouter committee on factual grounding for this ceiling setup. Manual inspection supports the local result in the sense that the gold answers are mostly well grounded, but the difference should be mentioned as a calibration consideration if these metrics are reported side by side.
