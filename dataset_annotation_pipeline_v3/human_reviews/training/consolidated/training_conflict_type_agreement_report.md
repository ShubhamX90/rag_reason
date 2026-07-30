# Training Conflict-Type Human-Review Agreement Statistics

## Consolidated population

The final released training population contains **943 records**: 862 training records and 81 validation records. Every record is represented by a common first/second conflict-type review schema for one consistent agreement analysis.

## Inter-annotator agreement (IAA)

### What is the IAA score in this study?

Inter-annotator agreement (IAA) measures the reliability of labels assigned to the same example by two reviewers. It is field-specific rather than a single universal number. The **primary training-set IAA** is Cohen's κ for the nominal five-way `reviewed_conflict_type` taxonomy across all 943 review pairs: **κ = 0.7694**, with **83.46% raw agreement**. This is the principal result to report for the reliability of the training conflict taxonomy.

Raw agreement gives the directly observed proportion of equal labels. Cohen's κ is the corresponding chance-corrected agreement coefficient: it accounts for agreement expected from the reviewers' marginal category frequencies. For categories `c`, observed agreement is `P_o`; expected agreement is `P_e = Σ_c p_{1,c} p_{2,c}`; and `κ = (P_o − P_e) / (1 − P_e)`.

### Calculation

For `N` review pairs, with first and second labels `y_i^(1)` and `y_i^(2)`, the reported quantities are:

$$ P_o = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[y_i^{(1)} = y_i^{(2)}] $$

$$ P_e = \sum_{c \in C} p_{1,c} p_{2,c} $$

$$ \kappa = \frac{P_o - P_e}{1 - P_e} $$

Here, `C` is the five-label conflict taxonomy, `p_{1,c}` and `p_{2,c}` are the reviewers' marginal proportions for category `c`, and `𝟙[·]` equals 1 when the labels match and 0 otherwise. Raw agreement is `P_o`; Cohen's κ is the chance-corrected agreement score. The implementation applies these formulas to the full 943 paired records.

The conflict taxonomy is nominal, so no ordinal weighting is applied. This is the standard two-reviewer nominal-label formulation introduced by [Cohen (1960)](https://journals.sagepub.com/doi/abs/10.1177/001316446002000104). Agreement coefficients should be interpreted with their task definition, label inventory, and category distributions, as discussed by [Artstein and Poesio (2008)](https://aclanthology.org/J08-4004/).

### Annotation dimensions and use

| Dimension | Type | What it establishes |
|---|---|---|
| `reviewed_conflict_type` | Five-way nominal taxonomy | Reliability of the final training conflict-type label; primary IAA. |
| `label_action` / `changed_label` | Binary retain-or-change decision | Stability of human validation versus the original stagewise committee label. |
| `review_confidence` | Ordered categorical diagnostic | Consistency of reviewer certainty; supplementary rather than a label-quality target. |

### Review-field glossary

| Field | Values | Meaning in this review |
|---|---|---|
| `reviewed_conflict_type` | No conflict; Complementary information; Conflicting opinions or research outcomes; Conflict due to outdated information; Conflict due to misinformation | The reviewer's final judgment of the conflict type supported by the retrieved documents. This is the released-training-label reliability target and the primary IAA field. |
| `label_action` | `accept_as_is`, `change_label` | Whether the reviewer retained the original stagewise committee label or replaced it with a different conflict type. |
| `changed_label` | `false`, `true` | Boolean version of `label_action`: `true` exactly when the final reviewed conflict type differs from the original committee label. It is reported alongside `label_action` as an equivalent audit-friendly encoding. |
| `review_confidence` | `high`, `medium`, `low` | The reviewer's stated confidence in their own conflict-type judgment after inspecting the query and retrieved evidence. It measures certainty, not a separate conflict label. |

## Overall training-review reliability (n = 943)

Field | n | Raw agreement | Cohen's kappa
---|---:|---:|---:
reviewed_conflict_type | 943 | 83.46% | 0.7694
label_action | 943 | 83.88% | 0.1341
changed_label | 943 | 83.88% | 0.1341
review_confidence | 943 | 64.37% | 0.0159

### Interpretation

The primary conflict-type IAA is **κ = 0.7694** with **83.46% raw agreement**. This indicates substantial stability of the final five-way conflict taxonomy used by the released training and validation data. The retain/change and confidence rows are supporting process diagnostics rather than substitutes for the taxonomy IAA.

The lower chance-corrected values for `label_action`, `changed_label`, and `review_confidence` should be read with their markedly imbalanced category distributions in mind: most reviewers accepted the committee label and selected high confidence. In such settings, raw agreement and κ answer different questions, which is why both are reported. The conflict-type κ remains the appropriate headline IAA because it corresponds directly to the released dataset's target label.

## Relationship to the committee-assigned label

Of 943 records, the first review retains the original committee label for 801 (84.94%); the second review retains it for 895 (94.91%); both reviews retain it for 772 (81.87%). These figures characterize how the human review layer validates or corrects the stagewise committee's initial conflict-type assignment.

## Final label distribution

```json
{
  "Complementary information": 289,
  "Conflict due to misinformation": 31,
  "Conflict due to outdated information": 130,
  "Conflicting opinions or research outcomes": 120,
  "No conflict": 373
}
```

## Files

- `training_conflict_type_consensus_943.jsonl`: consolidated review pair for every released training/validation item.
- `training_conflict_type_agreement_metrics.csv`: paper-ready IAA table.
- `training_conflict_type_agreement_metrics.json`: machine-readable summary.

## References

- Cohen, J. (1960). *A Coefficient of Agreement for Nominal Scales*. Educational and Psychological Measurement, 20(1), 37–46. https://doi.org/10.1177/001316446002000104
- Artstein, R., & Poesio, M. (2008). *Inter-Coder Agreement for Computational Linguistics*. Computational Linguistics, 34(4), 555–596. https://doi.org/10.1162/coli.07-034-R2
