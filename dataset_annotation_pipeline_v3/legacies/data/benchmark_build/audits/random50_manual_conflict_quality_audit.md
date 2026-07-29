# Random 50 Manual Conflict Quality Audit

Source file: `data/benchmark_build/retrieved/full650_tavily_20_keep10_tasb_readable_raw_with_fulltext_annotation_candidates.jsonl`

Sample generation: deterministic random sample of 50 records from the 632 usable retrieved records, using seed `61001`.

Reviewed evidence: top retrieved snippets and search snippets for each sampled query, usually top 6 docs per record.

## Summary

| Manual bucket | Count | Notes |
| --- | ---: | --- |
| No conflict | 20 | Retrieval is on-topic and mostly converges on one answer. |
| Complementary information / ambiguity | 9 | Usually scope, geography, measurement, or "we/current" ambiguity. |
| Conflicting opinions or research outcomes | 9 | Good evidence tension, especially medical/science/policy questions. |
| Conflict due to outdated information | 5 | Strong temporal cases where the answer changes over time. |
| Conflict due to misinformation | 5 | False premise or entity/event did not happen. |
| Low-quality / reject | 2 | One malformed query and one bad seed-answer case. |

Approximate quality: 39 high-confidence, 9 medium-confidence, 2 reject/low-confidence.

## Per-Record Labels

| # | id | source | manual bucket | confidence | note |
| ---: | --- | --- | --- | --- | --- |
| 1 | conflictingqa_d82e9ca5971a | conflictingqa | Conflicting opinions or research outcomes | medium-high | "All plants" vs caveats about non-photosynthetic plants. |
| 2 | situatedqa_temp_88bfa0096ff7 | situatedqa_temp | Conflict due to outdated information | high | Highest-paid actor varies by year/current list. |
| 3 | freshqa_fab2a6900efb | freshqa | No conflict | high | LSTM student author evidence converges on Sepp Hochreiter. |
| 4 | situatedqa_temp_daa3b7660964 | situatedqa_temp | No conflict | high | Corbin Bleu partnered with Karina Smirnoff. |
| 5 | situatedqa_geo_8481133d6a86 | situatedqa_geo | Complementary information / ambiguity | medium | "current health minister" is country-underspecified. |
| 6 | freshqa_69b23c159a7b | freshqa | Conflict due to misinformation | high | 2022 Afghanistan election premise is false after Taliban takeover. |
| 7 | freshqa_4a98eba95e97 | freshqa | Conflict due to misinformation | high | Biden did not visit Russia as president; older VP/meeting docs can mislead. |
| 8 | conflictingqa_3f3c3399259a | conflictingqa | Conflicting opinions or research outcomes | high | Fish oil evidence contains positive, null, and risk claims. |
| 9 | freshqa_1009f5c49e12 | freshqa | No conflict | high | Louvre location converges on Paris. |
| 10 | situatedqa_temp_f98a03f49fa7 | situatedqa_temp | Conflict due to outdated information | medium | "last World Cup" is temporally unstable/ambiguous. |
| 11 | situatedqa_geo_27aefd8e4a26 | situatedqa_geo | No conflict | high | National river evidence converges. |
| 12 | qacc_6b6a04b1c927 | qacc | No conflict | high | World Consumer Rights Day converges on March 15. |
| 13 | conflictingqa_bea0b68b5d7f | conflictingqa | Conflicting opinions or research outcomes | high | Paper vs plastic straw evidence is genuinely split. |
| 14 | freshqa_4d8566da53ca | freshqa | Conflict due to misinformation | high | Dartmouth Law School premise is false / confused with UMass Dartmouth. |
| 15 | freshqa_a05df5979a46 | freshqa | Conflicting opinions or research outcomes | medium | Jefferson vs Cleveland depends on interpretation of incumbent. |
| 16 | situatedqa_geo_8f3897efcc4f | situatedqa_geo | Complementary information / ambiguity | medium | "we" in Thirty Years' War is underspecified. |
| 17 | qacc_c2975d69d57c | qacc | No conflict | high | Lois's father evidence converges on Carter Pewterschmidt. |
| 18 | situatedqa_temp_7dd0bea41e4a | situatedqa_temp | Complementary information / ambiguity | high | Australia coastline differs by measurement scope. |
| 19 | conflictingqa_a02e3c5f0df0 | conflictingqa | Conflicting opinions or research outcomes | high | Compression garment performance evidence is mixed. |
| 20 | situatedqa_temp_4b1b506f4f5b | situatedqa_temp | Conflict due to outdated information | high | LeBron career points changed substantially from seed. |
| 21 | situatedqa_geo_531a300e92c1 | situatedqa_geo | Complementary information / ambiguity | medium-high | US troops to Europe in WWII depends on "Europe" meaning Iceland/arrival/invasion. |
| 22 | qacc_a37c797a2b1e | qacc | No conflict | high | First English child in North America converges on Virginia Dare. |
| 23 | freshqa_818f72053b7c | freshqa | No conflict | high | Most populous city in Chile converges on Santiago. |
| 24 | situatedqa_temp_587e89bbcbe1 | situatedqa_temp | Complementary information / ambiguity | medium-high | Uno deck count differs by edition. |
| 25 | conflictingqa_544ebeeccda5 | conflictingqa | No conflict | medium | Bicarbonate/CKD evidence mostly supports benefit; weak conflict signal. |
| 26 | conflictingqa_b2524e4883ad | conflictingqa | Conflicting opinions or research outcomes | medium-high | Meteor shower risk is framed as no practical threat vs small indicator/threat. |
| 27 | situatedqa_geo_c1630378c760 | situatedqa_geo | Complementary information / ambiguity | high | Lead-paint ban date differs by jurisdiction/legal scope. |
| 28 | freshqa_8db26074e92b | freshqa | Conflict due to misinformation | medium-high | Current-member premise clashes with One Direction inactivity/former-member evidence. |
| 29 | situatedqa_geo_7d3c28d8ac77 | situatedqa_geo | No conflict | high | US presidential age requirement converges on 35. |
| 30 | qacc_4387048ed24f | qacc | Low-quality / reject | low | Seed says Bette Davis, but retrieved evidence indicates the film's Oscar was costume design / Davis was nominee. |
| 31 | situatedqa_temp_fe419d876f3b | situatedqa_temp | No conflict | high | Queensland 2010/2011 flood dates converge, with some distractors. |
| 32 | conflictingqa_a7ff288bc615 | conflictingqa | Conflicting opinions or research outcomes | high | AI/Turing-test evidence splits across passed, sort of passed, and does not pass framings. |
| 33 | situatedqa_temp_8fc5a9f2d826 | situatedqa_temp | No conflict | high | India's first Test captain converges on C. K. Nayudu. |
| 34 | qacc_92f6e17665ce | qacc | No conflict | high | Glacial abrasion scratches converge on striations/striae. |
| 35 | freshqa_a5492f36ca23 | freshqa | No conflict | high | David Bowie death date converges. |
| 36 | conflictingqa_52e01830d2fe | conflictingqa | Conflicting opinions or research outcomes | high | Software patent policy evidence is genuinely debated. |
| 37 | situatedqa_geo_32ea0dca7eb5 | situatedqa_geo | Complementary information / ambiguity | medium | "we" in America's Cup gives US/Australia/NZ possibilities. |
| 38 | situatedqa_temp_d0579ca3907c | situatedqa_temp | Complementary information / ambiguity | medium | Battle of Kadesh date varies slightly by source/year convention. |
| 39 | situatedqa_geo_e8294455225c | situatedqa_geo | No conflict | high | State admission evidence converges on Congress, with presidential signature as process detail. |
| 40 | situatedqa_temp_16a9feb3c3e3 | situatedqa_temp | Complementary information / ambiguity | medium | "Most popular religion" can mean followers vs active practice. |
| 41 | freshqa_ab6a7c726697 | freshqa | Conflict due to outdated information | high | "Days left until 2022 World Cup" is now elapsed, not future. |
| 42 | freshqa_e502143179d6 | freshqa | No conflict | high | Musk/Twitter ownership date converges on Oct 27, 2022. |
| 43 | freshqa_5d6e5db69928 | freshqa | No conflict | high | Oldest DNA location converges on northern Greenland. |
| 44 | conflictingqa_6ea6bbcb8743 | conflictingqa | No conflict | high | Split ends cannot be truly repaired, only temporarily masked. |
| 45 | freshqa_439200ea9f67 | freshqa | Conflict due to misinformation | high | Zuckerberg did not acquire Twitter; rejected/past offer vs Musk purchase. |
| 46 | situatedqa_geo_22b8130e4a96 | situatedqa_geo | Low-quality / reject | low | Query is malformed/fragmentary: "according the treaty did not have to". |
| 47 | freshqa_1e3618891c7a | freshqa | No conflict | high | Menelaus's wife converges on Helen. |
| 48 | freshqa_08e9cdd2480f | freshqa | Conflict due to outdated information | medium-high | SLS launch count is temporally sensitive and docs differ by current status/scheduled launches. |
| 49 | conflictingqa_b2baeb94a759 | conflictingqa | No conflict | high | Termite evidence converges on "not all harmful; some are destructive." |
| 50 | conflictingqa_24fa0020a521 | conflictingqa | Conflicting opinions or research outcomes | high | Child multivitamin guidance splits between usually unnecessary and recommended. |

