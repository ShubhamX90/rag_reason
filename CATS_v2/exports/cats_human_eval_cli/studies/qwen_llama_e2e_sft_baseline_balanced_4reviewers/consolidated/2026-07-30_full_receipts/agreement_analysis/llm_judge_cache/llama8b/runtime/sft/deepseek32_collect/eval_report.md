# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 127 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.974 (over 736 samples)

**GR F1** *(used in CATS)*: 0.984

**Behavior Adherence**: 0.880 (over 609 applicable samples)

**Factual Grounding**: 0.863 (over 609 applicable samples)

**Single-Truth Recall**: 0.725 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.863

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.984
- **Precision**: 0.998
- **Recall**: 0.970
- **Accuracy**: 0.974
- TP=590, FP=1, FN=18, TN=127

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.876
- **Abstain Recall**: 0.992
- **Abstain F1**: 0.930
- **Specificity**: 0.970
- Abstain TP=127, FP=18, FN=1, TN=590


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (57 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.995
- **GR F1** *(used in CATS)*: 0.997
- **Behavior**: 0.948 (n=154)
- **Grounding**: 0.898 (n=154)
- **Recall**: 0.877 (n=154)
- **CATS**: 0.930

### Type 2: Complementary Info

- **Samples**: 221 (44 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.964
- **GR F1** *(used in CATS)*: 0.977
- **Behavior**: 0.853 (n=177)
- **Grounding**: 0.852 (n=177)
- **Recall**: 0.609 (n=156)
- **CATS**: 0.823

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.982
- **GR F1** *(used in CATS)*: 0.989
- **Behavior**: 0.792 (n=96)
- **Grounding**: 0.880 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.887

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.968
- **GR F1** *(used in CATS)*: 0.982
- **Behavior**: 0.903 (n=145)
- **Grounding**: 0.833 (n=145)
- **Recall**: 0.700 (n=140)
- **CATS**: 0.855

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.919
- **GR F1** *(used in CATS)*: 0.958
- **Behavior**: 0.865 (n=37)
- **Grounding**: 0.847 (n=37)
- **Recall**: 0.676 (n=37)
- **CATS**: 0.836


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2086

**Average Cost per Decision**: $0.000000


### Per-Model Cost Breakdown

#### local/deepseek-r1-distill-32b

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 2086
- **Average Cost per Request**: $0.000000


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample conflictingqa_0dba017da71c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that while this fungus is the largest single organism, a colony of aspen trees called Pando in Utah spans a larger land area (8 km), though the fungus covers a larger total mass

### Sample conflictingqa_11c5ef7c4545

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The debate is ongoing, with no consensus on a single answer

### Sample conflictingqa_220ec09fbb2c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2, d4, d1
- **Claim**: The retrieved evidence is mixed. Some sources state that cows have four stomachs, while others clarify that they have one stomach divided into four compartments

### Sample conflictingqa_24c25ef3a801

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d4
- **Claim**: However, other sources caution that money only buys happiness conditionally: money is not a universal formula for happiness spending on material possessions or to keep up with others does not produce the same results as spending on experiences, giving to others using money as a tool to enhance relationships

### Sample conflictingqa_34fef928d452

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Scaled feet and rigid first two digits in the hand, consistent with frequent ground walking, but with a third digit capable of movement suggesting tree climbing was also possible

### Sample conflictingqa_37ebad668bb7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The retrieved evidence is mixed. Some sources argue that data is always required for machine learning

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d4
- **Claim**: Real Christmas trees are generally considered more sustainable than artificial trees, though the comparison depends on usage patterns. The Joe Gardener Show argues that real trees are better for the environment because they are grown on farms using sustainable practices, sequester carbon can be recycled into mulch or compost after the holiday season — unlike artificial trees made from nonrenewable plastics and metals that mostly end up in landfills

### Sample conflictingqa_4317242e485c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The overall weight of evidence suggests that trophy hunting is not universally beneficial and that its impacts are highly context-dependent, with some populations and communities benefiting while others are harmed

### Sample conflictingqa_517b918aa677

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The Supreme Court has also ruled that the Pledge of Allegiance (without the words 'under God') is constitutional, that teaching the Bible in a literary or historical context is permissible that student-led prayer at school events is generally allowed, though school personnel cannot lead or encourage it

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d2
- **Claim**: The Kidney Disease Improving Global Outcomes (KDIGO) guidelines recommend bicarbonate supplementation only when serum bicarbonate falls below 18 mEq/L the evidence is considered insufficient to fully resolve the question of whether bicarbonate supplementation generally slows CKD progression

### Sample conflictingqa_6d7279b0a8ed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2, d3
- **Claim**: The Gutenberg Bible was not the first book ever printed with movable type — that distinction belongs to Jikji, a Korean Buddhist text printed in 1377 — but it was the first major book printed in Europe using mass-produced metal movable type the first to be commercially successful in the West

### Sample conflictingqa_73560dfab1ae

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Overall, the evidence does not establish vitamin C as a reliable or safe treatment for common cold symptoms for the general population

### Sample conflictingqa_9261438d6ee2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: d2
- **Claim**: A 2010 book by Victoria Braithwaite argues that fish have the brain structures necessary to feel pain, but that their pain perception is likely very different from humans' fish do not have the dense brain folds that are associated with human consciousness

### Sample conflictingqa_962d8f5d5574

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d4, d5
- **Claim**: The conflict_type is 'Conflicting opinions or research outcomes' because d2 and d4 assert a universal ability, d1 and d3 highlight significant information gaps d5 hedges the claim, reflecting methodological divergence in representativeness and reliability

### Sample conflictingqa_9ceca2645833

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1
- **Claim**: A third perspective holds that affirmative action is only vulnerable to reverse discrimination claims when it is applied in a way that is not narrowly tailored to address the specific historical discrimination it is meant to remedy that well-designed programs are not reverse discrimination per se

### Sample conflictingqa_a1e36a8db854

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: The conflict between these regulatory bodies reflects methodological differences in study scope and dataset size, with the EPA's conclusion based on 15 carcinogenicity studies and the IARC's based on eight

### Sample conflictingqa_a25014a5c5b5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2, d3
- **Claim**: Some plants can survive in complete darkness for extended periods, while others cannot survive without light at all. The retrieved evidence is mixed: high-credibility sources (Epic Gardening, RHS, Martha Stewart) confirm that some plants can tolerate low light or artificial light and that plants need light for photosynthesis and growth, while YouTube videos (The Sorry Girls) demonstrate that some plants can survive in zero light for up to 30 days

### Sample conflictingqa_a507c5b61631

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: However, other factors—such as the release of methane from ocean sediments or organic-rich permafrost—were also likely involved, as the PETM onset coincides with a mercury low and atmospheric CO2 rose by thousands of gigatons

### Sample conflictingqa_a507c5b61631

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Further research is needed to fully resolve the causes of this pivotal climate event

### Sample conflictingqa_b2524e4883ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Overall, while meteor showers do pose some risk — particularly to spacecraft and potentially large structures on Earth — the evidence suggests that the likelihood of a catastrophic, human-harming event is extremely low

### Sample conflictingqa_b9854bd5a19e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: d3
- **Claim**: Much of the apparent shrinkage in recent millennia is attributable to a decline in average body size rather than brain size itself

### Sample conflictingqa_bd2e652cd64d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: However, manual toothbrushes remain acceptable if brushing technique is excellent and the user brushes for the full recommended two minutes , so the choice between the two ultimately depends on your individual brushing habits and needs

### Sample conflictingqa_bf6e9339f2c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: d2, d3
- **Claim**: However, it is worth noting that unfortified nutritional yeast is not a great source of B vitamins and may not meet complete protein needs on its own , so fortified varieties are generally preferred

### Sample conflictingqa_e2e2361dc28b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: d2
- **Claim**: However, some philosophers argue that Gettier's counterexamples rely on a coherentist assumption that is itself problematic others contend that the very idea of a justified false belief is incoherent, since justification seems to entail at least partial truth

### Sample conflictingqa_e2e2361dc28b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The debate remains unresolved, with no consensus on whether a justified belief can ever be false

### Sample conflictingqa_f4811561af0c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The retrieved evidence supports both Dutch discovery and competing claims. The Dutch were among the first Europeans to discover Australia, with Willem Janszoon being the first recorded European to land on Australian soil in 1606. However, the first European to sight the mainland is disputed, as Portuguese explorer Willem van Veen is also known to have sighted the coast in 1605. The Dutch East India Company (VOC) did establish a presence on the continent, with Dirk Hartog landing on an island in 1616 and leaving behind a pewter plate as a record, but they never formally claimed the land

### Sample conflictingqa_fcdb9e210683

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The conflict is methodological, philosophical interpretive, reflecting differing opinions on what constitutes a living organism and how to define the tree of life

### Sample freshqa_150b9ed24a07

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1ef881d26e2e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: She is described as the 'genius child' who overcame significant obstacles to achieve this milestone her work focused on the dynamics and geometry of Riemann surfaces

### Sample freshqa_28e155139ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The latest stable Android version is Android 16

### Sample freshqa_3227ea6c6056

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: The latest major.NET version is.NET 7.0 (five-point-twenty), released on May 28, 2024. It is a part of the.NET 7.0 SDK and Runtime available for download from the official Microsoft download center

### Sample freshqa_3847b5cb9b42

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: It has caused over 1 million casualties and resulted in Ukraine's population declining by over 10 million people — roughly a quarter of its total population

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_5574b1447bdb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1, d5
- **Supporting Docs Found**: d2
- **Claim**: The earliest documented case of COVID-19 was identified in Wuhan, China, in early December 2019 , with a molecular clock analysis suggesting the virus had circulated undetected for months before that

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7dce5d575302

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Andrés Iniesta (2012)

### Sample freshqa_7e63fcff2dea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Laika

### Sample freshqa_7f1c3aae61a5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Bayonne, New Jersey

### Sample freshqa_97f3c1fe1fd4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Eminem

### Sample freshqa_a50d0f1f3cdf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4, d1
- **Claim**: The high-credibility Wikipedia article on her death and state funeral further specifies the location and circumstances of her death, noting she was succeeded immediately by her eldest child, Charles III

### Sample freshqa_c3f10dc1632d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5
- **Claim**: Jiangsu

### Sample freshqa_c479e83e408f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_c4976f8629cb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: It is a monitor lizard native to Komodo Island and the surrounding Lesser Sunda Islands of Indonesia the largest specimen ever recorded was 10.2 feet long and weighed 365 pounds

### Sample freshqa_c7ac9d61059a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3, d1, d5
- **Supporting Docs Found**: None
- **Claim**: Older or lower-accuracy sources citing prices around $42,000 or 44,000 are likely referring to earlier model years or different trim levels have been superseded by the more recent 2026 data

### Sample freshqa_d4d59d75b206

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3, d5
- **Claim**: 12

### Sample freshqa_f6ac249bdf53

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d4, d5
- **Claim**: Ta-Nehisi Coates

### Sample hotpotqa_0083

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0100

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d8
- **Supporting Docs Found**: d9
- **Claim**: It is a member of MedStar Health, the not-for-profit network serves as a teaching hospital for Georgetown University School of Medicine

### Sample hotpotqa_0153

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0155

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3
- **Claim**: The 'I'm Lovin' It' jingle was written by Pharrell Williams, with the official credit going to Pusha T

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0b3b6452e224

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: d4
- **Claim**: The statue's face was modeled after Bartholdi's mother its internal framework was designed by engineer Gustave Eiffel

### Sample qacc_0bd7153f19ad

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: From Italy, the Allies prepared for the D-Day invasion of Normandy (France) in June 1944, marking the beginning of the end of Germany's hold on Europe

### Sample qacc_17dc360eea55

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: 26 episodes — Season 5 of The Curse of Oak Island has 26 episodes, as confirmed by the TV Guide episode guide

### Sample qacc_17dc360eea55

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The high-quality TV Guide page for the show also lists all 26 episodes for Season 5, directly answering the query

### Sample qacc_1a764b8b6cf5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence does not justify a score above threshold, so the final answer cannot be generated

### Sample qacc_1b95727cc286

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: For a potential TV series adaptation, these same characters could be recast, with suggestions including Damson Idris for Ace, Algee Smith for Mitch Joey BADA$$ for Rico

### Sample qacc_2ed872eb1114

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The movie is set in the fictional town of 'the Bathtub,' a marshland community on the edge of the ocean draws strong connections to New Orleans and Southeast Louisiana culture

### Sample qacc_37fdedfe4478

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Joanna Cotten

### Sample qacc_403a59870dc2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The Rams have won three Super Bowls in their franchise history: Super Bowl XXXIV on January 30, 2000, when they defeated the Tennessee Titans 23-16 as the St. Louis Rams; Super Bowl LVI on February 13, 2022, when they defeated the Cincinnati Bengals 23-20 as the Los Angeles Rams; and Super Bowl XIV on January 20, 1980, when they lost to the Pittsburgh Steelers 31-19, which is not a win but is included for completeness

### Sample qacc_41c44ecfd0f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d4, d5
- **Claim**: Peyer's patches and lacteals are both lymphatic vessels found in the small intestine, but they are distinct structures with different functions: Peyer's patches are organized lymphoid follicles in the ileum involved in immune surveillance, while lacteals are central blunt-ended lymphatic capillaries in intestinal villi responsible for absorbing dietary lipids

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_9404250d756f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Sheppey Island

### Sample qacc_a1c73439eca9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_a44267c115d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: By weight, this means organs with high intracellular water content like the brain (73%) and muscles (79%) contribute disproportionately to the total body water, even though they may not be the largest in mass

### Sample qacc_a6a2f8b1f0b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: 245

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: Carter Pewterschmidt (played by Seth MacFarlane)

### Sample qacc_c675e6cd8ad6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The music for Disney's Robin Hood was composed by George Bruns, with songs written by Roger Miller. Roger Miller's songs include 'Whistle-Stop', 'Oo-de-lally' 'Love', which are all featured in the 1973 Disney animated Robin Hood film. Elton Hayes also composed music for the 1952 Disney Robin Hood, but his work was for a different film

### Sample qacc_d00b0063e747

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Additionally, Josh and Anna Duggar have seven children, making the total count of grandchildren at least 35

### Sample qacc_e6d89fce1b8e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Teddy Altman married Owen Hunt on Grey's Anatomy

### Sample qacc_ea469c846404

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Rangers last participated in the Champions League during the 2022–23 season

### Sample qacc_ecbc6adf8a48

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4, d3
- **Claim**: There is no scholarly consensus on a single definitive date, partly because the author does not identify himself or provide specific historical references partly due to ongoing debate over the relative chronology of Johannine literature

### Sample qacc_ecd3d9c0ca11

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: Vernon Wells played Wez in The Road Warrior

### Sample qacc_f10c7ad4bb81

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d4, d5
- **Claim**: An initialism is a type of abbreviation that is pronounced letter by letter, rather than as a word. Examples include DNA, RT-PCR FBI. While both acronyms and initialisms are formed from the first letters of a phrase, the key difference is that acronyms are pronounced as a word and initialisms are pronounced letter by letter

### Sample qacc_ff2cb00f4c03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d4, d5
- **Claim**: 7

### Sample situatedqa_geo_00f6ee9a6705

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The top 10 cities in Florida host 100% of The Villages' U.S. locations, with Sumter leading at 66, followed by Lake at 13 Marion at 4

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The number of casualties in World War II is approximately 70–80 million, with a central estimate of 73–74 million. The high-end estimate of 80 million comes from a detailed analysis by the International Centre for Microbial Ecology that combines demographic and archaeological data to estimate total deaths across all of Europe, while the low-end estimate of 50 million comes from a 2012 scholarly article arguing that many Soviet deaths were not recorded. The most commonly cited central estimate of 50 million is from a 1969 textbook by historian A. J. P. Taylor, but this is considered too low by modern scholars. The U.S. National World War II Museum's research starter on worldwide deaths in World War II provides a breakdown by country, with the Soviet Union alone accounting for an estimated 8.8–10.7 million military deaths and 10.4–13.3 million civilians, while the U.S. lost 418,500 personnel

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: 16

### Sample situatedqa_geo_5f5c20228969

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: 6 fronts

### Sample situatedqa_geo_eec0ebc0dd88

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The Boston Tea Party in 1773 turned tea into a politically charged drink in colonial America, causing Americans to switch to coffee as a patriotic alternative; coffee completely eclipsed hot tea in 1865 when Union soldiers returned from the Civil War and took their coffee rations with them

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d5, d1
- **Supporting Docs Found**: d3
- **Claim**: It is a part of the Federal Reserve System, consisting of seven members from the Board of Governors and five presidents from the Regional Federal Reserve Banks meets regularly to decide on interest rates and the money supply

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: New South Wales last won the State of Origin series in 2026, when they claimed Game 1 22-20 at home in Sydney

### Sample situatedqa_temp_35156c8be377

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: These differences reflect distinct measurement perspectives—GDP versus GDP per capita versus total GDP—each valid and contextually applicable, yet collectively they present a nuanced picture of Africa's economic landscape

### Sample situatedqa_temp_3781ac7b3ead

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_3df0e6082901

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Florida Gators

### Sample situatedqa_temp_4fbe76f974ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_5a59faf24972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4, d5
- **Claim**: The latest Android version is Android 16, released on June 10, 2025. It was first released on Google Pixel phones and has since rolled out to Samsung Galaxy and other devices. Android 16 does not have a dessert nickname like earlier versions; its internal codename is 'Baklava'

### Sample situatedqa_temp_657c130afab6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: 1980

### Sample situatedqa_temp_6a8aed3e3d5f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_7a7a7f43e575

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This is because many communities were named after George Washington, the first U.S. president, as a patriotic gesture or to assert legitimacy during the 18th and 19th centuries

### Sample situatedqa_temp_8fbdda192a13

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: September 1967

### Sample situatedqa_temp_a8c74bced99a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This was the first major battle between the Muslims and the Quraysh the Muslims emerged victorious with the help of Allah

### Sample situatedqa_temp_b6c75b99a2f4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: d3, d1
- **Claim**: The longest wavelengths in the visible spectrum are approximately 700 nanometers, which is the range of red light

### Sample situatedqa_temp_d0579ca3907c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The retrieved evidence supports 1274 BC as the most accurate date for the Battle of Kadesh

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d4, d5
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_dc4d0c4e24ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_f3b3face1b6b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Instant win tokens can also be found on certain items or earned digitally customers can request a game piece without a purchase at playatmcd.com

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample wikirevision_0001

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Twitter is currently known as X

### Sample wikirevision_0065

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2, d3
- **Claim**: Facebook, Inc. was the official name of the company from 2005 to 2021, when it rebranded as Meta Platforms, Inc. to reflect a strategic shift toward developing the metaverse

### Sample wikirevision_0067

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The current Ballon d'Or winner is Ousmane Dembélé, who won the 69th Ballon d'Or ceremony in 2025, marking his first win. This is confirmed across multiple sources, including the newer Wikipedia revision of the Ballon d'Or article, which supersedes an older version that had described him as the 2024 winner. The 2025 ceremony recognized the best footballers of the 2024–25 season, with Dembélé earning the award for his outstanding performance

### Sample wikirevision_0072

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2, d3
- **Claim**: This rebranding took place in April 2023 when Twitter merged with X Holdings and ceased to be an independent company, becoming part of X Corp

### Sample wikirevision_0082

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The current Ballon d'Or winner is Ousmane Dembélé (France), who won the 69th Ballon d'Or ceremony in 2025. This is confirmed across multiple sources, including the high-credibility Wikipedia articles on both the Ballon d'Or and the 2025 ceremony. The award recognized the best footballers of the 2024–25 season, with Dembélé earning his first-ever Ballon d'Or

### Sample wikirevision_0089

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Calcutta is officially called Kolkata. The city officially changed its name from Calcutta to Kolkata in 2001 this change is recognized across all sources. The current official name is Kolkata it has been so since 2001 when the city changed its name from Calcutta

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Australia

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current President of Germany is Frank-Walter Steinmeier, who has been in office since 19 March 2017. He is the Federal President of the Federal Republic of Germany, serving a five-year term that is renewable once consecutively resides at Bellevue Palace in Berlin

### Sample wikirevision_0134

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The current Prime Minister of Australia is Anthony Albanese, who has been in office since 23 May 2022. He is the 31st person to hold the role since the office was created in 1901 is appointed by the Governor-General on the advice of the monarch

### Sample wikirevision_0135

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by the newer Wikipedia revision of the Wimbledon Championships article, which supersedes an older version (September 2025) that had described the 2026 tournament as upcoming

### Sample wikirevision_0135

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Multiple sources note that the 2026 Wimbledon Championships took place from 29 June to 12 July 2026, further corroborating that the 2026 edition is the current tournament

### Sample wikirevision_0151

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Australia

### Sample wikirevision_0152

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The latest Ballon d'Or winner is Ousmane Dembélé (France), who won the 69th Ballon d'Or ceremony in 2025, marking his first win. This is confirmed across multiple sources, including the high-credibility Wikipedia articles on the 2025 Ballon d'Or and the Ballon d'Or award itself

### Sample wikirevision_0172

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: This victory confirmed his status as the current champion, superseding any earlier results from the 2026 French Open tournament


================================================================================

*Report generated by CATS v2.0*
