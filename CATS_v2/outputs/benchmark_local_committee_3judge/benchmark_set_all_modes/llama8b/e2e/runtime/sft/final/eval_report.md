# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 127 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.974 (over 736 samples)

**GR F1** *(used in CATS)*: 0.984

**Behavior Adherence**: 0.806 (over 609 applicable samples)

**Factual Grounding**: 0.849 (over 609 applicable samples)

**Single-Truth Recall**: 0.705 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.836

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
- **Behavior**: 0.935 (n=154)
- **Grounding**: 0.870 (n=154)
- **Recall**: 0.831 (n=154)
- **CATS**: 0.908

### Type 2: Complementary Info

- **Samples**: 221 (44 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.964
- **GR F1** *(used in CATS)*: 0.977
- **Behavior**: 0.870 (n=177)
- **Grounding**: 0.849 (n=177)
- **Recall**: 0.590 (n=156)
- **CATS**: 0.822

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.982
- **GR F1** *(used in CATS)*: 0.989
- **Behavior**: 0.656 (n=96)
- **Grounding**: 0.852 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.833

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.968
- **GR F1** *(used in CATS)*: 0.982
- **Behavior**: 0.710 (n=145)
- **Grounding**: 0.843 (n=145)
- **Recall**: 0.689 (n=140)
- **CATS**: 0.806

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.919
- **GR F1** *(used in CATS)*: 0.958
- **Behavior**: 0.730 (n=37)
- **Grounding**: 0.779 (n=37)
- **Recall**: 0.730 (n=37)
- **CATS**: 0.799


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2086

**Average Cost per Decision**: $0.000000


### Per-Model Cost Breakdown

#### local/qwen3.5-397b-a17b

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 0
- **Average Cost per Request**: $0.000000

#### local/mistral-small-4

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 0
- **Average Cost per Request**: $0.000000

#### local/deepseek-r1-distill-32b

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 0
- **Average Cost per Request**: $0.000000


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample conflictingqa_08757808c000

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: However, people with prehypertension or hypertension need to exercise caution and may need to modify their training, such as lifting lighter weights or avoiding Valsalva maneuvers

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
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: The retrieved evidence is mixed. Some sources state that cows have four stomachs, while others clarify that they have one stomach divided into four compartments

### Sample conflictingqa_34fef928d452

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Scaled feet and rigid first two digits in the hand, consistent with frequent ground walking, but with a third digit capable of movement suggesting tree climbing was also possible

### Sample conflictingqa_37ebad668bb7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2, d3
- **Claim**: The retrieved evidence is mixed. Some sources argue that data is always required for machine learning

### Sample conflictingqa_3afd7f725cb4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: The retrieved evidence presents both sides of the debate: some sources argue that audiobooks are legitimate reading because they provide full access to every word and facilitate empathy, while others argue that listening rather than visually reading diminishes the experience

### Sample conflictingqa_3bd13d25098b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The Moon has long been considered geologically inactive, with volcanism ceasing about 3 billion years ago and a core dynamo shutting down between 2.5 and 1 billion years ago

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: One Tree Planted similarly concludes that real trees are more eco-friendly because they can have negligible or even negative emissions when recycled or kept in a pot, whereas artificial trees produce up to 40 kg of greenhouse gas emissions and are mostly made from plastic and metal

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

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: The Kidney Disease Improving Global Outcomes (KDIGO) guidelines recommend bicarbonate supplementation only when serum bicarbonate falls below 18 mEq/L the evidence is considered insufficient to fully resolve the question of whether bicarbonate supplementation generally slows CKD progression

### Sample conflictingqa_56fd6bf22253

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: d3
- **Claim**: Regrowth is more common in children under five years old and those treated with antibiotics post-surgery in most cases it does not cause clinically significant nasal obstruction

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Most products that claim to repair split ends can only temporarily mask the damage, smoothing the cuticle or adding weight to frayed ends, but these effects do not last beyond the next shampoo

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
- **Supporting Docs Found**: d2, d5, d3, d4, d1
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

### Sample conflictingqa_b9854bd5a19e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: Much of the apparent shrinkage in recent millennia is attributable to a decline in average body size rather than brain size itself

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The evidence is mixed. Some studies suggest paper straws produce fewer emissions when reused but more when incinerated or sent to landfill compared to plastic straws

### Sample conflictingqa_bf6e9339f2c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: d2
- **Claim**: However, it is worth noting that unfortified nutritional yeast is not a great source of B vitamins and may not meet complete protein needs on its own , so fortified varieties are generally preferred

### Sample conflictingqa_e2e2361dc28b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: However, some philosophers argue that Gettier's counterexamples rely on a coherentist assumption that is itself problematic others contend that the very idea of a justified false belief is incoherent, since justification seems to entail at least partial truth

### Sample conflictingqa_e2e2361dc28b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
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

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: August 16, 1977

### Sample freshqa_122352ad92e3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: It begins before sundown on Wednesday, April 1 ends after nightfall on Thursday, April 9, according to the Hebrew calendar

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
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

### Sample freshqa_4a98eba95e97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: Joe Biden visited Russia as U.S. President on June 16, 2021, when he met with Russian President Vladimir Putin at Villa La Grange in Geneva, Switzerland

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_5d6e5db69928

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: This finding superseded earlier records of ~1 million-year-old DNA from a mammoth tooth and environmental sediments in Antarctica, confirming Greenland as the source of the world's oldest DNA

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The latest Academy Award for Best Picture was won by *Anora* (2025), directed by Sean Baker, at the 98th Academy Awards

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by Deadline's gallery of every Best Picture winner, which lists Anora as the 2025 winner, superseding earlier reports of 'One Battle After Another' (2026) as the most recent winner

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The 98th Academy Awards also included wins for Paul Thomas Anderson's 'One Battle After Another' in multiple categories including Best Director and Best Adapted Screenplay, further corroborating the validity of the 2025 Best Picture winner

### Sample freshqa_7dce5d575302

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
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
- **Supporting Docs Found**: d4, d3
- **Claim**: Bayonne, New Jersey

### Sample freshqa_8f302f0bfe82

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The latest Nebula Award for Best Novel was won by *The Dragonfly Gambit* (2025), as listed on the official Nebula Awards page

### Sample freshqa_8f302f0bfe82

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: This supersedes older information from multiple sources that had previously identified *The Saint of Bright Doors* (2023) as the winner , as the 2025 award is the most recent and authoritative data

### Sample freshqa_97f3c1fe1fd4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Eminem

### Sample freshqa_a41257e9d6f6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: His death occurred two years after Minsky and Papert published their 1969 book arguing against the perceptron's viability, which caused funding for perceptron research to dry up for over a decade

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The 2023–24 season was a disappointing year for the Raptors, who had hoped to build on their 2019 championship but struggled to find consistency and ultimately fell short of the playoffs

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

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: It is a monitor lizard native to Komodo Island and the surrounding Lesser Sunda Islands of Indonesia the largest specimen ever recorded was 10.2 feet long and weighed 365 pounds

### Sample freshqa_d4d59d75b206

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5
- **Claim**: 12

### Sample freshqa_f6ac249bdf53

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2, d3
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

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0155

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The 'I'm Lovin' It' jingle was written by Pharrell Williams, with the official credit going to Pusha T

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0b3b6452e224

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: Bartholdi's design was commissioned by French historian Édouard de Laboulaye, who proposed the monument to commemorate the upcoming centennial of U.S. independence and the abolition of slavery

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

### Sample qacc_2243f17ccc38

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: d2
- **Claim**: Leeds United's FA Cup history also includes a runner-up finish in 1969-70 and a semi-final appearance in 1976-77, but the 1972 victory remains their only FA Cup win

### Sample qacc_2ed872eb1114

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The film's production designer, Tamara Deverell, confirmed that the team shot on location in Louisiana, using real environments and local buildings whenever possible, including in the abandoned gas station in Montegut where the production team was based

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
- **Supporting Docs Found**: d4, d5, d2, d3
- **Claim**: Peyer's patches and lacteals are both lymphatic vessels found in the small intestine, but they are distinct structures with different functions: Peyer's patches are organized lymphoid follicles in the ileum involved in immune surveillance, while lacteals are central blunt-ended lymphatic capillaries in intestinal villi responsible for absorbing dietary lipids

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_6837d86d03ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: This order, with Prince William being the Duke of Cambridge and Prince George being the Duke of Cambridge's eldest son

### Sample qacc_6f8df54650a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by the Henley Passport Index 2025, which ranks the U.S. passport 12th in the world for travel freedom , up from 13th in 2024

### Sample qacc_720d8b210357

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: These origins are not fixed: in metazoans like humans, a specific consensus DNA sequence has not been identified origin selection appears to be mainly epigenetic , varying substantially between species and even within chromosomes

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: The Phoenix location without contradicting this fact, noting that the restaurant's iconic golden arches and nostalgic design elements remain a recognizable landmark in mid-20th-century American dining

### Sample qacc_9404250d756f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Sheppey Island

### Sample qacc_9fbf28f5786f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5, d3
- **Claim**: The International Space Station (ISS) did not have a single launch date; it was constructed in stages, with the first module being Zarya, launched on November 20, 1998 the first crew arriving in 2000

### Sample qacc_a1c73439eca9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_a6a2f8b1f0b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2, d3
- **Claim**: 245

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Carter Pewterschmidt (played by Seth MacFarlane)

### Sample qacc_c675e6cd8ad6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3
- **Claim**: The music for Disney's Robin Hood was composed by George Bruns, with songs written by Roger Miller. Roger Miller's songs include 'Whistle-Stop', 'Oo-de-lally' 'Love', which are all featured in the 1973 Disney animated Robin Hood film. Elton Hayes also composed music for the 1952 Disney Robin Hood, but his work was for a different film

### Sample qacc_d00b0063e747

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Additionally, Josh and Anna Duggar have seven children, making the total count of grandchildren at least 35

### Sample qacc_d03e85bdc95a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: George Bernard Shaw

### Sample qacc_e064a7a717ed

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The movie's production spanned these diverse locations to bring Jeannette Walls' nomadic childhood to life on screen

### Sample qacc_e6d89fce1b8e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Teddy Altman married Owen Hunt on Grey's Anatomy

### Sample qacc_ea469c846404

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: Rangers last participated in the Champions League during the 2022–23 season

### Sample qacc_ecbc6adf8a48

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4
- **Claim**: There is no scholarly consensus on a single definitive date, partly because the author does not identify himself or provide specific historical references partly due to ongoing debate over the relative chronology of Johannine literature

### Sample qacc_ecd3d9c0ca11

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: Vernon Wells played Wez in The Road Warrior

### Sample qacc_ff2cb00f4c03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3
- **Claim**: 7

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

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5
- **Claim**: California grizzly bear (Ursus arctos californicus)

### Sample situatedqa_geo_eec0ebc0dd88

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_f2031e426cee

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: The present Law Minister of India is Shri Kiren Rijiju, who is also the Minister of Parliamentary Affairs, as per the official Law Ministry of India website

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d4, d5
- **Claim**: This was followed by the drafting and ratification of the U.S. Constitution in 1787–1788, which superseded the Articles of Confederation and established the current federal system

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d2, d1
- **Supporting Docs Found**: d3
- **Claim**: It is a part of the Federal Reserve System, consisting of seven members from the Board of Governors and five presidents from the Regional Federal Reserve Banks meets regularly to decide on interest rates and the money supply

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: New South Wales last won the State of Origin series in 2026, when they claimed Game 1 22-20 at home in Sydney

### Sample situatedqa_temp_32d33d503f69

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d2, d5
- **Supporting Docs Found**: d1
- **Claim**: This date, while d1 references Season 6 of the original series coming in 2025 multiple docs note the TV series has completed its current season , making the movie the primary focus of the query

### Sample situatedqa_temp_3781ac7b3ead

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
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
- **Supporting Docs Found**: d4
- **Claim**: The latest Android version is Android 16, released on June 10, 2025. It was first released on Google Pixel phones and has since rolled out to Samsung Galaxy and other devices. Android 16 does not have a dessert nickname like earlier versions; its internal codename is 'Baklava'

### Sample situatedqa_temp_657c130afab6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2, d3
- **Claim**: 1980

### Sample situatedqa_temp_6a683d0d04f0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: This is consistent with the broader pattern that major keys with increasing numbers of sharps correspond to higher numeric values, where C Major (0 sharps) is followed by G Major (1 sharp), D Major (2 sharps), A Major (3 sharps) finally B Major (5 sharps)

### Sample situatedqa_temp_6a8aed3e3d5f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: Todd Monken

### Sample situatedqa_temp_7a7a7f43e575

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This is because many communities were named after George Washington, the first U.S. president, as a patriotic gesture or to assert legitimacy during the 18th and 19th centuries

### Sample situatedqa_temp_7f07f7a6c607

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5, d4, d1
- **Claim**: Mohamed Salah

### Sample situatedqa_temp_8fbdda192a13

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: September 1967

### Sample situatedqa_temp_9049c197a579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: This figure is corroborated by the U.S. Energy Information Administration, which also notes that California's state gasoline excise tax is the highest in the United States at $0.60 per gallon

### Sample situatedqa_temp_9a91e43d8136

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: The 2018 India–South Africa Test series was marked by several close contests, with India ultimately winning 2–1

### Sample situatedqa_temp_9a91e43d8136

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The series itself was decided by a 63-run victory for India in the third and final Test at the Wanderers in Johannesburg

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_a8c74bced99a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This was the first major battle between the Muslims and the Quraysh the Muslims emerged victorious with the help of Allah

### Sample situatedqa_temp_b3ad3248b7d4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: The empire reached its peak under the 9th Sapa Inca (emperor) Pachacuti and his descendants, who expanded it beyond Cusco and into the Sacred Valley

### Sample situatedqa_temp_b3ad3248b7d4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: d3
- **Claim**: The empire was conquered by the Spanish in 1533, when Francisco Pizarro captured and killed the last Sapa Inca, Atahualpa the remnants retreated to the Neo-Inca State in Vilcabamba

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: This is the first of the two Queen Elizabeth-class carriers, with her sister ship HMS Prince of Wales (R09) following in 2019

### Sample situatedqa_temp_bdc0853f9a23

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2
- **Claim**: This ranking is based on 23 assessment criteria across three domains — domestic and international conflict, societal safety and security militarization — and is part of the twelfth edition of the GPI

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: 164

### Sample situatedqa_temp_d0579ca3907c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: The retrieved evidence supports 1274 BC as the most accurate date for the Battle of Kadesh

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2, d3
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_dc4d0c4e24ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample trust_align_059

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This casting is consistently confirmed across multiple sources, with no contradictions

### Sample wikirevision_0001

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3
- **Claim**: Twitter is currently known as X

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
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
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Multiple sources note that the 2026 Wimbledon Championships took place from 29 June to 12 July 2026, further corroborating that the 2026 edition is the current tournament

### Sample wikirevision_0151

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Australia

### Sample wikirevision_0172

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: This victory confirmed his status as the current champion, superseding any earlier results from the 2026 French Open tournament


================================================================================

*Report generated by CATS v2.0*
