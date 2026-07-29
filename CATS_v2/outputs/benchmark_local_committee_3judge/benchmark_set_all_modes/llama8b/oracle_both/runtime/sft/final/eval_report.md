# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 122 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.974 (over 736 samples)

**GR F1** *(used in CATS)*: 0.984

**Behavior Adherence**: 0.819 (over 614 applicable samples)

**Factual Grounding**: 0.897 (over 614 applicable samples)

**Single-Truth Recall**: 0.740 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.860

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.984
- **Precision**: 0.990
- **Recall**: 0.979
- **Accuracy**: 0.974
- TP=595, FP=6, FN=13, TN=122

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.904
- **Abstain Recall**: 0.953
- **Abstain F1**: 0.928
- **Specificity**: 0.979
- Abstain TP=122, FP=13, FN=6, TN=595


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (52 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.976
- **GR F1** *(used in CATS)*: 0.984
- **Behavior**: 0.962 (n=159)
- **Grounding**: 0.945 (n=159)
- **Recall**: 0.860 (n=154)
- **CATS**: 0.938

### Type 2: Complementary Info

- **Samples**: 221 (44 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.968
- **GR F1** *(used in CATS)*: 0.980
- **Behavior**: 0.887 (n=177)
- **Grounding**: 0.848 (n=177)
- **Recall**: 0.644 (n=156)
- **CATS**: 0.840

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.982
- **GR F1** *(used in CATS)*: 0.989
- **Behavior**: 0.708 (n=96)
- **Grounding**: 0.913 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.870

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.987
- **GR F1** *(used in CATS)*: 0.993
- **Behavior**: 0.683 (n=145)
- **Grounding**: 0.920 (n=145)
- **Recall**: 0.718 (n=140)
- **CATS**: 0.828

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.919
- **GR F1** *(used in CATS)*: 0.958
- **Behavior**: 0.703 (n=37)
- **Grounding**: 0.793 (n=37)
- **Recall**: 0.730 (n=37)
- **CATS**: 0.796


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2133

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

### Sample conflictingqa_0717d0e62f3b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: In the European Union, fashion designs can receive protection for up to three or five years under the Creative Designs Directive, though this protection applies only to new designs and does not cover all fashion designs broadly

### Sample conflictingqa_0875b5f3262a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: d1
- **Claim**: Historians and legal scholars note that the 1957 ruling was a significant victory for First Amendment protections, as Judge Horn explicitly stated that 'no two persons think alike; we were all made from the same mold but in different patterns' and that reducing vocabulary to innocuous euphemisms would undermine freedom of press and speech , a ruling that has been celebrated as a classic defense of artistic expression

### Sample conflictingqa_0dba017da71c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that while it is the largest single living organism, it is not the heaviest; a colony of aspen trees called Pando in Utah is estimated to be heavier, though it covers less area

### Sample conflictingqa_11c5ef7c4545

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The debate is further nuanced by the observation that nearly three quarters of entrepreneurs start their businesses in pursuit of opportunity rather than necessity, suggesting that a desire to pursue opportunity is at least as important as any innate talent or skill

### Sample conflictingqa_24c25ef3a801

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: A YouTube video by philosopher David Cooper discussing the book 'The Price of Happiness' adds a further dimension, noting that money's impact is not solely about the amount spent, but also about whether one's values are aligned with what they buy that many people in wealthy countries report feeling empty or unfulfilled despite their material comfort

### Sample conflictingqa_24fa0020a521

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: In addition, megadoses of certain vitamins can be toxic in children, so any supplement should be chosen with a pediatrician's guidance

### Sample conflictingqa_264e97231363

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: The retrieved evidence is mixed and does not definitively confirm or deny that hair can turn green from chlorine in swimming pools

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5, d1, d4
- **Claim**: Real Christmas trees are generally considered more sustainable than artificial trees across multiple dimensions. A 2009 Ellipsos study cited by The Joe Gardener Show found that an artificial tree must be reused for about 20 Christmases before its climate impact equals that of a real tree even then, artificial trees remain in landfills indefinitely whereas real trees are recycled into mulch or compost. Real trees are grown on farms using sustainable agricultural practices, sequester carbon dioxide produce oxygen; in contrast, artificial trees are made from non-renewable plastics and metals, require fossil fuels for manufacturing and transport cannot be recycled — with most ending up in landfills after only 5–7 years. However, the sustainability comparison is conditional: if an artificial tree is used for 20 years or more, its cumulative carbon footprint can approach that of a real tree buying a potted tree for annual reuse is considered the most sustainable option of all

### Sample conflictingqa_517b918aa677

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: Weisman, 1992; Santa Fe, 2000), while the First Amendment also protects students' individual right to pray alone or in voluntary groups (Lamb's Chapel v. Center Moriches Union Free School District, 1993; Mergens, 1990). Recent Trump administration guidance further clarified that schools must allow students and staff to act in accordance with their faith while maintaining neutrality that public schools may not sponsor or organize compulsory prayer, though students are permitted to pray privately or with like-minded participants (February 2026 guidance)

### Sample conflictingqa_5233eab573e5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d3, d4, d1
- **Claim**: All available evidence consistently supports the same conclusion: captivity surpasses wild populations

### Sample conflictingqa_613a0093714b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: d3
- **Claim**: Multiple sources note that the etymology remains unknown and that the phrase may have originated from Norse mythology, medieval superstitions poor sanitation , with Jonathan Swift's 1738 writing providing some of the earliest recorded evidence but not a definitive origin

### Sample conflictingqa_63fde268aa8c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This methodological and interpretive divergence between dualist philosophy and embodied science produces directly opposing conclusions on the same question, reflecting a genuine conflict that cannot be resolved through reasoning from the available evidence alone

### Sample conflictingqa_734859937e46

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: This repeal of FCC privacy protections applies to major US ISPs like Verizon, Comcast AT&T allows them to share users' browsing history with advertisers as long as they anonymize the data

### Sample conflictingqa_734859937e46

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: d1
- **Claim**: However, some states are pushing back against this federal standard — for example, Maine passed a law requiring ISPs to obtain individual express permission before selling personal data California's Consumer Privacy Act gives residents the right to opt out of data sales — so the answer is not uniform across all US states

### Sample conflictingqa_73560dfab1ae

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Vitamin C is generally not a substitute for other treatments the evidence is not strong enough to recommend high-dose vitamin C as a primary treatment for colds

### Sample conflictingqa_7cf85109a70d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1, d2
- **Claim**: All available documents consistently support the same conclusion: bronze is more durable than brass, with no contradictions across sources

### Sample conflictingqa_80baf25496cd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: The conflict is methodological: d2 and d4 present research and philosophical arguments linking multiculturalism to social fragmentation, while d3 and d1 present research and conceptual frameworks suggesting it does not hinder unity or even facilitates it, with d5 offering a conditional perspective

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The question of whether bird calls are unique to each individual is a genuinely contested one, with some sources suggesting that certain calls are shared across individuals within a species , while others argue that bio-acoustic analysis can distinguish calls between individual birds

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: However, the evidence does not establish a universal answer: some calls are shared within a species while some are unique to individuals the degree of individuality varies by call type and species

### Sample conflictingqa_9261438d6ee2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: The evidence is mixed

### Sample conflictingqa_bd2e652cd64d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3
- **Claim**: However, manual toothbrushes can still be effective if used correctly and are generally more affordable, so the best choice ultimately depends on your personal brushing habits and budget

### Sample conflictingqa_dc6f972e8447

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: However, the retrieved evidence does not establish a definitive, comprehensive measure of how easily cryptocurrencies can be manipulated in general

### Sample conflictingqa_f3ba8599e370

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: The conflict arises from differing definitions of'religion' and varying interpretations of yoga's nature, with no definitive resolution in the available evidence

### Sample conflictingqa_f4811561af0c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: The question of who actually discovered Australia first depends on how one defines 'discovery'—whether it refers to the first sighting, the first recorded European landing the first complete mapping of the coastline

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: Epidemiological studies show mixed results—some large studies found odds ratios as high as 34.6 for certain populations , while others found no significant association —and the NIH review of mate consumption and cancer noted that results varied greatly across studies

### Sample conflictingqa_f7fec8c0688b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3, d5, d1, d2
- **Claim**: The Phoenix Lights incident on March 13, 1997, was officially explained as military flares by the U.S. Department of Defense, though witnesses and some sources, including former Arizona Governor Fife Symington, disputed this explanation, describing the formation as a massive, silent, unmanmade craft

### Sample freshqa_0293a11bd364

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This ranking is based on the most recent data from Ethnologue and Visual Capitalist (2025) and is consistent with the expectation that English would lead with the largest total number of speakers, followed by Mandarin Chinese then Hindi

### Sample freshqa_047057d22309

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: August 16, 1977

### Sample freshqa_122352ad92e3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1ef881d26e2e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: The Fields Medal has had two female recipients so far Mirzakhani's 2014 award was the first

### Sample freshqa_31ad09b9cd22

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Samara Joy

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_6f42c128eb6c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Harry Maguire has never won the Ballon d'Or; no confirmed Ballon d'Or win exists in his career per available evidence

### Sample freshqa_7f1c3aae61a5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_97f3c1fe1fd4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Eminem

### Sample freshqa_a41257e9d6f6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: His death was a significant loss for the field of AI research, as funding for perceptron-related projects dried up in the following years following the publication of Minsky and Papert's 1969 book Perceptrons, which argued that simpler perceptrons had fundamental limitations

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The 2023–24 result is also corroborated by the fact that the Raptors missed the playoffs, further confirming a losing season

### Sample freshqa_b0ffe73c1789

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: This figure is consistent with the broader context of her career, which has spanned over a decade and produced multiple bestsellers

### Sample freshqa_b99c189f2222

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Jeff Bezos sold Amazon shares in late June and July 2025, but did not sell the entire company

### Sample freshqa_c479e83e408f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0083

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0100

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0153

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d6
- **Claim**: The authorship of the 'I'm Lovin' It' jingle is disputed, with multiple parties claiming credit

### Sample qacc_08cf866bcb9b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Norma Koch won the 1963 Oscar for Best Costume Design, Black-and-White for the film Whatever Happened to Baby Jane?

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0d85f1089c4a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: These are distinct regional and national designations that complement each other, reflecting the campaign's scope across different levels of government and geography

### Sample qacc_132167e66120

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_15ffab2466f7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d5, d1
- **Claim**: These venues for different productions, with no contradictions

### Sample qacc_2ed872eb1114

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The film's connection to Louisiana and the Bathtub setting, though the fictional nature of the Bathtub means it is not a physical location where filming occurred in the same way that Isle de Jean Charles and Montegut were

### Sample qacc_51c89636151e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Manwë

### Sample qacc_67b35f41ba84

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1, d4
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_720d8b210357

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: These origins are not fixed in number and are subject to epigenetic regulation, varying in number between species and even within the same organism depending on cell type and conditions

### Sample qacc_8daf80e943fa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2
- **Claim**: 5.88 trillion miles

### Sample qacc_9fbf28f5786f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: d3
- **Claim**: Multiple modules were launched over the next several years, with the station being assembled in orbit in stages

### Sample qacc_a44267c115d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: By weight, the average adult human is approximately 60% water, with the brain and heart at 73%, lungs at 83% muscles and kidneys at 79% , consistent with the broader distribution across all body fluids

### Sample qacc_b198a514fff8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: New Mexico was admitted to the Union as the 47th state on January 6, 1912

### Sample qacc_ba7aaa9b36c8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: No one was injured in the fire the following Christmas, White House staff and their children gathered again to celebrate the holidays and the Hoovers presented the children with toy fire trucks as gifts

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Seth MacFarlane

### Sample qacc_d7c6682b5335

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: The Pokémon Company's official Trading Card Game (TCG) was first released in Japan in 1996 by Media Factory, with the first USA cards following in 1999

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: XXXTENTACION

### Sample qacc_e064a7a717ed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4
- **Claim**: The Glass Castle was released on August 11, 2017 grossed $22 million in North America, receiving mixed reviews from critics with praise for its cast performances

### Sample qacc_eb7c676e133e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: December 19, 1972

### Sample qacc_ec5b0067c29a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1, d2
- **Supporting Docs Found**: d5
- **Claim**: It is located on the grounds of the U.S. Naval Observatory, a historic building dating to 1830

### Sample qacc_f1776add7672

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: 3–7

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_5f5c20228969

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: 6 fronts

### Sample situatedqa_geo_7222d6123c27

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This shift was formally announced by King George V during the Delhi Durbar and was completed by 1931, when New Delhi was inaugurated as the new capital

### Sample situatedqa_geo_897e47478bbc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This figure comes directly from the official Census 2011 data, which is the most authoritative source for administrative boundaries in the country

### Sample situatedqa_geo_9f6c5f849b40

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: It is worth noting that by metropolitan area population, Tokyo's ranking jumps to first at 10.3 million, while Jakarta's metropolitan area ranks second at 9.4 million, reflecting the difference between city proper and urban agglomeration population measures

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: d2
- **Claim**: Eisenhower was the first U.S. president to send military advisers, with Kennedy later expanding the program significantly

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: However, this cultural shift was gradual — and not universal: Southern Americans continued to drink sweet tea and other immigrant groups retained their own tea traditions, meaning coffee never fully eclipsed tea everywhere

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The U.S. government issued coffee as part of Civil War rations starting in 1861, which further accelerated coffee's rise and eventually led to coffee completely eclipsing hot tea by 1865

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: In the United States, environmental policy can be set at both the federal and state levels of government. The federal government sets broad national policies through agencies such as the Environmental Protection Agency (EPA) and the Council on Environmental Quality, while state governments play a significant role in implementing and supplementing these policies through their own agencies and programs. At the federal level, the EPA is responsible for setting and enforcing pollution control standards, while the Council on Environmental Quality coordinates overall environmental policy across all federal programs. State governments, in turn, have their own environmental agencies and programs that work in conjunction with the EPA to address issues such as climate change, conservation waste management

### Sample situatedqa_temp_14f29fe7ff15

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: British General Sir William Howe's army of about 16,000 troops defeated the Continental Army of about 15,000 in the vicinity of Chadds Ford, Pennsylvania, near Philadelphia

### Sample situatedqa_temp_35156c8be377

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: The conflict arises because GDP and GDP per capita are different metrics — GDP measures the absolute size of the economy while GDP per capita measures wealth per person — and the answer depends on which metric is used

### Sample situatedqa_temp_45cbc4dc3e28

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Justice Rajput had previously served as a senior puisne judge of the SHC and was confirmed as a permanent judge on August 31, 2013

### Sample situatedqa_temp_5e6b388e88aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d2
- **Claim**: 2022 (most recent in the retrieved evidence)

### Sample situatedqa_temp_657c130afab6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d5, d2
- **Claim**: 1980

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5
- **Claim**: Todd Monken

### Sample situatedqa_temp_7d5f559b313d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d5, d2
- **Supporting Docs Found**: d4
- **Claim**: All sources are from high-quality government agencies (BEA/Bureau of Economic Analysis) and reputable economic trackers, providing a complete and up-to-date picture of US GDP

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_a8c74bced99a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This battle is considered the first major battle of Islam and is remembered for its significance in the life of the Prophet Muhammad

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Emily Fields is a fictional character and her age is not a real person's age, so this answer is about the actress, not the character

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: Her sister ship, HMS Prince of Wales (R09), was commissioned in 2019, further corroborating the 2017–2020 timeline for the class

### Sample situatedqa_temp_bdc0853f9a23

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1
- **Claim**: This ranking is further contextualized by the Institute for Economics and Peace (IEP), the producer of the GPI, which notes that 92 countries deteriorated in peacefulness in 2018 while 71 improved, resulting in India's position at 136th out of 163 states

### Sample situatedqa_temp_bdc0853f9a23

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that by 2024, India had risen to 116th position , reflecting a modest improvement over the six-year period, though the 2018 data remains the primary focus of this query

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d5, d2
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: August 20, 1989

### Sample trust_align_149

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5, d2
- **Supporting Docs Found**: None
- **Claim**: However, it is important to note that potassium iodide (a form of iodine) only protects against radioactive cesium, not all types of radiation taking too much iodine can be harmful , so dosage and timing are critical

### Sample wikirevision_0035

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The latest Prime Minister of Japan is Sanae Takaichi, who became Japan's first female Prime Minister on 21 October 2025. She is the 32nd Prime Minister under the Meiji Constitution and the most recent incumbent, having assumed office after being nominated by the National Diet and appointed by the Emperor

### Sample wikirevision_0097

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1
- **Claim**: Bangalore is officially called Bengaluru. Bengaluru is the capital and largest city of the southern Indian state of Karnataka its official name changed from Bangalore on 1 November 2014

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Australia won the 2023 Cricket World Cup, defeating India in the final on 19 November 2023 at the Narendra Modi Stadium in Ahmedabad to claim their sixth Cricket World Cup title. This is confirmed across multiple sources, with the 2023 tournament being the most recent completed edition of the competition

### Sample wikirevision_0142

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This is confirmed across multiple authoritative sources, including the newer Wikipedia revision of the Vice President of the United States article, which supersedes an older version that had described him as the 51st Vice President

### Sample wikirevision_0172

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: The latest French Open men's singles champion is Carlos Alcaraz, who defeated Jannik Sinner in the 2025 final to defend his title. He also won the 2026 French Open, defeating Alexander Zverev in the final. This makes him a three-time French Open champion, with his first title coming in 2024 when he defeated Jannik Sinner


================================================================================

*Report generated by CATS v2.0*
