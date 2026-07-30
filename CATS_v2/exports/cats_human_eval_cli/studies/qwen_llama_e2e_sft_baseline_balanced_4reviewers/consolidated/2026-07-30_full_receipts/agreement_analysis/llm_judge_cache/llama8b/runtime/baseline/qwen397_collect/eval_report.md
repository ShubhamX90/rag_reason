# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 2 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.826 (over 736 samples)

**GR F1** *(used in CATS)*: 0.904

**Behavior Adherence**: 0.753 (over 734 applicable samples)

**Factual Grounding**: 0.905 (over 734 applicable samples)

**Single-Truth Recall**: 0.710 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.818

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.904
- **Precision**: 0.828
- **Recall**: 0.997
- **Accuracy**: 0.826
- TP=606, FP=126, FN=2, TN=2

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.500
- **Abstain Recall**: 0.016
- **Abstain F1**: 0.030
- **Specificity**: 0.997
- Abstain TP=2, FP=2, FN=126, TN=606


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (2 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.739
- **GR F1** *(used in CATS)*: 0.848
- **Behavior**: 0.904 (n=209)
- **Grounding**: 0.876 (n=209)
- **Recall**: 0.818 (n=154)
- **CATS**: 0.862

### Type 2: Complementary Info

- **Samples**: 221
- **GR Accuracy**: 0.792
- **GR F1** *(used in CATS)*: 0.884
- **Behavior**: 0.891 (n=221)
- **Grounding**: 0.907 (n=221)
- **Recall**: 0.571 (n=156)
- **CATS**: 0.813

### Type 3: Conflicting Opinions

- **Samples**: 109
- **GR Accuracy**: 0.881
- **GR F1** *(used in CATS)*: 0.937
- **Behavior**: 0.229 (n=109)
- **Grounding**: 0.911 (n=109)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.692

### Type 4: Outdated Info

- **Samples**: 158
- **GR Accuracy**: 0.918
- **GR F1** *(used in CATS)*: 0.957
- **Behavior**: 0.753 (n=158)
- **Grounding**: 0.945 (n=158)
- **Recall**: 0.779 (n=140)
- **CATS**: 0.858

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.973
- **GR F1** *(used in CATS)*: 0.986
- **Behavior**: 0.622 (n=37)
- **Grounding**: 0.872 (n=37)
- **Recall**: 0.595 (n=37)
- **CATS**: 0.769


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2178

**Average Cost per Decision**: $0.000000


### Per-Model Cost Breakdown

#### local/qwen3.5-397b-a17b

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 2179
- **Average Cost per Request**: $0.000000


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample conflictingqa_1e9c6cd8e30b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4
- **Claim**: While it has some health benefits, the environmental concerns surrounding its production outweigh these benefits

### Sample conflictingqa_2500bfd10608

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d5, d1, d3
- **Claim**: The safety of fluoride in drinking water is a topic of ongoing debate, with both positive and negative views on its effects on human health

### Sample conflictingqa_264e97231363

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3, d5, d2
- **Claim**: The hair does not turn green from chlorine in swimming pools

### Sample conflictingqa_264e97231363

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: To prevent this, it's recommended to wet your hair before going into the pool, apply a leave-in conditioner wash your hair immediately after getting out of the pool

### Sample conflictingqa_264e97231363

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: If your hair is already green, you can try at-home remedies such as rinsing with tomato juice, ketchup lemon juice

### Sample conflictingqa_2c0ea18839df

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d2
- **Claim**: Further research is needed to fully understand the mechanisms involved

### Sample conflictingqa_3601f7480501

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Requiring employees to use all their vacation days and unplugging while on vacation can create the best company culture and a partnership between employees and employers

### Sample conflictingqa_3601f7480501

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: Unlimited PTO can also attract and retain top talent, but there is a risk of policy abuse and conflict among employees

### Sample conflictingqa_39fe5c441657

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: However, the nature of the astral body and the brain activity associated with these experiences are still not fully understood

### Sample conflictingqa_4317242e485c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d5, d1, d3
- **Claim**: The final answer is:
Trophy hunting can have both positive and negative impacts on conservation, with some arguing that it can generate revenue for conservation and others criticizing it as a form of animal cruelty

### Sample conflictingqa_4317242e485c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d4, d5
- **Claim**: While some studies suggest that trophy hunting can be an effective way to conserve wildlife populations and generate income for local communities, others raise concerns about the ethics of killing animals for sport and the potential negative impacts on animal welfare

### Sample conflictingqa_517b918aa677

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The Constitution does not explicitly prohibit or allow school prayer, but it does guarantee the right to free exercise of religion

### Sample conflictingqa_517b918aa677

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d2
- **Claim**: The Supreme Court has ruled that school-led or endorsed prayers are unconstitutional, but it has also allowed for certain forms of religious expression, such as teaching the Bible and the Pledge of Allegiance

### Sample conflictingqa_63fde268aa8c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Ultimately, the relationship between the mind and body is a complex and multifaceted issue that is still debated among philosophers and scientists

### Sample conflictingqa_6fe31cd2ef65

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: To practice rolling the R, try placing the tip of your tongue against the roof of your mouth, right behind your upper front teeth then vibrating it rapidly

### Sample conflictingqa_6fe31cd2ef65

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: You can also try repeating the /d/ or /t/ sound, which places your tongue in the correct position then gradually make the sound lighter and quicker

### Sample conflictingqa_747727772a30

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3, d5
- **Claim**: Bees can fly in the rain, but they prefer to keep their wings dry and their flight paths clear

### Sample conflictingqa_747727772a30

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: They will only fly in the rain if they absolutely must, such as when they need to defend their hive or find emergency food

### Sample conflictingqa_9251c7a33ec5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The neutering/spaying of a pet can have both positive and negative effects on their health

### Sample conflictingqa_9251c7a33ec5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: d2
- **Claim**: However, the benefits of neutering, such as reduced pet overpopulation and improved behavior, are also significant

### Sample conflictingqa_9ceca2645833

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: However, the debate around affirmative action is multifaceted its implications and consequences are still being discussed

### Sample conflictingqa_a507c5b61631

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: However, other sources of carbon, such as the release of carbon from methane-rich ocean sediments or organic-rich permafrost, may have also contributed to the event

### Sample conflictingqa_b9854bd5a19e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The relationship between brain size and intelligence is complex and not fully understood

### Sample conflictingqa_bac0f4d62f96

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: Some meteorites, like those from the Leonids, may have originated in comets

### Sample conflictingqa_bdee100fa8e5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: The newspapers at the time sensationalized the panic to discredit radio as a source of news,

### Sample conflictingqa_ece626a6cba9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3, d4, d5
- **Supporting Docs Found**: d1
- **Claim**: The exact cause of the Black Death remains a topic of debate among historians and scientists

### Sample conflictingqa_f22b389be1d6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: However, it's essential to note that running shoes may also have benefits, such as providing arch support and increasing overall stiffness

### Sample conflictingqa_f4693bea2c31

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3, d2
- **Claim**: The final answer is: Emojis are a form of written language, but not a replacement for it

### Sample conflictingqa_f777a43ba278

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: d1
- **Claim**: However, the evidence is not conclusive more research is needed to fully understand the relationship between yerba mate consumption and cancer risk

### Sample conflictingqa_f7fec8c0688b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d1, d2
- **Claim**: The incident remains unexplained and continues to be a topic of debate and speculation

### Sample conflictingqa_f8da23d84ecc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: The Brontosaurus and Apatosaurus were initially considered to be the same species, but a 2015 study found them to be distinct

### Sample conflictingqa_f8da23d84ecc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: However, the rules of paleontology state that the first name has precedence Apatosaurus was the first name given to the species

### Sample conflictingqa_f8da23d84ecc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: Therefore, Brontosaurus and Apatosaurus are the same species, with Apatosaurus taking precedence

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3, d4, d5
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_2e51f51132ee

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: However, the exact date is not specified in the provided documents

### Sample freshqa_7e63fcff2dea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d5, d1, d3
- **Claim**: The first animal to land on the moon was not mentioned in the provided documents

### Sample freshqa_7e63fcff2dea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: The first animals to circle the Moon were two Russian tortoises

### Sample freshqa_c3f10dc1632d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d5
- **Claim**: The province bordering Shanghai to the north is Zhejiang

### Sample freshqa_c479e83e408f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The final answer is 69

### Sample freshqa_c4976f8629cb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The heaviest reptile in the world is the green anaconda, with the largest specimen ever recorded weighing 550 pounds

### Sample freshqa_c7315f8b3029

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1
- **Claim**: The final answer is:
OpenAI released GPT-5.5 on May 5, 2026

### Sample freshqa_c7ac9d61059a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The base price of the 2026 Tesla Model Y Premium All-Wheel Drive is $43,380

### Sample freshqa_ddd643091cbe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_e1a40e76ba15

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Therefore, the final answer is that the game was suspended 21 minutes after Hamlin suffered cardiac arrest

### Sample hotpotqa_0083

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the birth year of Sébastien Buemi is not mentioned in the retrieved documents

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3, d5, d2
- **Supporting Docs Found**: None
- **Claim**: The mother said "I never should" when Jackie, the daughter, fell pregnant with Rosie, her grandchild

### Sample qacc_1a764b8b6cf5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: They shared certain characteristics such as being close companions of the Prophet Muhammad

### Sample qacc_3c1297608017

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d5, d1, d3
- **Claim**: The gesture of crossing fingers for luck has a long history, with multiple theories on its origins

### Sample qacc_3c1297608017

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d4
- **Claim**: Two main theories are presented: one related to pre-Christian pagan beliefs and the other to early Christianity

### Sample qacc_3c1297608017

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4, d5
- **Supporting Docs Found**: d3
- **Claim**: The gesture has evolved over time, with the original two-person version becoming a solo gesture

### Sample qacc_41c44ecfd0f0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Peyer's patches are organized lymphoid nodules that appear as oval or round lymphoid follicles extending from the mucosa layer of the ileum into the submucosa layer they play a role in filtering foreign particles and antigens from the intestines

### Sample qacc_ac5341df9a83

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: The UN gets troops for military actions from UN Member States, which contribute troops to the UN for specific peacekeeping operations

### Sample qacc_bfbb5f55a63f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The Old Spice commercial featuring a coach is played by Isaiah Mustafa

### Sample qacc_d8b24beb2f90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: It shows the company's total assets, liabilities equity, which are the three components of the accounting equation

### Sample qacc_f10c7ad4bb81

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d5, d1, d3
- **Claim**: Initialisms are abbreviations formed from the initial letters of a phrase are pronounced as individual letters

### Sample qacc_fbe562911999

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d5, d1, d3
- **Claim**: The Speaker of the Lok Sabha is placed at Sl

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The minimum age to drive a transport vehicle is 16 years of age

### Sample situatedqa_geo_864fc7e36714

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The largest numbers of immigrants are found in certain states, such as California, Texas, Florida, New York New Jersey

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d3
- **Claim**: The FOMC is composed of members from the Board of Governors and Federal Reserve Banks it meets regularly to influence money supply and interest rates through open market operations

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: The FOMC's decisions have significant effects on the economy, including inflation and employment levels

### Sample situatedqa_temp_14a587def215

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1
- **Claim**: The Carolina Hurricanes last made the playoffs in 2026

### Sample situatedqa_temp_43c0aaae9828

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d5, d1, d3
- **Claim**: The song "Pursue / All I Need Is You" is performed by Hillsong Worship

### Sample situatedqa_temp_9049c197a579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: This discrepancy may be due to the fact that the total tax includes other fees and surcharges, but the exact breakdown is not provided in the documents

### Sample situatedqa_temp_9a91e43d8136

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The highest runs scored by India in the 2018 India-South Africa test series

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Emily Fields is 31 years old

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: It was formally declared operational in 2020

### Sample situatedqa_temp_bdc0853f9a23

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1
- **Claim**: The 2018 report ranks 163 independent states and territories according to their level of peacefulness, with India being ranked 136th

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Given the conflicting information, it is possible that the number of members may have changed since the documents were written that the documents may be referring to different time periods or contexts

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5
- **Claim**: Therefore, the most accurate answer based on the available evidence is 164, but with the caveat that the number may have changed since the documents were written

### Sample trust_align_004

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The St. Louis Cardinals have spring training in

### Sample trust_align_009

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: The Black Death started in the UK in 1350

### Sample trust_align_015

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: The high school in Japan starts in grade 7

### Sample trust_align_019

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: Michigan lost to Michigan State in 2017

### Sample trust_align_029

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The Sacramento Kings play at the Golden 1 Center in downtown Sacramento, California

### Sample trust_align_034

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: The Declaration also includes the right to life, liberty security of person, as well as freedom from slavery and servitude

### Sample trust_align_043

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: In other forms, tick boxes are used to confirm understanding of statements or to select options

### Sample trust_align_050

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4, d5, d1, d3
- **Supporting Docs Found**: None
- **Claim**: The dates of death of persons that held the position Bishop of Carlisle are 5 April 1478 , 1535 , 2 December 1745 , 18 January 1804 5 January 1943

### Sample trust_align_062

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2, d5
- **Claim**: The voice of Snowball in Stuart Little is Nathan Lane

### Sample trust_align_063

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d5
- **Claim**: The magnetic north pole moves due to changes in the Earth's magnetic field

### Sample trust_align_068

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Given the conflicting information, the best course of action is not explicitly stated in the documents

### Sample trust_align_068

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: However, based on the general advice to switch, it is reasonable to suggest that switching to door 2 may be a good option

### Sample trust_align_070

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The character present in the work "Nineteen Eighty-Four" is Big Brother

### Sample trust_align_071

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The final answer is:
CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample trust_align_072

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: The capital gains tax rate on real estate in Canada is 6% , however, another source states it is 15%

### Sample trust_align_086

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d1, d3, d5
- **Claim**: However, the exact reasons for the price differences between stations may vary depending on the specific location and circumstances

### Sample trust_align_087

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: The song "it's a thin line between love and hate" could not be identified in the retrieved documents

### Sample trust_align_089

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the retrieved documents do not provide information about the current captain

### Sample trust_align_091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The person who has won the second most NBA championships is Red Auerbach, who is mentioned in d2 as being surpassed by Phil Jackson for most championships all-time by an NBA coach

### Sample trust_align_095

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The liver's ability to regenerate may be a factor in its ability to recover from damage, but the permanence of scarring caused by excessive alcohol consumption is still unclear

### Sample trust_align_105

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: The ski jumpers do not sustain injury when landing due to their specialized technique and equipment, which allows them to absorb the impact of landing on a challenging slope

### Sample trust_align_106

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d4, d5
- **Claim**: The tendons and ligaments in human anatomy play important roles in supporting and stabilizing various parts of the body, including joints and organs

### Sample trust_align_106

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d4, d5
- **Claim**: They enable movement, maintain position prevent over-extension or dislocation

### Sample trust_align_109

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d4, d2
- **Claim**: Explosions kill by causing a rapid release of energy, which can lead to destruction of buildings and harm or death to people

### Sample trust_align_110

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: The song "Band on the Run" was released in 1973

### Sample trust_align_114

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The phrase "All quiet on the Western Front" originates from the novel "All Quiet on the Western Front" written in 1927

### Sample trust_align_117

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The books written by Thomas Middleton are Timon of Athens and possibly other plays, but the exact titles are not specified in the provided documents

### Sample trust_align_125

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: Credit card reward systems work by giving users money back or points on certain purchases, with the value of points increasing with higher spending

### Sample trust_align_125

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5
- **Claim**: The amount of cashback or points earned can vary depending on the credit card and the type of purchase

### Sample trust_align_129

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d4, d5
- **Claim**: The actors who played Michael Myers in different films are Don Shanks (2007), Tony Moran (1978), James Jude Courtney (2018) Dick Warlock (unspecified film)

### Sample trust_align_130

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: The current leader of the opposition in Uganda is not explicitly stated in the provided documents, but Nathan Nandala Mafabi is mentioned as the seventh Leader of Opposition

### Sample trust_align_130

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: However, it is unclear if he is still in office

### Sample trust_align_130

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample trust_align_136

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d1
- **Claim**: The founding of New Zealand as a country occurred in 1840

### Sample trust_align_145

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The final answer is:
CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample trust_align_150

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The bass player for the Eagles is Timothy B. Schmit

### Sample trust_align_152

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The Battle of San Jacinto started and ended on June 22, 1911

### Sample trust_align_154

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: The Commonwealth Games were first hosted by India in 2006

### Sample trust_align_155

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: The films that Heather Graham is a member of the cast are "Single White Female" (1992) and "Ecstasy" (2011)

### Sample trust_align_167

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: The release date of "Harry Potter and the Deathly Hallows Part 1" was 21 July 2007

### Sample trust_align_171

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The new Star Wars movie was released in 2017

### Sample trust_align_173

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3, d5, d2
- **Claim**: The final answer is: Natural sugars found in whole foods, such as fruits and vegetables, are generally good for health because they contain antioxidants, vitamins, minerals fiber

### Sample trust_align_174

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The answer is CANNOT ANSWER, INSUFFICIENT EVIDENCE, as the retrieved documents do not provide enough information to determine who has been on the Sports Illustrated cover the most

### Sample trust_align_175

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4, d5, d1, d3
- **Supporting Docs Found**: None
- **Claim**: However, the exact reason for the temperature difference between the North and South Poles is not explicitly stated in the provided documents

### Sample trust_align_189

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: However, there is no information provided about its use in computer casings

### Sample trust_align_191

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: d5
- **Claim**: The war of the Spanish succession likely ended before 1668

### Sample trust_align_192

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d4
- **Claim**: The final answer is:
Pat Metheny Group, Joshua Redman Quartet, Trio 99 – 00

### Sample trust_align_193

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4, d5
- **Supporting Docs Found**: None
- **Claim**: However, it is worth noting that the exact reason why blue cheese is safe to eat with mould on is not explicitly stated in the provided documents


================================================================================

*Report generated by CATS v2.0*
