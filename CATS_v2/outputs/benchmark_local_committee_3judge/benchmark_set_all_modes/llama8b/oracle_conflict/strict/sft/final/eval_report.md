# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 126 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.974 (over 736 samples)

**GR F1** *(used in CATS)*: 0.984

**Behavior Adherence**: 0.810 (over 610 applicable samples)

**Factual Grounding**: 0.844 (over 610 applicable samples)

**Single-Truth Recall**: 0.710 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.837

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.984
- **Precision**: 0.997
- **Recall**: 0.972
- **Accuracy**: 0.974
- TP=591, FP=2, FN=17, TN=126

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.881
- **Abstain Recall**: 0.984
- **Abstain F1**: 0.930
- **Specificity**: 0.972
- Abstain TP=126, FP=17, FN=2, TN=591


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (57 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.909 (n=154)
- **Grounding**: 0.889 (n=154)
- **Recall**: 0.802 (n=154)
- **CATS**: 0.900

### Type 2: Complementary Info

- **Samples**: 221 (43 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.977
- **GR F1** *(used in CATS)*: 0.986
- **Behavior**: 0.876 (n=178)
- **Grounding**: 0.833 (n=178)
- **Recall**: 0.654 (n=156)
- **CATS**: 0.837

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.972
- **GR F1** *(used in CATS)*: 0.984
- **Behavior**: 0.719 (n=96)
- **Grounding**: 0.825 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.843

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.962
- **GR F1** *(used in CATS)*: 0.979
- **Behavior**: 0.703 (n=145)
- **Grounding**: 0.855 (n=145)
- **Recall**: 0.704 (n=140)
- **CATS**: 0.810

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.865
- **GR F1** *(used in CATS)*: 0.928
- **Behavior**: 0.730 (n=37)
- **Grounding**: 0.721 (n=37)
- **Recall**: 0.595 (n=37)
- **CATS**: 0.743


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2128

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
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Outside the U.S., protection varies greatly from country to country — for example, in the European Union, the Creative Designs Directive and European Designs Directive protect new designs for up to three or five years , while in the U.S., the Digital Millennium Copyright Act (DMCA) of 1998 brought further limits on fashion design copyrighting

### Sample conflictingqa_07e290a954ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: d2
- **Claim**: St. John's Wort is generally considered safe with mild side effects but can cause serious interactions its effectiveness for severe depression remains uncertain due to limited data

### Sample conflictingqa_08757808c000

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: The temporary increase is not caused by the weight itself, but by the Valsalva maneuver and bracing that heavy lifting often involves by the body's stress response to the exercise

### Sample conflictingqa_08757808c000

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: For people with prehypertension or hypertension, the risk is not in the weight lifting itself, but in not adapting their training — such as by lifting too heavily or holding their breath — and in the failure to combine it with other lifestyle changes

### Sample conflictingqa_114c06976f62

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3
- **Claim**: The conflict arises from differing interpretations of what constitutes a'religion' — whether it must be held as a sincere, literal belief or can also be a satirical social movement

### Sample conflictingqa_1e9c6cd8e30b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3, d5
- **Claim**: Yes, palm oil is generally considered bad for the environment: it is a leading driver of deforestation, causing habitat destruction for endangered species like orangutans; it produces significant greenhouse gas emissions; and it leads to soil and water pollution. The extent of the harm varies by producer and can be mitigated through certifications like RSPO, but the basic economic dynamics of the palm oil industry incentivize deforestation and habitat destruction

### Sample conflictingqa_2500bfd10608

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: d3
- **Claim**: The conflict arises because the benefits of fluoride on dental health are well established, but the safe upper limit for acceptable intake remains uncertain, with some studies suggesting harms at levels currently used in water fluoridation programs

### Sample conflictingqa_264e97231363

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2, d5
- **Claim**: The retrieved evidence is mixed. Some sources argue that chlorine is not the direct cause of green hair and that copper from algaecide is the actual culprit, while others argue that chlorine can cause bleaching and contribute to the problem

### Sample conflictingqa_287debe5bb90

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: This methodological contradiction between incompleteness proofs and asserted factual claims reflects conflicting research outcomes regarding knowledge limits

### Sample conflictingqa_29f69e16a0c3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: These signals can be so distinct that researchers were able to use artificial flowers with different electric field shapes to demonstrate bees preferentially visiting certain patterns flowers will even modify their nectar reward based on whether a bee has previously visited another flower of the same species

### Sample conflictingqa_311fca0928d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1, d2
- **Claim**: IPv6 is not fundamentally more secure than IPv4; both protocols are equally secure because the main security mechanism (IPsec) works with either and is not unique to IPv6

### Sample conflictingqa_37ebad668bb7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3, d1, d5, d2
- **Claim**: The conflict type is complementary information: d1/d4 detail variability factors, d2 states data's general importance, d3 compares ML/DL types d5 defines data's functional role, but together they do not form a single universal answer

### Sample conflictingqa_42d60ecaee9f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: The debate is further complicated by platform-specific design differences and cultural context, which affect how emojis are interpreted and understood

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: However, critics counter that trophy hunting is often poorly regulated, that it primarily benefits wealthy hunters and local elites rather than the broader community that the revenue generated could be derived from non-consumptive tourism or other sources without the moral costs

### Sample conflictingqa_4317242e485c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3, d2
- **Claim**: The conflict is further complicated by the fact that some of the same critics argue that blanket bans on trophy hunting could actually increase animal deaths by reducing the financial incentives for conservation, though this view is itself contested

### Sample conflictingqa_4786f87b62be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The evidence does not support a definitive yes/no answer; rather, it reflects methodological and interpretive disagreements about the causes of the gap

### Sample conflictingqa_52e01830d2fe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: CLS Bank ruling and the European Patent Convention's Article 52(2)(a) exclusion of programs "as such" are notable legal landmarks in this debate patent eligibility criteria such as the machine-or-transformation test continue to evolve. Ultimately, whether software should be patented depends heavily on one's values regarding the balance between intellectual property rights and competition, as well as the specific context and jurisdiction in question

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d5
- **Claim**: The evidence is mixed and the answer depends on the stage of CKD and the dose used

### Sample conflictingqa_57190bca6f7a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: d3, d1
- **Claim**: It caused a global cooling of up to 3°C and is known as the 'Year Without a Summer'

### Sample conflictingqa_57190bca6f7a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: However, it is worth noting that the exact death toll is still a matter of debate, with some sources citing 59,000 to 90,000 deaths , while others report 118,000 to 140,000 , reflecting the difficulty of accurately quantifying the full impact of such a massive event

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: However, there are temporary solutions: products can coat the hair cuticle, add weight to frayed ends create a temporary 'glue' effect to make split ends look better, though these effects typically last only until the next shampoo

### Sample conflictingqa_734859937e46

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2, d1
- **Supporting Docs Found**: None
- **Claim**: However, some U.S. states have enacted their own laws prohibiting such sales: California's Consumer Privacy Act gives residents the right to opt out of data sharing , Maine has passed a law requiring express permission before selling personal data Oregon's H.B. 3284 similarly prohibits the sale of personal health data without affirmative consent

### Sample conflictingqa_747727772a30

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: A bee's body is adapted to sense changes in humidity and temperature, which helps them anticipate and prepare for rain by returning to the hive before it starts in some cases they will forage in light rain when food is scarce or the hive is low on stores

### Sample conflictingqa_7998e59d9b97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: However, organic farming offers distinct environmental benefits—such as producing fewer pollutants and conserving biodiversity—so the comparison is not solely about efficiency

### Sample conflictingqa_7cf85109a70d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d1
- **Claim**: However, it is worth noting that brass has its own advantages — being easier to machine and more corrosion-resistant in some contexts — so the comparison is not universal across all applications

### Sample conflictingqa_80baf25496cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: A third perspective holds that cultural differences are secondary to human spirituality and can be overcome through dialogue and shared values that monoculturalism poses its own risks through ethnocentrism and absolutist thinking

### Sample conflictingqa_9b73cb6cce52

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Some sources note that they are hermaphrodites and can reproduce on their own, producing hundreds of eggs if kept in pairs , which may be a concern for some owners

### Sample conflictingqa_a2f06d54b240

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: Stalactites form through the slow accumulation of calcite crystals from water drips underwater caves can provide the necessary environment for this process to occur

### Sample conflictingqa_a507c5b61631

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: However, other studies present a more nuanced picture—mercury levels actually dropped at the PETM onset, suggesting multiple carbon reservoirs were involved models indicate that organic carbon feedbacks played a crucial role in sustaining the event

### Sample conflictingqa_a994724a28e7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3, d1, d5, d2
- **Claim**: The conflict_type is 'Conflicting opinions or research outcomes' because these expert opinions and experimental findings on the same physiological mechanism produce directly opposing conclusions

### Sample conflictingqa_a9bed39d234d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: In practice, the Academy of Nutrition and Dietetics treats foods like celery and cucumbers as calories that must be accounted for in the daily diet, even though they are low in calories, further complicating the negative-calorie claim

### Sample conflictingqa_b9854bd5a19e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The conflicting findings reflect methodological and interpretive differences across studies the answer depends on which dataset and time frame is consulted

### Sample conflictingqa_cc71318e5853

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3, d1, d5, d2
- **Claim**: The overall picture is therefore mixed: death is not universally discussed or accepted in modern society, though attitudes may vary significantly by culture and context

### Sample conflictingqa_dd426f7706e8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The scientific community provides no empirical support for the full moon's transformative powers, further suggesting that the idea remains firmly in the realm of fiction

### Sample conflictingqa_e2e2361dc28b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: Others, however, insist that justification is tied to truth: on this view, a belief can only be justified if it is true the Gettier-style counterexamples are either blocked or do not pose a serious challenge to this requirement

### Sample conflictingqa_e2e2361dc28b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: The conflict is therefore a genuine and unresolved debate, with no consensus on whether a justified belief can ever be false

### Sample conflictingqa_f22b389be1d6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d5
- **Claim**: The debate is unresolved: some studies and experts claim barefoot running is healthier (fewer injuries, stronger foot muscles, better proprioception), while others argue shoes are healthier (stronger arch muscles, reduced heel striking)

### Sample conflictingqa_f3ba8599e370

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: The word 'yoga' in ancient texts referred to a spiritual practice of meditation and breath-control that yoga was practiced by both Hindus and Buddhists, suggesting a broader scope than Hinduism alone

### Sample conflictingqa_f43b2c51deea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: The USGS notes that while there is anecdotal evidence of animals behaving strangely before earthquakes, consistent and reliable prediction has yet to be found

### Sample conflictingqa_f4693bea2c31

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: A minority view holds that each emoji represents a distinct 'word' with its own semantic meaning, though this perspective is not widely supported among language specialists

### Sample conflictingqa_f4811561af0c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence is mixed and does not support a clear, definitive answer

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2, d5
- **Supporting Docs Found**: None
- **Claim**: On the other hand, laboratory research has shown that yerba mate exhibits cytotoxic effects on cancer cells some studies have suggested it may lower the risk of colon cancer or other cancers under certain conditions , though these findings are not conclusively confirmed in human epidemiological studies

### Sample conflictingqa_fbedb688b1d9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2, d5
- **Claim**: The question of whether Mormons are Christian is genuinely contested. Some sources argue that Mormonism is Christian because it affirms belief in Jesus Christ and the church's official website states members 'unequivocally affirm themselves to be Christians,' while others argue that Mormonism's distinctive doctrines—such as the godhead, pre-mortal life restorationist claims—represent a departure from the historic, orthodox faith and that the church's own repudiation of'many core Christian doctrines' is grounds for exclusion

### Sample conflictingqa_fcdb9e210683

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This methodological divergence reflects a deeper debate over how to define 'life' and what criteria should be used to construct the universal tree, with no consensus resolution in the available evidence

### Sample freshqa_0436c0b3a9d7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5
- **Claim**: The 2025 match between these two players is the most recent US Open women's singles final, making it the answer to who the finalists were last year

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d5
- **Claim**: August 16, 1977

### Sample freshqa_122352ad92e3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: It runs through April 9, with the first seder on April 1 after nightfall and the second seder on April 2 after nightfall

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_28e155139ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The latest stable Android version is Android 16

### Sample freshqa_2e51f51132ee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_3227ea6c6056

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: d3
- **Claim**: This is consistent across multiple sources , which supersede older information that had described.NET 6.0 as the latest

### Sample freshqa_3847b5cb9b42

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: It began with a Russian drone strike in the Sumy region of eastern Ukraine and has resulted in over 1 million casualties and a population decline of over 10 million people, roughly a quarter of Ukraine's total population

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_5d6e5db69928

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This finding superseded the previous record of one-million-year-old DNA from a mammoth tooth is currently considered the oldest DNA discovered so far

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The latest Academy Award for Best Picture was won by *Anora* (2025), directed by Sean Baker, which won at the 99th Academy Awards

### Sample freshqa_7f1c3aae61a5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_8f302f0bfe82

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The latest Nebula Award for Best Novel is 'The Dragonfly Gambit' (2025), as listed on the BookBrowse awards page

### Sample freshqa_97f3c1fe1fd4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Eminem

### Sample freshqa_a41257e9d6f6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: His death occurred two years after the publication of Minsky and Papert's 1969 book Perceptrons, which argued against the perceptron's viability, causing funding for Rosenblatt's research to dry up

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_a50d0f1f3cdf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: All available evidence uniformly confirms this date, with no conflicting reports

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

### Sample freshqa_d510972df578

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: By global box office gross, Avengers: Endgame (2020) holds the record with over $2.79 billion , but that metric includes marketing and distribution costs not reflected in production budget figures

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5
- **Claim**: 12

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: 9 minutes

### Sample freshqa_f6ac249bdf53

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2, d5
- **Claim**: Ta-Nehisi Coates

### Sample hotpotqa_0056

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d6
- **Supporting Docs Found**: d10
- **Claim**: England

### Sample hotpotqa_0063

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d8
- **Supporting Docs Found**: None
- **Claim**: Stanford University, on the other hand, is a private research university located in Stanford, California is thus not the institution implied by the query

### Sample hotpotqa_0083

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d7
- **Claim**: Sébastien Buemi (born 31 October 1988)

### Sample hotpotqa_0155

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: My Mother Said I Never Should is a play by Charlotte Keatley about the complex relationships between mothers and daughters across four generations, exploring themes of independence, growing up secrets

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The play spans 1940 to 1987 and follows the lives of Doris, Margaret, Jackie Rosie as they navigate societal changes and family dynamics

### Sample qacc_0b3b6452e224

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: Bartholdi's design was commissioned by French historian Édouard de Laboulaye, who proposed the monument to commemorate the upcoming centennial of U.S. independence and the liberation of the nation's slaves

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: After defeating the Vichy French and Axis forces in North Africa, the Allies launched a major invasion of Italy , with US troops landing near Salerno on September 9, 1943 British forces advancing up the Italian peninsula

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The official surrender of Axis forces in North Africa on May 13, 1943 was followed by the full surrender of Italy on October 13, 1943 the Allies continued their march through Italy, eventually crossing into France in June 1944

### Sample qacc_0bd7153f19ad

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Throughout the campaign, the Allies also maintained a presence in Morocco and Algeria, with French forces under de Gaulle playing a key role in liberating Paris in August 1944

### Sample qacc_15ffab2466f7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: These locations without contradiction

### Sample qacc_1a764b8b6cf5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d5
- **Claim**: These four caliphs were all close companions of Muhammad and are considered models of righteous rule in Sunni Islam, though Shia Muslims dispute the legitimacy of the first three

### Sample qacc_1b95727cc286

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The TV series adaptation currently in development will recast these roles, with suggestions including Damson Idris as Ace, Algee Smith as Mitch Joey BADA$$ as Rico , but the above names reflect the original cast of the 2002 film

### Sample qacc_2243f17ccc38

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: Leeds United's FA Cup runner-up appearances in 1969, 1970 1973 are also documented, further contextualizing the club's participation in the competition during that era

### Sample qacc_3c1297608017

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1, d2
- **Supporting Docs Found**: None
- **Claim**: A third theory points to the ichthys fish symbol of early Christianity, where touching thumbs and crossing index fingers formed a symbol meaning 'Jesus Christ, Son of God, Savior' in Greek , while others attribute the gesture to sacred geometry or Norse/Germanic pagan oath-making rituals

### Sample qacc_403a59870dc2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d2
- **Claim**: These two specific wins, with no contradicting evidence in the retrieved set

### Sample qacc_41c44ecfd0f0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Peyer's patches (also known as Peyer's patches)

### Sample qacc_531aff489b71

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: These are distinct characters, so the query is ambiguous but both actresses are relevant

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_6f8df54650a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that these figures reflect visa-free travel specifically and do not account for visa-on-arrival countries or countries requiring an Electronic System for Travel Authorization (ESTA), so the total number of destinations available to U.S. citizens is even higher

### Sample qacc_883303a2d535

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The Australian Shepherd is a medium-sized breed known for its intelligence, agility herding ability, making it a fitting choice for a sled dog character

### Sample qacc_8dd2323c077d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: The Airdrome

### Sample qacc_9404250d756f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Sheppey Island

### Sample qacc_9fbf28f5786f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: The International Space Station (ISS) did not have a single launch date, as it was constructed in stages over several years

### Sample qacc_a6a2f8b1f0b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2
- **Claim**: 245

### Sample qacc_aaf0f638e99b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: d2
- **Claim**: The sign is part of the Manual on Uniform Traffic Control Devices (MUTCD) category of Horizontal Alignment Signs, which indicate a change in roadway alignment and suggest a measured safe speed for the curve

### Sample qacc_ba7aaa9b36c8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: President Hoover and his wife watched from the West Terrace as firefighters battled the blaze, which was brought under control by approximately 10:30 pm

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Carter Pewterschmidt (played by Seth MacFarlane)

### Sample qacc_c675e6cd8ad6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The music for Disney's Robin Hood was composed by George Bruns, with songs written by Roger Miller and Floyd Huddleston. Roger Miller penned the popular ballad 'Oo-De-Lally' and the whimsical 'Whistle-Stop,' while Floyd Huddleston contributed the Academy Award-nominated 'Love.' George Bruns, a veteran Disney composer, scored the majority of the film's soundtrack, bringing the score together with Miller's songs

### Sample qacc_ce4983c8a9c5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Later, conical and platform mounds were also built, but the effigy mound tradition died out about 800 years ago

### Sample qacc_d03e85bdc95a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: George Bernard Shaw

### Sample qacc_d8b24beb2f90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The Balance Sheet

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: The tolls are denominated in Mexican pesos and are typically around 15–30 cents per mile for private cars and motorcycles , though some sources note they can run as high as $1–$2 per kilometer

### Sample qacc_ea469c846404

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: Rangers last participated in the Champions League during the 2022–23 season

### Sample qacc_ecd3d9c0ca11

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_f10c7ad4bb81

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2, d5
- **Claim**: An initialism is a type of abbreviation formed from the initial letters of a phrase or name it is pronounced letter by letter rather than as a word. Examples of initialisms include DNA, RT-PCR FBI

### Sample qacc_f1776add7672

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2, d5
- **Claim**: The ICD-10 code structure is defined as having from three to seven characters, meaning the minimum is 3 and the maximum is 7. This range is explicitly confirmed by the official NHS ICD-10 data dictionary, which states that ICD-10 codes are at least four characters in length with an alphabetic first character by the Outsource Strategies guide that describes the full range of 3–7 characters. The 3–7 character range is further contextualized by the fact that ICD-9 codes, which they replace, had a maximum of only 5 characters and were limited to around 13,000 codes, whereas ICD-10 fully expanded to at least 68,000 diagnoses

### Sample qacc_ff2cb00f4c03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2
- **Claim**: 7

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

### Sample situatedqa_geo_864fc7e36714

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3, d5
- **Claim**: The answer depends on the time period: earlier waves brought mostly Europeans (especially from Germany, Ireland the UK), while more recent immigration has shifted toward Latin America and Asia, with Mexico being the single largest source country

### Sample situatedqa_geo_8c889f8ce07a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4, d1, d2
- **Claim**: This process is consistent with the constitutional division of powers, where the executive branch negotiates treaties and the legislative branch (Senate) provides oversight and consent

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: d4
- **Claim**: These advisers were sent to help stop a military invasion from North Vietnam and to prevent the spread of communism in Southeast Asia, a key Cold War goal

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: This bear was chosen as a symbol of strength and resistance during the Bear Flag Revolt of 1846, when a group of American settlers created a flag featuring a grizzly bear to replace the Mexican flag after capturing the town of Sonoma

### Sample situatedqa_geo_eec0ebc0dd88

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_f2031e426cee

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: The present Law Minister of India is Kiren Rijiju, who is the Minister of Law and Justice, as per the official Law Ministry of India website

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5, d2
- **Supporting Docs Found**: None
- **Claim**: It is composed of twelve members — seven from the Board of Governors and five presidents from Federal Reserve Banks — and meets regularly to decide on interest rates and the money supply through open market operations

### Sample situatedqa_temp_14f29fe7ff15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: This battle was the largest single-day engagement of the American Revolution and is considered one of the most significant British victories of the war

### Sample situatedqa_temp_180f238d8296

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: Australia, India, West Indies, Pakistan, Sri Lanka, England

### Sample situatedqa_temp_1987d35f994b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: Great Basin National Park was designated as Lehman Caves National Monument on January 24, 1922, when President Warren G. Harding officially established the monument

### Sample situatedqa_temp_1987d35f994b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1
- **Claim**: Today, the park spans 77,180 acres and contains a diverse range of landscapes including caves, meadows, forests mountains, including Wheeler Peak, the second-tallest peak in Nevada at 13,063 feet

### Sample situatedqa_temp_35156c8be377

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3, d5
- **Supporting Docs Found**: None
- **Claim**: These different metrics—total GDP versus GDP per capita—complement each other in identifying which country is the 'richest' depending on whether one prioritizes absolute output or standard of living

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

### Sample situatedqa_temp_44d30ce08bf5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2, d1
- **Supporting Docs Found**: None
- **Claim**: Oklahoma is second with eight titles, including four consecutive from 2021 to 2024 , while Arizona and others have also claimed championships throughout the tournament's history

### Sample situatedqa_temp_4fbe76f974ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: 1939 (original release year)

### Sample situatedqa_temp_5a59faf24972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4, d5
- **Claim**: The latest Android version is Android 16, released on June 10, 2025. It was first released on Google Pixel phones and has since rolled out to Samsung Galaxy and other devices. A more detailed history shows that Android 15 (released September 2024) was the previous latest version, but it has since been updated to Android 16

### Sample situatedqa_temp_60095fdf39a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_657c130afab6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2, d5
- **Claim**: 1980

### Sample situatedqa_temp_6c424fc78a69

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Multiple election observers noted that the post-election environment was marred by allegations of widespread rigging from the majority of political parties, though the PTI and army both denied military interference

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d1, d5
- **Claim**: Todd Monken

### Sample situatedqa_temp_7bb7fe6c9287

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Beowulf contains kennings for both Grendel and the sea

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3
- **Claim**: 59,681 km

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_8f57822ed7d2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: LeBron James and Anthony Davis led the team to their 17th title, which was their first since 2010 and briefly tied them with the Boston Celtics for the most championships in NBA history

### Sample situatedqa_temp_901be1437bc7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Kent County, Maryland

### Sample situatedqa_temp_9049c197a579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Environmental compliance costs add up to $0.54 per gallon as of March 2025, bringing the total to nearly $1.44 per gallon

### Sample situatedqa_temp_956e1f6e518e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1, d5
- **Claim**: December 19, 1972

### Sample situatedqa_temp_9a91e43d8136

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d5
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: This is the first of the two Queen Elizabeth-class carriers, with her sister ship HMS Prince of Wales (R09) following in 2019

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: The 2020 declaration of operational status confirms that the carrier has completed its sea trials and is fully integrated into the Royal Navy's fleet

### Sample situatedqa_temp_bdc0853f9a23

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This ranking is directly reported in a table of the 2018 Global Peace Index, which ranks 163 independent states and territories according to their peacefulness

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: It is also documented that the variant forms Gerrard and Jarrard are used the name was first recorded in the Domesday Book of 1086 , with one source tracing it to the grandson of Edward the Confessor

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d1, d5
- **Claim**: 164

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2, d5
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_f3b3face1b6b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4
- **Claim**: Eligible items may also include other sandwiches, drinks possibly other menu items, though the exact list varies by location and promotion

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample wikirevision_0002

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The rebranding is also corroborated by the fact that Twitter's official website now resolves to x.com instead of twitter.com the company's social media handles have been updated to reflect the new name

### Sample wikirevision_0007

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Alphabet Inc. is a public company listed on the Nasdaq stock exchange under the ticker symbols GOOGL (Class A) and GOOG (Class C)

### Sample wikirevision_0072

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2
- **Claim**: Twitter is currently known as X

### Sample wikirevision_0090

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1, d2
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple sources, including the high-credibility Wikipedia articles on both the President of Indonesia and Prabowo Subianto himself

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Australia

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current President of Germany is Frank-Walter Steinmeier, who has been in office since 19 March 2017. He is the Federal President of the Federal Republic of Germany, serving a five-year term that is renewable once consecutively resides at Bellevue Palace in Berlin


================================================================================

*Report generated by CATS v2.0*
