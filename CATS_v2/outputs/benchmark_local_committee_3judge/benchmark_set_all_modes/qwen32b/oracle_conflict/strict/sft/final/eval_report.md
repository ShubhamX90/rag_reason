# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 128 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.955 (over 736 samples)

**GR F1** *(used in CATS)*: 0.972

**Behavior Adherence**: 0.821 (over 608 applicable samples)

**Factual Grounding**: 0.002 (over 608 applicable samples)

**Single-Truth Recall**: 0.738 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.633

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.972
- **Precision**: 1.000
- **Recall**: 0.946
- **Accuracy**: 0.955
- TP=575, FP=0, FN=33, TN=128

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.795
- **Abstain Recall**: 1.000
- **Abstain F1**: 0.886
- **Specificity**: 0.946
- Abstain TP=128, FP=33, FN=0, TN=575


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (57 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.986
- **GR F1** *(used in CATS)*: 0.990
- **Behavior**: 0.974 (n=154)
- **Grounding**: 0.000 (n=154)
- **Recall**: 0.841 (n=154)
- **CATS**: 0.701

### Type 2: Complementary Info

- **Samples**: 221 (45 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.950
- **GR F1** *(used in CATS)*: 0.968
- **Behavior**: 0.926 (n=176)
- **Grounding**: 0.006 (n=176)
- **Recall**: 0.689 (n=156)
- **CATS**: 0.647

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.945
- **GR F1** *(used in CATS)*: 0.968
- **Behavior**: 0.615 (n=96)
- **Grounding**: 0.000 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.527

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.968
- **GR F1** *(used in CATS)*: 0.982
- **Behavior**: 0.648 (n=145)
- **Grounding**: 0.000 (n=145)
- **Recall**: 0.704 (n=140)
- **CATS**: 0.584

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.784
- **GR F1** *(used in CATS)*: 0.879
- **Behavior**: 0.892 (n=37)
- **Grounding**: 0.000 (n=37)
- **Recall**: 0.649 (n=37)
- **CATS**: 0.605


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2116

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

### Sample conflictingqa_04e1627e9fc5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: Nematodes generally do not increase soil fertility directly; rather, certain beneficial species (e.g., bacterivores, fungivores) cycle nutrients by consuming microorganisms and releasing plant-available forms of nitrogen and phosphorus

### Sample conflictingqa_04e1627e9fc5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: Plant-parasitic nematodes can actually reduce fertility by damaging roots and hindering nutrient uptake, while free-living nematodes often serve as indicators of soil health and microbial activity

### Sample conflictingqa_04e1627e9fc5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Research also shows that fertilization practices shape nematode community composition, with organic matter promoting diverse, growth-supporting populations and inorganic fertilizers sometimes having mixed effects

### Sample conflictingqa_05b33f4ca156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Yes — many salamanders are poisonous to touch, as nearly all species possess toxins in their skin that can cause serious illness if ingested some (like the fire salamander) are well-documented as dangerous to humans

### Sample conflictingqa_05b33f4ca156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: However, a few specific species — such as tiger salamanders and yellow-spotted salamanders — are notable exceptions, carrying little to no risk of poisoning from handling ; the general hazard arises from the toxic secretion present on most salamander skins, which can be absorbed through cuts or mucus membranes is universally advised against

### Sample conflictingqa_060e5f26c453

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Yes — the Great Pacific Garbage Patch is larger than Texas

### Sample conflictingqa_0717d0e62f3b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Fashion designs can receive copyright protection, but only for specific elements such as graphic patterns, surface designs logos—not for the overall useful article or functional aspects

### Sample conflictingqa_0717d0e62f3b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Under U.S. law, clothing and accessories are classified as functional items and therefore generally do not qualify for broader copyright protection, though limited protection exists for certain surface appliqués and fabric prints

### Sample conflictingqa_07e290a954ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The evidence is mixed

### Sample conflictingqa_07e290a954ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Some clinical studies, particularly those reviewed by the NIH and others, suggest that St. John's wort is statistically significantly more effective than placebo for mild to moderate depression and comparable to conventional antidepressants like SSRIs

### Sample conflictingqa_07e290a954ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: However, a large NCCAM-sponsored study found it was not more effective than a placebo for moderately severe major depression user surveys rated it as distinctly less effective than most prescription antidepressants

### Sample conflictingqa_07e290a954ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: The overall picture suggests it may help with mild depression specifically, but evidence is insufficient to support its use for moderate or severe depression

### Sample conflictingqa_08757808c000

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Not directly; weight lifting may cause a temporary, acute increase in blood pressure during exertion — especially during heavy lifts — but research generally indicates that regular strength training improves long-term blood pressure control

### Sample conflictingqa_0875b5f3262a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2, d1
- **Claim**: No, Allen Ginsberg's poem "Howl" was not judged to be obscene

### Sample conflictingqa_0875b5f3262a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: A San Francisco municipal court judge found the book to be 'not obscene' because it possessed redeeming social, literary artistic value the U.S. Court of Appeals for the Ninth Circuit affirmed this ruling

### Sample conflictingqa_0875b5f3262a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: This legal precedent helped pave the way for broader protections of free speech in art

### Sample conflictingqa_0a05aabca56a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: yes

### Sample conflictingqa_0ad05303220b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Judaism is not a race; it is a religion (and, depending on perspective, also an ethnicity or nation)

### Sample conflictingqa_0c3c7b487766

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Yes, iodine supplementation can cause thyroid problems — specifically hyperthyroidism, hypothyroidism, goiter autoimmune thyroiditis — particularly in susceptible populations such as those with preexisting thyroid disease, the elderly pregnant women; excess iodine intake may also disrupt thyroid homeostasis and increase dysfunction risk even in healthy individuals

### Sample conflictingqa_0dba017da71c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Yes — the world's largest known organism is a fungus, specifically *Armillaria ostoyae* (the Humongous Fungus) living in Oregon's Malheur National Forest

### Sample conflictingqa_111bef268aa6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: The evidence presents conflicting findings

### Sample conflictingqa_111bef268aa6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Some sources say peeling removes much of the fiber and certain antioxidants, while others say peeling does not reduce vitamin levels and that nutrients are present throughout the fruit

### Sample conflictingqa_114c06976f62

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The Church of the Flying Spaghetti Monster is a subject of genuine debate regarding its legitimacy as a religion

### Sample conflictingqa_114c06976f62

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Some view it as a satirical protest against intelligent design and creationism, while others argue it meets established criteria for religious recognition

### Sample conflictingqa_11c5ef7c4545

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The retrieved evidence presents competing views

### Sample conflictingqa_11c5ef7c4545

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Some sources argue that anyone can become an entrepreneur if they are willing to learn, adapt face risks , while others contend that entrepreneurship is not for everyone because it requires specific innate traits, skills a penchant for risk that not all individuals possess

### Sample conflictingqa_11c5ef7c4545

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: The general consensus leans toward opportunity being broadly accessible, even if successful entrepreneurship is challenging and not guaranteed for all participants

### Sample conflictingqa_151865dc414b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: The retrieved evidence indicates that pulsatile tinnitus can often be successfully treated and cured in cases where the underlying cause is identified and treatable

### Sample conflictingqa_151865dc414b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: Common causes include venous sinus stenosis, arteriovenous malformations, high blood pressure certain tumors treatments such as venous sinus stenting, surgery medication to address these conditions can eliminate symptoms in many patients

### Sample conflictingqa_151865dc414b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: No universal cure exists for all cases of pulsatile tinnitus, as some causes cannot be changed or fully resolved management in these situations may focus on reducing the impact of the symptom

### Sample conflictingqa_1d741779a31e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: Yes

### Sample conflictingqa_1e9c6cd8e30b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d2
- **Claim**: Yes — palm oil production causes serious environmental harm through deforestation, biodiversity loss pollution; however, sustainably certified palm oil may mitigate these risks

### Sample conflictingqa_21f33954c8af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Ethical perspectives on dog breeding differ significantly

### Sample conflictingqa_21f33954c8af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Some argue that breeding is inherently unethical because it treats dogs as products rather than individuals deserving of autonomous existence, while others contend that breeding can be ethical if conducted responsibly with proper health screenings, lineage tracking avoidance of overbreeding

### Sample conflictingqa_21f33954c8af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: The debate remains unresolved, as critics point to widespread inherited health problems and shelter overpopulation caused by irresponsible breeding practices, whereas defenders emphasize the preservation of working breeds and the joy dogs bring to families

### Sample conflictingqa_220ec09fbb2c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: Cows technically have one stomach that is divided into four distinct compartments — the rumen, reticulum, omasum abomasum — rather than four separate stomachs

### Sample conflictingqa_237adb87065f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The retrieved evidence indicates that the Silurian period marks an important milestone for land plants—specifically the first appearance of small vascular plants such as Cooksonia on land

### Sample conflictingqa_237adb87065f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: However, one source notes that the earliest radiation of land plants actually began slightly earlier, in the Middle Ordovician, making the Silurian more of a continuation than the absolute birth of land plants

### Sample conflictingqa_237adb87065f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Additionally, the presence of the more complex lycophyte Baragwanathia in the Silurian has led some researchers to speculate that plants may have already existed during the Ordovician, further complicating the claim that the Silurian was the sole birth of land plants

### Sample conflictingqa_2395695f1604

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: The majority of scientific research does not support the idea that dairy products increase mucus production; a 2005 review found that milk consumption does not lead to mucus production or asthma occurrence a 2012 study confirmed that 'studies have not been able to provide a definitive link' between milk and increased mucus production

### Sample conflictingqa_2395695f1604

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: However, some evidence presents a dissenting view: a peer-reviewed study notes that excessive milk consumption has been associated with increased respiratory tract mucus production and asthma research also suggests that dairy may affect sensory perception or the viscosity of existing mucus without necessarily producing more of it

### Sample conflictingqa_2395695f1604

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: Overall, the evidence is mixed the prominent clinical consensus, echoed by specialist Dr. Ian Balfour-Lynn, holds that milk does not cause excess mucus and should not be avoided

### Sample conflictingqa_24c25ef3a801

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: The retrieved evidence is mixed

### Sample conflictingqa_24c25ef3a801

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Some sources argue that money can buy happiness, but usually only up to a point — for example, one study found that emotional well-being rises logarithmically with income and flattens beyond ~$75,000–$100,000/year , while another argues the logarithmic relationship persists beyond $75,000

### Sample conflictingqa_24c25ef3a801

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: Money may be effective at buying happiness when spent strategically on experiences, prosocial goods small indulgences , but these gains are conditional and do not guarantee that more money will make anyone happier

### Sample conflictingqa_24fa0020a521

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: Most healthy children do not need multivitamins if they are growing normally and eating a varied, well-balanced diet

### Sample conflictingqa_24fa0020a521

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: The American Academy of Pediatrics (AAP) does not recommend a daily multivitamin for children eating a well-rounded diet, cautioning that supplements should not be used as a substitute for balanced nutrition ; a 2024 review found that even when parents perceive their children's diets as insufficient, a multivitamin cannot compensate for poor eating habits the FDA does not regulate multivitamins in the same way as medications, so consulting a doctor before use is advised

### Sample conflictingqa_24fa0020a521

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: However, some children may benefit from targeted supplementation: the AAP specifically recommends 400 IU of vitamin D per day for exclusively breastfed infants and 600 IU for children over 1 year, as well as iron screening at 12 months multivitamins may be considered for children with restrictive diets (such as vegan), food allergies, chronic absorption conditions persistent picky eating

### Sample conflictingqa_2500bfd10608

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The question of whether fluoride in drinking water is dangerous is genuinely contested

### Sample conflictingqa_2500bfd10608

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Some research and advocacy groups argue that emerging evidence links fluoridated water to neurobehavioral problems in children and potentially lowered IQ, with one study published in JAMA Network Open finding an association between fluoride exposure during pregnancy and increased neurobehavioral issues Food & Water Watch citing mounting scientific evidence of neurotoxic effects especially among children

### Sample conflictingqa_2500bfd10608

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: On the other hand, established institutions and researchers note that fluoride is largely considered safe at the recommended drinking water concentration of 0.7 mg/L or below , with the CDC describing water fluoridation as a cost-effective public health measure experts at the Harvard School of Public Health arguing that while fluoride clearly benefits dental health, the key question is whether the benefits justify any potential risk from systemic exposure

### Sample conflictingqa_2500bfd10608

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: Additionally, high-burden evidence from peer-reviewed research cautions that chronic ingestion of excessive fluoride can lead to dental and skeletal fluorosis and that some countries are reducing fluoridation due to toxicity risk , underscoring that the danger depends heavily on concentration and population susceptibility rather than fluoridation per se

### Sample conflictingqa_264e97231363

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2, d1
- **Claim**: The retrieved evidence consistently states that chlorine is not the direct cause of green hair; rather, copper (often from algaecides or tap water) bonds with chlorine to form a film that adheres to hair proteins and turns them green

### Sample conflictingqa_287debe5bb90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Philosophers and researchers hold conflicting views on whether we can know anything beyond our minds

### Sample conflictingqa_287debe5bb90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: Some argue that genuine self-knowledge requires transcending abstract thought to engage directly with immediate experiential phenomena, while others contend that formal systems of understanding are inherently self-defeating when applied recursively

### Sample conflictingqa_287debe5bb90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Still, other perspectives explore the possibility of knowing the mind by observing external objects or events, drawing on the philosophical concept of transparency

### Sample conflictingqa_287debe5bb90

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: These competing opinions reflect ongoing debates across multiple disciplines without a single definitive resolution

### Sample conflictingqa_288cd1b45aab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Yes

### Sample conflictingqa_29f69e16a0c3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Yes — flowers communicate with bees using multiple signals: researchers found that bees can detect the shape and strength of a flower's electric field flowers respond to the sound of approaching bees by changing their nectar composition and electric potential

### Sample conflictingqa_2c0ea18839df

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Yes

### Sample conflictingqa_311fca0928d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The retrieved evidence presents conflicting views

### Sample conflictingqa_311fca0928d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Some sources argue that IPv6 is not fundamentally more secure than IPv4 — for example, IPsec can be used with both protocols the absence of NAT in IPv6 does not represent a meaningful security improvement

### Sample conflictingqa_311fca0928d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: Other sources argue that IPv6 is safer on a basic level due to its built-in IPSec support, which is not natively available in IPv4

### Sample conflictingqa_311fca0928d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The consensus across multiple sources is that neither protocol is inherently more or less secure; most security incidents result from design and implementation flaws rather than limitations in the protocols themselves

### Sample conflictingqa_34610226ee3f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: No — dinosaurs died out 65 million years ago, far exceeding the ~1 million year threshold beyond which DNA degrades entirely, making the Jurassic Park premise impossible with current technology

### Sample conflictingqa_34fef928d452

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: Yes

### Sample conflictingqa_35491baf4f4b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Yes, the Moon does have an atmosphere, though it is very thin and technically classified as an exosphere rather than a conventional atmosphere

### Sample conflictingqa_35491baf4f4b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: This exosphere consists of a tenuous mixture of gases including helium, argon, neon, ammonia, methane, carbon dioxide traces of sodium, potassium rubidium was confirmed during the Apollo missions of the 1960s and 1970s

### Sample conflictingqa_35491baf4f4b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Additionally, NASA research has shown that the Moon once possessed a thicker, transient atmosphere approximately 3 to 4 billion years ago, formed when intense volcanic eruptions released gases faster than they could escape to space

### Sample conflictingqa_3601f7480501

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: Research and opinion on unlimited vacation time are divided

### Sample conflictingqa_3601f7480501

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: Some studies, such as those cited by the University of Phoenix, found that 72% of employees believed unlimited PTO reduced stress and improved work-life balance Norwich University reported that taking vacations increases productivity, job satisfaction cardiovascular health ; similarly, research published via NIH noted that the policy is linked to lower turnover intentions and stronger organizational commitment among high performers

### Sample conflictingqa_3601f7480501

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: However, other evidence presents a paradoxical picture: a Namely survey found that employees with unlimited PTO took an average of only 13 days off per year compared to 15 days under traditional accrual systems early adopters of the policy reported that some employees took less time off than before, potentially increasing burnout rates ; moreover, one in three HR leaders surveyed by Ceridien indicated that unlimited PTO had a negative impact on employee well-being

### Sample conflictingqa_3601f7480501

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The effectiveness of the policy thus appears to depend heavily on implementation, communication individual factors — such as whether employees feel pressure to restrict their time off even when the policy is theoretically unlimited

### Sample conflictingqa_37ab7146eb9b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: Robots can be programmed to detect and respond to stimuli analogous to pain, but they cannot genuinely feel pain in the human sense

### Sample conflictingqa_37ebad668bb7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The retrieved evidence collectively indicates that data is nearly always required for Machine Learning, though the specific quantity and quality depend heavily on the algorithm, problem complexity performance requirements

### Sample conflictingqa_39fe5c441657

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The retrieved evidence presents competing views

### Sample conflictingqa_39fe5c441657

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Some sources argue that astral projection is real as an experience but not as a literal physical event, as it correlates with known brain activity patterns recorded in scientific studies

### Sample conflictingqa_39fe5c441657

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Others, particularly skeptics and critical analysts, contend that the phenomenon is better explained as a type of hallucination, lucid dream placebo effect, with no tangible physical evidence supporting literal travel

### Sample conflictingqa_39fe5c441657

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The ongoing debate reflects genuine disagreement across spiritual, mystical scientific communities, with users and practitioners also reporting a range of subjective experiences that further complicate the question

### Sample conflictingqa_3afd7f725cb4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Opinions on whether audiobooks count as real reading are divided

### Sample conflictingqa_3afd7f725cb4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Some argue that listening to an audiobook is no different from reading a physical book — the words are the same, the brain processes narratives similarly accessibility benefits for blind, visually impaired dyslexic readers make them a legitimate alternative

### Sample conflictingqa_3afd7f725cb4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Others, however, contend that listening while multitasking undermines the immersive, focused experience traditionally associated with reading that engaging with content through a different medium changes the nature of consumption in meaningful ways

### Sample conflictingqa_3bd13d25098b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: The Moon is generally considered geologically inactive compared to Earth, but recent research suggests it was recently active and may still be moving

### Sample conflictingqa_3bd13d25098b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: Multiple studies report tectonic landforms on the far side formed within the last 200 million years features like lobate scarps detected by India's Chandrayaan-1 mission indicate ongoing tectonic deformation

### Sample conflictingqa_3bd13d25098b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Beyond these larger movements, the Moon also experiences continuous geological activity from meteorite impacts and chemical interactions with the solar wind, blurring the line between 'dead' and 'active.'

### Sample conflictingqa_3c835387fe6d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Yes, the Komodo dragon is native to Australia, having evolved there approximately four million years ago before dispersing westward to Indonesia, where it currently persists

### Sample conflictingqa_3c835387fe6d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: Fossil evidence from Queensland dating back 300,000 to 4 million years ago is identical to modern Komodo dragon bones, confirming Australia as its birthplace , a finding further corroborated by a subsequent ANU study that confirmed the dragon's Australian origins through hybridisation evidence

### Sample conflictingqa_3c835387fe6d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: While the Komodo dragon is no longer found in Australia today, having gone locally extinct around 300,000 years ago , the available evidence consistently supports its Australian nativity

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d2
- **Claim**: Real Christmas trees are generally considered more sustainable than artificial ones

### Sample conflictingqa_3f3c3399259a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The relationship between fish oil and heart disease risk is genuinely contested in the evidence

### Sample conflictingqa_3f3c3399259a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Some clinical trials and authoritative reviews conclude that fish oil supplements do not significantly reduce the risk of heart attack or stroke and may even increase the risk of atrial fibrillation at high doses , while other research—particularly earlier observational studies and some systematic reviews—report potential benefits for cardiovascular events, hypertension heart failure

### Sample conflictingqa_3f3c3399259a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: The consensus is that dietary omega-3 from fish is associated with heart benefits, but whether fish oil supplements confer the same protection remains uncertain and debated

### Sample conflictingqa_411445406724

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Cycads are commonly associated with the Mesozoic Era and were indeed abundant during that period , but they are generally not considered to have dominated the entire Mesozoic plant kingdom

### Sample conflictingqa_42d60ecaee9f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Not unanimously — some researchers argue emojis are a new language, while others argue they are not a new language but rather an evolution of older visual systems or merely a writing convention

### Sample conflictingqa_4317242e485c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: Yes — the IUCN's 2016 report, cited by the NRA, concludes that trophy hunting is the most effective way to save wildlife populations, particularly those in decline, by providing revenue, incentives funding for anti-poaching efforts

### Sample conflictingqa_4317242e485c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: However, critics and some researchers argue that trophy hunting is morally inappropriate, financially benefits only a small elite that photo-tourism or alternative revenue models may be more sustainable long-term

### Sample conflictingqa_4786f87b62be

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The retrieved evidence presents competing views

### Sample conflictingqa_4786f87b62be

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Some researchers argue that the gender wage gap is largely a parenting pay gap or is explained by different career choices, while others argue that the gap is real and reflects systemic discrimination

### Sample conflictingqa_517b918aa677

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The constitutional status of prayer in U.S. public schools is nuanced and contested

### Sample conflictingqa_517b918aa677

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The Supreme Court has ruled that school-led or endorsed prayer — including participation by faculty and staff — is unconstitutional under the Establishment Clause, while the First Amendment also protects the rights of individual students to pray privately and quietly without coercion

### Sample conflictingqa_517b918aa677

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Federal guidance from the U.S. Department of Education has further clarified that schools must maintain religious neutrality, permitting students and employees to express their faith on equal terms as long as it does not disrupt the educational environment or constitute official school sponsorship

### Sample conflictingqa_52181cd092aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: Yes — the Great Pacific Garbage Patch (GPGP) is roughly the same size as Texas, though estimates vary depending on methodology and threshold concentrations used to define its boundaries

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Software patents are a subject of genuine debate

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Some argue that software should not be patented at all

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The evidence on whether bicarbonate supplementation prevents chronic kidney disease (CKD) progression is mixed and depends on disease stage and dose

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: Some studies suggest a benefit: a prospective study found that sodium bicarbonate slowed the rate of creatinine clearance decline from 5.93 to 1.88 mL/min per 1.73 m²/year in patients with stage 4 CKD a peer-reviewed study noted that oral bicarbonate supplementation slowed eGFR decline in stage 4 CKD but not in stage 5 CKD

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: However, other research has produced negative or inconclusive results — a randomized trial with a mean follow-up of 1.35 years found no effect of bicarbonate on kidney failure progression another study found that a low dose (0.5 mEq/kg/day) did not significantly reduce urinary TGF-β in advanced diabetic CKD — suggesting that any protective effect may be conditional on stage, dose patient population

### Sample conflictingqa_56fd6bf22253

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Adenoids can grow back after removal, although it is generally considered uncommon and rarely causes significant problems

### Sample conflictingqa_57190bca6f7a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: The 1815 Mount Tambora eruption was the most powerful volcanic eruption in recorded history , but the retrieved evidence does not confirm it was the deadliest in terms of total fatalities

### Sample conflictingqa_5e7a6a2debfd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Male bees do not perform organized labor like females — drones do not build the hive, gather food care for young; instead, they spend their lives eating honey and waiting to mate

### Sample conflictingqa_613a0093714b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The phrase is popularly associated with 17th century England, but the claim lacks definitive evidence

### Sample conflictingqa_62b1aff6586d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: The ozone layer is healing, but the process is gradual and not yet complete

### Sample conflictingqa_62b1aff6586d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: A 2025 MIT-led study confirmed with 95% statistical confidence that the Antarctic ozone layer is recovering as a direct result of global reductions in ozone-depleting substances like CFCs NASA data continues to show the hole growing smaller year after year

### Sample conflictingqa_62b1aff6586d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: However, some sources note that a small hole still persists over Antarctica and that emerging factors such as rocket launches are slowing the rate of recovery , while one source describes the overall problem as 'essentially solved'

### Sample conflictingqa_63fde268aa8c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Religious and philosophical traditions such as Sanatana Dharma and Cartesian dualism assert that the mind is separate from the body, while some scientists and philosophers argue that the mind-body distinction is a fiction and that thoughts, sensations movements arise from the same psychobiology

### Sample conflictingqa_676188bf8139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: Yes — the Lantern Festival does involve honoring/deifying ancestors for the recently deceased, though the main ritual is lighting lanterns rather than direct ancestor worship

### Sample conflictingqa_6988dd820a61

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The retrieved evidence is conflicting

### Sample conflictingqa_6988dd820a61

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Some studies, particularly a University of Tokyo study published in Nature Geoscience, found that large earthquakes are more likely to occur near the time of maximum tidal stress, during full and new moons, with high tidal stress associated with a greater chance of an earthquake growing to magnitude 8 or above

### Sample conflictingqa_6988dd820a61

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: However, a U.S. Geological Survey study published in Seismological Research Letters found that the incidence of earthquakes showed no relationship to the moon's phase or position, concluding that the idea is 'not some wild, crazy idea' but rather a persistent lore where people ascribe significance to coincidences

### Sample conflictingqa_6988dd820a61

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: The strength of evidence is contested, with d4 (high-quality) directly challenging , while d5 hedges its findings with conditional language

### Sample conflictingqa_6d7279b0a8ed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: No, the Gutenberg Bible was not the first book printed with movable type

### Sample conflictingqa_6d7279b0a8ed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: While it is widely recognized as the earliest major European book produced using mass-produced metal movable type, significant evidence exists of earlier printed works from Asia

### Sample conflictingqa_6d7279b0a8ed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: A 2019 article on Open Culture confirms that the Jikji, a Korean collection of Buddhist teachings printed in 1377, predates the Gutenberg Bible by 78 years and is considered the oldest existing text printed with movable type

### Sample conflictingqa_6d7279b0a8ed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Additionally, a Gizmodo article notes that Chinese and Korean inventors had been producing printed books for centuries before Gutenberg was even born, further challenging the claim that his Bible was the first such publication

### Sample conflictingqa_6d7279b0a8ed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: In essence, the Gutenberg Bible holds a prominent place in the history of Western printing but cannot be accurately described as the first book ever printed with movable type

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The retrieved evidence is conflicting

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: Some sources argue that split ends cannot be permanently repaired because hair is dead tissue and the damage cannot be biologically reversed, meaning that trimming is the only true solution

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Other sources argue that specialized bond-building products can temporarily mask the appearance of split ends and prevent further damage, serving as a practical alternative to frequent cutting

### Sample conflictingqa_6fe31cd2ef65

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Not always — roll the R for double-RR words and single-R words at the beginning of a sentence, but not for single-R sounds in the middle or end of a word

### Sample conflictingqa_734859937e46

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: It depends on jurisdiction; in the U.S. federal rules generally allow ISPs to sell browsing history without consent, but some states require opt-in California's CCPA gives users an opt-out right

### Sample conflictingqa_73560dfab1ae

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: The evidence on whether high doses of vitamin C alleviate common cold symptoms is mixed and contested

### Sample conflictingqa_73560dfab1ae

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: High-credibility sources indicate that vitamin C does not reliably prevent colds, though some research suggests it may modestly reduce cold duration and severity

### Sample conflictingqa_73560dfab1ae

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: A peer-reviewed meta-analysis found that vitamin C reduced the severity of common colds by 15% across multiple randomized trials, with the greatest benefit observed for severe symptoms rather than mild ones

### Sample conflictingqa_73560dfab1ae

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, another study argued that clinical trial evidence is insufficient to recommend vitamin C supplementation for cold prevention or treatment, noting that while vitamin C supports immune function, taking extra doses does not consistently alter cold outcomes

### Sample conflictingqa_73560dfab1ae

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: Overall, while vitamin C may offer some benefit in reducing cold severity—particularly for those with compromised immune systems—most high-quality research concludes that routine supplementation does not prevent colds or produce meaningful symptom relief for the general population

### Sample conflictingqa_747727772a30

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Bees generally avoid flying in the rain because wet wings make it difficult to generate lift, though they may still forage during light rain or when driven by urgent needs such as defending the hive

### Sample conflictingqa_76956c2fba7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The relationship between saturated fats and heart disease risk is genuinely contested in the scientific literature

### Sample conflictingqa_76956c2fba7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: Some studies, including research presented at the European Society of Cardiology Congress 2024, report that a diet high in saturated fat can adversely affect cardiovascular disease risk factors such as liver fat and cholesterol levels, even without weight gain the American Heart Association notes that saturated fats tend to raise LDL cholesterol, which is linked to increased heart disease risk ; however, a 2014 meta-analysis of randomized controlled trials found that reducing saturated fat and replacing it with carbohydrate did not significantly reduce heart disease outcomes a 2021 systematic review similarly concluded there was insufficient evidence to confirm a causal relationship between saturated fat intake and cardiovascular events

### Sample conflictingqa_76956c2fba7d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Methodological differences in how these studies were designed and conducted—particularly around confounding variables and the type of fat used to replace saturated fat—appear to drive these divergent conclusions, meaning that definitive policy guidance remains elusive for this widely debated question

### Sample conflictingqa_7998e59d9b97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: On average, organic farming produces lower crop yields than conventional farming, typically estimated at 20–25% less

### Sample conflictingqa_7998e59d9b97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, this yield gap is part of a broader sustainability trade-off: organic systems often perform more favorably on environmental metrics such as chemical pollution, soil health biodiversity preservation

### Sample conflictingqa_7998e59d9b97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The choice between organic and conventional farming is therefore contested, with some researchers arguing that high-yield conventional farming is more land-efficient and flexible to sustainable improvement , while others contend that organic methods are the more resilient and environmentally preferable option overall

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d2
- **Claim**: The retrieved evidence presents competing views

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: Some sources (e.g., ecatholic2000) assert that the Catholic Church is the one true church by claiming apostolic succession, scriptural foundations divine institution

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Other sources (e.g., GotQuestions, Reddit) argue that Scripture alone determines which church is true, that the Catholic Church is not clearly identified in the Bible as the one true church that the claim of exclusivity is contested by other Christian denominations

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The conflict reflects differing interpretive and hermeneutical approaches to scriptural authority and church history

### Sample conflictingqa_7cf85109a70d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Bronze is more durable than brass

### Sample conflictingqa_80857a692531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: Farmed and wild salmon are broadly similar in nutritional value, with both serving as excellent sources of protein, omega-3 fatty acids vitamins

### Sample conflictingqa_80857a692531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: However, some studies note that wild salmon tends to have higher levels of certain vitamins — such as vitamin D and vitamin A — and lower fat content compared to farmed salmon , while farmed salmon can accumulate higher levels of environmental contaminants such as PCBs

### Sample conflictingqa_80857a692531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: Overall, the consensus across large-scale nutritional analyses is that the differences are moderate and the health benefits of eating either type of salmon are considered to outweigh these distinctions

### Sample conflictingqa_80baf25496cd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Multiculturalism may hinder unity in cases where strong cultural affiliations prevent assimilation or integration, but research also indicates that multiculturalism can facilitate civic and political cohesion

### Sample conflictingqa_8848765fc18a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: Spelunking and caving are generally considered the same activity — the American Heritage Dictionary defines both as 'the sport or pastime of exploring caves,' and Merriam-Webster similarly treats them as synonyms

### Sample conflictingqa_8848765fc18a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: However, some sources draw subtle distinctions: Wikipedia notes that 'caving' is the more technically oriented, experienced-focused term, while 'spelunking' carries connotations of casual or amateur exploration , a view echoed by the Cave Research Foundation and others who use 'caving' for advanced, equipped trips and 'spelunking' for simpler, recreational experiences

### Sample conflictingqa_8848765fc18a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: In practice, the two terms are used interchangeably in most contexts, including by major caving organizations and publications

### Sample conflictingqa_894f4a4b9552

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: The majority of scientific evidence strongly supports the existence of dark matter: observations of galaxy rotations, gravitational lensing, the cosmic microwave background the large-scale structure of the universe all indicate the presence of a massive, invisible component influencing gravity

### Sample conflictingqa_894f4a4b9552

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, a minority of researchers argue that modifying gravity theories (e.g., MOND) or invoking exotic physics could explain these phenomena without invoking dark matter direct experimental detection of dark matter particles remains elusive, leaving the question contested at the frontier of research

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: The retrieved evidence indicates that bird calls are not universally unique to each individual but vary by species and call type

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Some birds, such as songbirds, do develop highly individualized songs through learning, while others—such as waterfowl and shorebirds—are born with innate calls that show less variation

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Further research shows that ecological factors like habitat, body mass beak size also shape call diversity across species

### Sample conflictingqa_8cf8ebb94554

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: The evidence on knee brace effectiveness is mixed and depends heavily on the type of brace and context

### Sample conflictingqa_8cf8ebb94554

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Some studies suggest prophylactic braces may reduce the risk of knee injuries in contact sports functional braces are generally accepted as beneficial for supporting knees during rehabilitation after an injury ; however, the American Academy of Family Physicians notes there is currently no conclusive evidence supporting the broader preventive use of knee braces the OHSU Center for Health & Healing similarly states that no research confirms a knee brace is a cure for knee problems

### Sample conflictingqa_8cf8ebb94554

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Effectiveness also varies by population and activity type, so individual consultation with a healthcare provider is strongly recommended before relying on a knee brace for injury prevention

### Sample conflictingqa_8efa53ba7c60

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Birds are indeed descendants of dinosaurs they share a common theropod ancestor with T-Rex, though they did not descend directly from T-Rex itself

### Sample conflictingqa_9251c7a33ec5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: he evidence is mixed

### Sample conflictingqa_9251c7a33ec5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: Some research suggests that spaying or neutering can increase the risk of certain conditions, such as some cancers, joint disorders urinary issues, particularly when performed very early in life

### Sample conflictingqa_9251c7a33ec5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Other studies argue that these procedures reduce mortality rates and decrease the incidence of other diseases, such as mammary tumors and prostatic diseases, thereby conferring substantial health benefits

### Sample conflictingqa_9251c7a33ec5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Factors such as age at sterilization, sex, breed outcome being studied appear to significantly influence the balance of risks and benefits

### Sample conflictingqa_9261438d6ee2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: Yes — the scientific consensus, based on multiple peer-reviewed studies, is that fish do possess pain receptors and respond to noxious stimuli in ways similar to mammals

### Sample conflictingqa_9261438d6ee2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Research has shown that fish exhibit avoidance behavior, physiological stress responses altered brain activity when exposed to painful stimuli that these responses can be reduced by analgesic drugs

### Sample conflictingqa_9261438d6ee2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: While some researchers debate whether fish experience pain as humans do — citing differences in brain anatomy and the absence of confirmed subjective awareness — the evidence overwhelmingly supports the conclusion that fish do feel pain

### Sample conflictingqa_9275911a2961

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Direct evidence: yes

### Sample conflictingqa_962d8f5d5574

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The retrieved evidence is mixed

### Sample conflictingqa_962d8f5d5574

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Some sources assert that all snakes can swim , while others qualify this by noting that swimming ability is unknown for the vast majority of snake families and species one study found that all 525 snake species with available information appear able to swim but noted data is incomplete for the full 3951 species considered

### Sample conflictingqa_962d8f5d5574

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Additionally, some sources distinguish by saying most snakes can swim readily while all can manage it to some degree certain specialized snakes like sea snakes are particularly adept

### Sample conflictingqa_9b11b8e571aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2, d1
- **Claim**: Gonorrhea is primarily and most commonly transmitted through sexual contact, but limited non-sexual transmission is also possible

### Sample conflictingqa_9b73cb6cce52

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Giant African Land Snails can make acceptable pets if provided with specific care including temperature range 24–30°C and humidity management, though failure to meet these needs can cause health problems

### Sample conflictingqa_9ceca2645833

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Affirmative Action is not per se reverse discrimination

### Sample conflictingqa_a1e36a8db854

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The question of whether glyphosate is harmful to humans is genuinely contested

### Sample conflictingqa_a1e36a8db854

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: The EPA has concluded that glyphosate does not pose a risk to humans when used according to directions and that it is unlikely to cause cancer, while Health Canada similarly found that proper use does not cause harmful effects; however, the Seattle Statement—a consensus of Washington University scientists—concluded that evidence linking glyphosate to cancer, kidney and liver disease, reproductive toxicity neurological harm is now so strong that regulatory action is urgently justified EPA's own scientists disagreed with the agency's conclusion, finding that glyphosate causes DNA damage in human cells at real-world exposure levels

### Sample conflictingqa_a1e36a8db854

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Research published in Nature and presented at the American Association for the Advancement of Science further supports harmful effects, including the herbicide's ability to cross the blood-brain barrier and contribute to neuroinflammation

### Sample conflictingqa_a25014a5c5b5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: Generally, plants cannot survive without any light because light is essential for photosynthesis—the process by which they convert carbon dioxide and water into energy and oxygen

### Sample conflictingqa_a25014a5c5b5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: However, some plants possess remarkable adaptations that allow them to endure extended periods of darkness: for instance, certain species like snake plants and philodendrons can thrive in low-light conditions or with artificial grow lights a plant's roots can sustain it temporarily if attached to another plant exposed to sufficient light ; similarly, some species such as Epipremnum (pothos) exhibited continued growth in a zero-light experiment, suggesting that complete darkness is not universally fatal

### Sample conflictingqa_a2f06d54b240

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Yes

### Sample conflictingqa_a3264277980a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The question of whether Orson Welles's 1938 War of the Worlds radio broadcast caused mass panic is genuinely contested among scholars and historians

### Sample conflictingqa_a3264277980a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Traditional accounts, including Welles's own recollections and contemporary newspaper reports, describe widespread hysteria, suicides heart attacks attributed to listeners mistaking the fictional alien invasion for reality

### Sample conflictingqa_a3264277980a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: However, modern research—such as surveys showing only about 2% of the listening audience found the broadcast realistic hospital records failing to confirm any broadcast-related casualties—lead many academics to conclude that the notion of mass panic was a media-driven myth perpetuated by newspapers seeking to discredit radio as a competing news source

### Sample conflictingqa_a3264277980a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: A middle-ground view holds that while true full-scale national panic may have been exaggerated, the broadcast did demonstrate the powerful persuasive capability of the new medium of radio and likely caused genuine concern among a subset of listeners

### Sample conflictingqa_a3980a2921cf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Yes — hair oil is beneficial for all hair types

### Sample conflictingqa_a507c5b61631

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: Volcanic activity is among the leading hypotheses for triggering the PETM, but the evidence is contested and incomplete

### Sample conflictingqa_a7ff288bc615

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: AI has passed the Turing test a peer-reviewed arXiv study confirms that large language models exhibit human-like intelligence

### Sample conflictingqa_a7ff288bc615

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: GPT-4.5 was judged to be human 73% of the time in a UC San Diego experiment, statistically surpassing real humans , while GPT-3.5 also achieved scores indicating human-indistinguishable performance in prior evaluations

### Sample conflictingqa_a7ff288bc615

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: However, critics such as Gary Marcus argue that these claims are premature, as the bar for 'passing' was set artificially low and a trained judge would likely be far harder to fool , suggesting ongoing debate about the significance of these results

### Sample conflictingqa_a864ff85e648

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Yes

### Sample conflictingqa_a8ec1c0d2b92

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The evidence is mixed

### Sample conflictingqa_a8ec1c0d2b92

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Some sources argue that green tea does not directly cause kidney stones and may even reduce the risk due to its antioxidant content and diuretic effects , while others advise that individuals with a history of calcium oxalate stones consume it in moderation because the tea contains oxalates, which can contribute to stone formation in susceptible individuals

### Sample conflictingqa_a8ec1c0d2b92

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Overall, the consensus leans toward green tea being safe and potentially beneficial for most people, but those with a strong risk or existing condition should exercise caution and, if possible, consult a healthcare provider

### Sample conflictingqa_a994724a28e7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: The44 majority of experts and research concludes that cold water does not make hair shinier, as hair contains no living cells and the temperature difference between hot and cold water is considered negligible

### Sample conflictingqa_a994724a28e7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: However, some sources argue that cold water can help seal the cuticle temporarily, potentially enhancing shine, though this effect is generally regarded as minor, impermanent easily reversed by subsequent heat styling

### Sample conflictingqa_a9bed39d234d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2, d1
- **Claim**: The negative-calorie food concept is controversial, with d1 and d5 asserting that certain foods can burn more calories than they provide, while state that evidence is lacking and it is unlikely any food is truly negative-calorie

### Sample conflictingqa_b2524e4883ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The retrieved evidence presents competing views

### Sample conflictingqa_b2524e4883ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Some sources argue that meteor showers primarily pose a threat to spacecraft and satellites rather than to people or the planet generally; for example, NASA's Chandra X-ray Observatory team must implement protective measures during showers like the Camelopardalids

### Sample conflictingqa_b2524e4883ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Other sources emphasize that the majority of meteor shower debris is too small to survive atmospheric entry that even larger chunks (e.g., boulder-sized Taurid objects) remain statistically rare and unproven as past threats

### Sample conflictingqa_b2524e4883ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Overall, the evidence suggests that meteor showers represent a real but limited risk—significant for technology in orbit, but generally not catastrophic for life on Earth

### Sample conflictingqa_b323dd4b5820

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Current CO2 levels are not unprecedented in Earth's history in absolute terms, as levels were similarly high during the mid-Pliocene 3.3 million years ago and potentially as recently as 14 million years ago, according to Columbia University and NOAA ; however, the rate of increase is exceptional, with human-driven emissions causing CO2 to rise 100–200 times faster than natural increases did at the end of the last ice age

### Sample conflictingqa_b323dd4b5820

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: Wikipedia confirms that CO2 has varied widely over Earth's history, from 180 ppm during glaciations to 4,000 ppm during the Cambrian , meaning that while today's levels are within the historical range—roughly 420 ppm—they are also rising at a pace never before observed in human history

### Sample conflictingqa_b7fd50f9f980

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2, d1
- **Claim**: Yes — 'alright' is a recognized and widely used variant of 'all right,' accepted by major dictionaries and sources, though 'all right' is generally preferred in formal or academic writing

### Sample conflictingqa_b9854bd5a19e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: Yes, the human brain has decreased in size over time

### Sample conflictingqa_b9854bd5a19e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: Fossil evidence indicates that modern human skulls are on average 12.7% smaller than those of Homo sapiens who lived during the last ice age brain size has decreased by approximately 10% since the Late Pleistocene (around 30,000 years ago)

### Sample conflictingqa_b9854bd5a19e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This reduction is linked to a shift from brute-force information processing to more metabolically efficient symbolic thinking, as well as declining body size and reduced physical demands in post-industrial societies

### Sample conflictingqa_b9854bd5a19e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: However, some researchers contest this interpretation, arguing that the supposed shrinkage is a statistical artifact or that brain size has not genuinely decreased ; alternatively, the Wikipedia entry on brain size notes that human brain volume has historically increased from about 600 cm³ in Homo habilis to 1680 cm³ in modern Homo sapiens, with the exception of Neanderthals whose brains exceeded ours , suggesting ongoing debate across temporal and methodological dimensions

### Sample conflictingqa_bac0f4d62f96

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The retrieved evidence is conflicting

### Sample conflictingqa_bac0f4d62f96

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Some sources argue that meteorites can come from comets, while others argue that few if any large meteorites come from comets

### Sample conflictingqa_bd2e652cd64d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Yes, electric toothbrushes are generally better for your teeth than manual ones

### Sample conflictingqa_bd2e652cd64d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: Research cited by Cleveland Clinic shows that electric toothbrushes remove plaque more effectively than manual brushes, which can help prevent cavities and gum disease

### Sample conflictingqa_bd2e652cd64d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: An 11-year study of over 2,000 patients found that those using electric toothbrushes had 22% less gum recession and 18% less tooth decay compared to manual users the American Dental Association notes that electric toothbrushes with timers and pressure sensors can improve gum health and reduce enamel damage

### Sample conflictingqa_bd2e652cd64d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: While manual toothbrushes can still clean teeth well when used properly, the majority of dental evidence supports electric toothbrushes as the superior option for most people

### Sample conflictingqa_bdee100fa8e5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The question of whether Orson Welles's 1938 War of the Worlds broadcast caused a real-life panic is genuinely contested among scholars and historians

### Sample conflictingqa_bdee100fa8e5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: While the broadcast is legendary for allegedly triggering mass hysteria, including suicides and heart attacks, these dramatic claims are regarded by many academics as overblown media sensationalism rather than factual documentation

### Sample conflictingqa_bdee100fa8e5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Careful analysis of contemporaneous survey data and listener letters suggests that few people actually tuned in to the broadcast and that those who did were generally aware it was fictional, with most panic-stricken reports coming from secondhand accounts or newspaper exaggeration

### Sample conflictingqa_be17259fe5c0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: Not according to a 2020 genomic study, which concluded that penguins first evolved in Australia and New Zealand and then spread to Antarctica — though an earlier fossil-based analysis argued an Antarctic origin was 'highly likely.'

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The environmental comparison between paper and plastic straws is nuanced and contested

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Some sources argue that paper straws are not significantly more environmentally friendly, as their production can generate 44 times the greenhouse gas emissions of plastic straws a UK government assessment found that paper straws actually emit more greenhouse gases when they decompose in landfills than plastic straws do

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: On the other hand, paper straws are biodegradable and do not contribute to the same degree of long-term marine pollution as plastic straws, which can persist for centuries and harm marine life ; additionally, plastic straws are rarely recycled most end up in incineration or landfills, where they produce methane emissions

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Experts and researchers consistently emphasize that refusing straws altogether is the most environmentally sound choice, as the evidence remains mixed depending on which stage of the life cycle is prioritized

### Sample conflictingqa_bf6e9339f2c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: Yes, nutritional yeast is considered a complete protein source for vegans

### Sample conflictingqa_bf6e9339f2c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: Nutritional yeast is an 'excellent source of highly digestible complete protein' and serves as a valuable B12 supplement for those who do not eat animal products ; research published via NIH similarly confirms that yeast protein contains all essential amino acids in quantities meeting FAO/WHO dietary recommendations , while GoodRx notes that nutritional yeast is 'loaded with protein' and serves as a key resource for vegans alongside other plant-based sources like beans, lentils nuts

### Sample conflictingqa_bfbbc2c7a1af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Yes

### Sample conflictingqa_c1119b945459

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: The retrieved evidence indicates that Hindu beliefs are complex and not universally monolithic

### Sample conflictingqa_c1119b945459

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: Some sources argue that Hindus believe in one ultimate reality (Brahman) manifested through many forms, while others describe Hinduism as polytheistic or henotheistic

### Sample conflictingqa_c34991d9897e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Yes, copyright can protect logos — but only if the logo contains original artistic or creative elements; otherwise, trademark law is the more appropriate protective mechanism

### Sample conflictingqa_c3e66630928c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2, d1
- **Claim**: Research and user reports are divided on whether coffee grounds deter slugs and snails — some studies and gardeners report modest success, particularly with stronger caffeine solutions, while others find the grounds insufficient or ineffective when applied dry

### Sample conflictingqa_c418fecfc1e2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: Yes

### Sample conflictingqa_c574530da7a3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Religious and theological views differ; science offers no settled answer

### Sample conflictingqa_cc71318e5853

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The retrieved evidence presents conflicting views on whether death remains a taboo topic in modern society

### Sample conflictingqa_cc71318e5853

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: Some sources argue that death is still widely avoided in conversation and media, with one source claiming it is 'the last taboo' in American culture others noting that discussing death makes people uncomfortable unless they are personally affected or work in related professions

### Sample conflictingqa_cc71318e5853

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: On the other hand, some researchers and commentators argue that death is not truly a taboo topic in modern society, but rather that it is openly discussed in certain contexts, such as healthcare and academia that the pandemic has brought death into broader public consciousness in recent years

### Sample conflictingqa_cd661c2c20b6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Gwen Stacy's death is frequently cited as a transformative moment in Spider-Man's history and the comics industry, but sources differ on whether it marks the actual end of the Silver Age

### Sample conflictingqa_d295f9ea94b3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Botox is not considered a type of plastic surgery; it is classified as a non-surgical cosmetic procedure

### Sample conflictingqa_d295f9ea94b3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Unlike plastic surgery, which typically involves surgical incisions, Botox consists of botulinum toxin injections that relax facial muscles to reduce the appearance of wrinkles

### Sample conflictingqa_d295f9ea94b3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2, d1
- **Claim**: High-credibility sources such as the American Society of Plastic Surgeons and Columbia University Medical Center further corroborate that Botox is categorized as a cosmetic injectable procedure, distinct from surgical interventions like facelifts or breast augmentations

### Sample conflictingqa_d9a36fe4c135

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: Religious views differ; the Bible is infallible to many Christians (especially Catholics and conservative Protestants), but not all scholars or denominations agree

### Sample conflictingqa_dc6f972e8447

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: Yes, cryptocurrency markets are vulnerable to a variety of manipulative practices, including wash trading, spoofing, pump-and-dump schemes leverage exploitation, which can amplify the impact of relatively small initial price moves

### Sample conflictingqa_dd426f7706e8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The retrieved evidence is conflicting

### Sample conflictingqa_dd426f7706e8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Some sources claim that a full moon can create werewolves or trigger transformations, while others argue that the full moon only exposes existing lycanthropy rather than causing it

### Sample conflictingqa_e2e2361dc28b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: Yes

### Sample conflictingqa_e60bdcf75ba7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Yes, organic yields are generally lower than conventional yields — on average 18–25% less , with the gap widening to 84% across U.S. crop comparisons ; however, the gap varies considerably by crop type, region management practices some studies suggest organic can nearly match conventional yields for legumes, perennials under optimal conditions

### Sample conflictingqa_e93e708d49a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Over their lifetime, typical solar panels generate substantially more energy than they consume during manufacturing, installation recycling — with a typical energy payback time of around 2 to 4 years — meaning they easily pay for themselves multiple times over their usual 25-year lifespan

### Sample conflictingqa_ece626a6cba9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Yes

### Sample conflictingqa_f1932b75ace7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2, d1
- **Claim**: There is significant scientific debate and conflicting evidence regarding whether bee stings treat arthritis

### Sample conflictingqa_f1932b75ace7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Proponents, including early 20th-century physicians and contemporary anecdotal reports, claim that bee venom eases joint pain and inflammation, with one study noting that bee venom contains multiple anti-inflammatory components

### Sample conflictingqa_f1932b75ace7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: However, peer-reviewed research has documented cases of 'beekeeper's arthropathy'—an inflammatory arthritis occurring in beekeepers after stings—suggesting that bee venom can sometimes cause rather than treat arthritis-like conditions

### Sample conflictingqa_f1932b75ace7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: While some sources describe remarkable individual improvements, experts and health guidelines consistently emphasize that more rigorous clinical trials are needed before bee venom can be recommended as a valid arthritis treatment

### Sample conflictingqa_f22b389be1d6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: Heads differ on whether barefoot running is healthier; the predominant scientific evidence suggests it can reduce certain injuries and strengthen foot muscles, but risks exist

### Sample conflictingqa_f39c966c2ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: The retrieved evidence presents competing views

### Sample conflictingqa_f39c966c2ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: Some sources claim the curse was manifest from the first performance—citing the sudden death of the original Lady Macbeth actor and the consequent replacement by Shakespeare himself—while others argue that statistical analysis shows Macbeth does not experience more accidents than other plays, suggesting the curse is a theatrical legend rather than a demonstrable historical fact

### Sample conflictingqa_f3b163170581

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: Evolutionary evidence and mainstream science indicate that humans share a common ancestor with apes, though the precise details of the split and subsequent evolution remain subjects of ongoing research

### Sample conflictingqa_f3ba8599e370

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: The retrieved evidence presents competing views

### Sample conflictingqa_f3ba8599e370

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Some sources argue that yoga is not a religion in itself — for example, it is described as a 'spiritual discipline' or 'technology' that emphasizes direct experience rather than religious faith

### Sample conflictingqa_f3ba8599e370

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: On the other hand, other sources argue that the essence of yoga and religion are exactly the same, as both aim to join the individual to divinity yoga does contain significant religious elements

### Sample conflictingqa_f3ba8599e370

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: Ultimately, whether yoga constitutes a religion depends on how one defines 'religion'; if religion is defined broadly as a spiritual practice connecting the individual to divinity, yoga could be seen as religious, but if religion is defined narrowly as an organized system of faith, worship belief in a higher power, yoga may be distinguished from formal religion

### Sample conflictingqa_f43b2c51deea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Yes

### Sample conflictingqa_f4693bea2c31

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Emoji are widely debated as a form of written language, with experts offering conflicting views

### Sample conflictingqa_f4693bea2c31

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Some argue that emoji are not a new language but rather an enhancement or supplement to existing written language, functioning as a complex system of pictographs that add nuance, tone emotion without replacing words

### Sample conflictingqa_f4693bea2c31

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: Others, however, contend that emoji exhibit linguistically significant features—such as contextual ambiguity, polysemy emerging compositional use—that bring them closer to a distinct written system, even if they lack the full morphological and grammatical complexity of conventional languages

### Sample conflictingqa_f4811561af0c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: The retrieved evidence indicates that the Dutch were among the earliest Europeans to discover and explore Australia, though they did not settle or fully comprehend the continent

### Sample conflictingqa_f777a43ba278

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The link between yerba mate and cancer is nuanced and subject to ongoing scientific debate

### Sample conflictingqa_f777a43ba278

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: The NIH cites research indicating that high daily consumption of very hot yerba mate is associated with increased risks of esophageal, laryngeal oral cavity cancers, with the mechanism potentially related to thermal damage rather than the tea itself

### Sample conflictingqa_f777a43ba278

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Additionally, some studies suggest that PAHs (polycyclic aromatic hydrocarbons)—known carcinogens found in yerba mate—may contribute to this risk, particularly when consumed in large quantities over prolonged periods

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: On the other hand, laboratory research has shown that yerba mate possesses anti-cancer properties, capable of inducing cell death in cancer cells, though these findings have yet to be confirmed in human clinical trials

### Sample conflictingqa_f777a43ba278

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Overall, the evidence suggests that the temperature at which yerba mate is consumed may be a key factor in determining risk, with cooler temperatures carrying lower esophageal cancer risk that moderation is generally advised

### Sample conflictingqa_f7fec8c0688b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2, d1
- **Claim**: The retrieved evidence presents competing views

### Sample conflictingqa_f7fec8c0688b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: The official military explanation, cited by multiple sources, attributes the lights to LUU-2B/B rescue flares dropped by A-10 aircraft during a training mission one source describes this theory as "more or less proven." However, many witnesses, including Arizona's former governor Fife Symington, found this explanation unconvincing, arguing that flares cannot replicate the silent, star-blocking formation observed

### Sample conflictingqa_f7fec8c0688b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2, d1
- **Claim**: The conflict between the official military account and eyewitness testimonies persists, with no consensus established

### Sample conflictingqa_f8da23d84ecc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Once considered the same dinosaur, Apatosaurus and Brontosaurus were reclassified as distinct genera by a 2015 scientific study

### Sample conflictingqa_f970957c5e52

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The retrieved evidence presents conflicting opinions

### Sample conflictingqa_f970957c5e52

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Some sources argue that the Oxford comma is optional and its use is primarily a style-choice matter, while others argue that it is necessary for clarity, especially in complex lists

### Sample conflictingqa_f97fef94decc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2, d1
- **Claim**: There is no consensus that VR headsets cause permanent eye damage , but conflicting opinions exist: some sources argue that modern headsets are less harmful than smartphones or computers while others note that prolonged use can lead to temporary symptoms like eye strain and dryness some users have reported specific vision problems attributed to prolonged headset use

### Sample conflictingqa_f97fef94decc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: Research outcomes further diverge, with studies showing no serious vision deterioration in children after short-term use contrasting with anecdotal reports of convergence issues and expert warnings about long-term risks

### Sample conflictingqa_fa98c00bd697

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Black holes are not directly visible to optical telescopes because their gravity is so strong that nothing, including light, can escape

### Sample conflictingqa_fa98c00bd697

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: However, scientists can detect black holes indirectly through their effects on nearby matter, such as accretion disks and gravitational lensing advanced radio telescopes have captured direct images of black holes' event horizons

### Sample conflictingqa_fa98c00bd697

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: One notable example is the black hole at the center of the galaxy M87, which was imaged by the Event Horizon Telescope in 2019, showing a black hole silhouetted against a bright, orange ring of hot gas

### Sample conflictingqa_fad0d30903d2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Yes, Woodstock festival promoted peace and love

### Sample conflictingqa_fbedb688b1d9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: Religious scholars and commentators hold differing views on whether Mormons are Christian, depending on the definition of Christianity they adopt

### Sample conflictingqa_fcdb9e210683

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: Yes — viral genomes are placed in a phylogenetic tree

### Sample freshqa_0293a11bd364

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Hindi

### Sample freshqa_02b3ba89ebd0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_0436c0b3a9d7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: Aryna Sabalenka and Amanda Anisimova were the finalists in the 2025 US Open women's singles tournament

### Sample freshqa_047057d22309

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: King Charles did not strip Prince Harry's title as the Duke of Sussex; Harry and Meghan Markle agreed to stop using their HRH titles in early 2020 as part of their departure from the Royal Family the official Royal Family website subsequently removed all references to 'HRH' Prince Harry

### Sample freshqa_049cc3f14d5e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: St. Petersburg State University

### Sample freshqa_1009f5c49e12

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Paris

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2, d1
- **Claim**: August 16, 1977

### Sample freshqa_122352ad92e3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Apr 1, 2026; Passover 2026 began on April 1

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Hillary Clinton has not enacted any executive orders as president; she has never held the office of President of the United States

### Sample freshqa_1ef881d26e2e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: Maryam Mirzakhani

### Sample freshqa_1fe3ba1b1cba

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_2130eea851fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Geoffrey Hinton has accumulated 1,035,072+ total citations on Google Scholar as of June 2026, with an h-index of 190 across 776+ publications

### Sample freshqa_2130eea851fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Earlier reporting from March 2023 noted he had just become the second computer scientist to exceed one million Google Scholar citations subsequent tracking platforms confirm his citation count has continued to grow well beyond that milestone

### Sample freshqa_25b286cb2af1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Venus has no confirmed moons

### Sample freshqa_263eca8e024e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Dangal

### Sample freshqa_2877cf4bd00f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: 78

### Sample freshqa_28e155139ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: The latest stable version of Android is Android 16, released in December 2025

### Sample freshqa_28e155139ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Android 16 introduced features such as AI-powered notification summaries, lock screen widgets, grouped notifications improved desktop mode, as well as a shift to more frequent update releases

### Sample freshqa_28e155139ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Earlier reports had described Android 15 as the latest official release, but this was superseded by the subsequent 2025 release of Android 16

### Sample freshqa_2b9ba7e192e2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Dina Boluarte (Dina Elisa Boluarte Zavaleta), who became Peru's first female president on December 7, 2022, after being sworn in during a turbulent political crisis that saw her predecessor Pedro Castillo impeached for attempting to dissolve Congress

### Sample freshqa_2b9ba7e192e2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: She had previously served as Vice President under Castillo and was the sixth Peruvian president in less than five years

### Sample freshqa_2b9ba7e192e2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Multiple authoritative sources, including Reuters, BBC Foreign Policy, consistently confirm this historical milestone with no contradictions

### Sample freshqa_2d7eb41139aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: 6

### Sample freshqa_2e51f51132ee

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The 2021 Children's & Family Emmy Awards took place on December 10 and 11, 2022

### Sample freshqa_2e51f51132ee

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The ceremony was held at the Wilshire Ebell Theatre in Los Angeles and honored programming from the 2021–2022 eligibility window

### Sample freshqa_2e51f51132ee

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: This date is confirmed by the Wikipedia entry on the 1st Children's and Family Emmy Awards and aligns with the NATAS announcement that the competition would debut in 2022

### Sample freshqa_31ad09b9cd22

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The latest Grammy Award for Best Jazz Performance was won by Chick Corea, Christian McBride Brian Blade at the 68th Annual Grammy Awards in 2026

### Sample freshqa_31ad09b9cd22

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Their recording "Windows - Live" earned the honor, making them the most recent recipients of this category

### Sample freshqa_31ad09b9cd22

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: While Samara Joy had previously won the award in 2025 for "Twinkle Twinkle Little Me," the 2026 win superscedes that as the most current

### Sample freshqa_3227ea6c6056

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_354e7097602d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The first atomic bomb test in the United States took place at the Trinity Site, located on the U.S. Army's Alamogordo Bombing and Gunnery Range in New Mexico

### Sample freshqa_354e7097602d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: Known officially as the Trinity test, this historic detonation occurred on July 16, 1945, at 5:30 a.m., releasing approximately 18.6 kilotons of power and scattering radioactive fallout across over 1,100 square miles of the region

### Sample freshqa_354e7097602d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: The site is now part of the White Sands Missile Range and is marked by a black lava rock obelisk, situated in the Jornada del Muerto desert approximately 210 miles south of Los Alamos, New Mexico

### Sample freshqa_35bf342002aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: 7

### Sample freshqa_3847b5cb9b42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2, d1
- **Claim**: Russia's invasion of Ukraine (2022–present)

### Sample freshqa_39dcd7b38c39

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: Maya Angelou was the first African American woman to appear on a U.S. quarter

### Sample freshqa_3ad16f379533

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Russia has been invading Ukraine, beginning with its full-scale land, sea air assault on February 24, 2022

### Sample freshqa_3ad16f379533

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: This invasion is further corroborated by the Australian government, which stated that Russia began its unprovoked, full-scale invasion of Ukraine on that same date, violating the United Nations Charter

### Sample freshqa_3ad16f379533

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: NPR additionally confirmed that it has been more than four years since Russia's full-scale invasion began, with the conflict persisting without major changes in battle lines

### Sample freshqa_3cd0b514193b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: The minimum hourly wage in Tokyo is ¥1,226 — the highest of any prefecture in Japan

### Sample freshqa_3cd0b514193b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: This rate is confirmed across multiple sources and supersedes older 2026 estimates citing lower figures ; the national weighted average minimum wage in Japan is approximately ¥1,121 per hour, with Tokyo standing at ¥1,226 as of October 2025

### Sample freshqa_3dc3cf00bce6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: Pembroke Welsh Corgi

### Sample freshqa_42796b35e143

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: 3

### Sample freshqa_4590bdd9e269

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_4a98eba95e97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Joe Biden did not visit Russia as president of the United States

### Sample freshqa_4a98eba95e97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: His only meeting with Russian President Vladimir Putin took place on June 16, 2021, at the Villa La Grange in Geneva, Switzerland, not within Russia itself

### Sample freshqa_4a98eba95e97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: This is confirmed across multiple authoritative sources, which note that Biden's foreign travel to Russia was ruled out due to the ongoing war in Ukraine, making the Geneva summit the sole bilateral meeting between the two leaders during Biden's presidency

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_4e635a2542a8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: Red Garland

### Sample freshqa_50f8f03fd30f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: The youngest passenger on board the Titanic was Millvina Dean, who was approximately nine weeks old (born 2 February 1912) when the ship departed on its ill-fated 1912 voyage

### Sample freshqa_50f8f03fd30f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: This, noting that she was just two months old when the ship sank on 15 April 1912

### Sample freshqa_50f8f03fd30f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: It is worth noting that while Sidney Leslie Goodwin was the youngest recovered victim at 19 months old , Millvina Dean holds the record as the youngest passenger aboard she ultimately survived the disaster

### Sample freshqa_5574b1447bdb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: Wuhan

### Sample freshqa_5d6e5db69928

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Greenland

### Sample freshqa_5eb89aae15f3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: KGF: Chapter 1

### Sample freshqa_5ecee1c55713

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Portugal

### Sample freshqa_64c12116affc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: Donald J. Trump is the President of the United States, having served two terms in office

### Sample freshqa_6927f007d7cc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: The Voice US season 29 winner

### Sample freshqa_6a45fadeb16b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Based on available evidence, Costco's Executive membership appears to cost approximately $120–$130 annually, depending on the source and timeframe consulted

### Sample freshqa_6f42c128eb6c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Harry Maguire has never won the Ballon d'Or, so there is no valid first year to report

### Sample freshqa_7bc7bb2dde20

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: The latest Academy Award for Best Picture was won by **One Battle After Another** (2025), directed by Paul Thomas Anderson, at the 98th Academy Awards

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This was confirmed by multiple sources covering the ceremony, with the movie also winning awards for Best Director and Best Adapted Screenplay

### Sample freshqa_7bc7bb2dde20

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Earlier reports citing 'Parasite' (2019) or 'All Quiet on the Western Front' (2022) as the most recent winner are outdated, superseded by the 2026 ceremony results

### Sample freshqa_7bc92b47dc43

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: 2

### Sample freshqa_7dce5d575302

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Kaka

### Sample freshqa_7e63fcff2dea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: The first animal to circle the Moon was a Soviet spacecraft carrying two Russian tortoises, named Major and Minor, which flew around the Moon in September 1968 on the Zond 5 mission

### Sample freshqa_7e63fcff2dea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: While several sources mention the first animal to orbit Earth (Laika the dog, 1957) or the first to survive Earth orbit, these differ from the specific query about landing on the Moon, which no source confirms was achieved by any animal

### Sample freshqa_7f1c3aae61a5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Luke Humphries

### Sample freshqa_80642f637dc7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Lionel Messi

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: George R. R. Martin, the author of "A Game of Thrones", was born in Bayonne, New Jersey

### Sample freshqa_8eca5bd62ae0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: Beijing is the first city in history to have hosted both the Summer and Winter Olympics

### Sample freshqa_8eca5bd62ae0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: It hosted the 2008 Summer Olympics and was then selected to host the 2022 Winter Olympics, making it the only city to have held both games

### Sample freshqa_8eca5bd62ae0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: This distinction is recognized by multiple authoritative sources, including Wikipedia's comprehensive list of Olympic host cities, which confirms Beijing as the first city to have hosted both editions

### Sample freshqa_8f302f0bfe82

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_97f3c1fe1fd4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Eminem

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Boating accident

### Sample freshqa_a47283064972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: No, the Toronto Raptors do not have a winning record in the latest NBA season

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The Britannica record shows the Raptors finished 25–57 in the 2023–24 season, which is well below .500 they missed the playoffs entirely

### Sample freshqa_a47283064972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: This is the most recent full season record available in the evidence it clearly indicates a losing record rather than a winning one

### Sample freshqa_a50d0f1f3cdf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: 9 September 2022

### Sample freshqa_a5492f36ca23

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: January 10, 2016

### Sample freshqa_a8b908895e11

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2, d1
- **Claim**: San José

### Sample freshqa_ab11b5dce00e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: USA

### Sample freshqa_b0ffe73c1789

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Colleen Hoover has published at least 26 books, though some sources report a higher total of 34 books

### Sample freshqa_b0ffe73c1789

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This discrepancy reflects the fact that some sources are outdated and do not account for her more recent publications

### Sample freshqa_b0ffe73c1789

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: Her most notable works include *It Ends with Us* (2016), *Verity* (2018) the *Slammed* series, among many others

### Sample freshqa_b3264b37f54b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Yes, Arsenal is on top of the latest Premier League standings

### Sample freshqa_b99c189f2222

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_c3f10dc1632d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: Jiangsu

### Sample freshqa_c479e83e408f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: 15

### Sample freshqa_c4976f8629cb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: The heaviest reptile in the world is the saltwater crocodile (Crocodylus porosus), according to Quora and Britannica

### Sample freshqa_c7315f8b3029

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: May 5, 2026

### Sample freshqa_c7ac9d61059a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: $51,380

### Sample freshqa_cbfca321cce4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: Vincent van Gogh

### Sample freshqa_cf331ed7d09f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: macOS 26 Tahoe

### Sample freshqa_d4d59d75b206

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_d510972df578

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: By methodology: depending on whether inflation adjustment is applied and marketing costs are included, sources differ on which film holds the record

### Sample freshqa_dd85dcbc2262

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Aryna Sabalenka is the number 1 ranked female tennis player in the world as of May 4, 2026

### Sample freshqa_dd85dcbc2262

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: This is confirmed by the WTA official rankings, which list her at rank 1 with 9,960 points, ahead of Elena Rybakina (rank 2) and Iga Świątek (rank 3)

### Sample freshqa_dd85dcbc2262

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Earlier records show she first reached the top position after the 2023 US Open, holding it for eight weeks before being supplanted by Iga Świątek, though she has since reclaimed the No. 1 spot and accumulated over 82 weeks there as of March 2026

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: 12

### Sample freshqa_ddd643091cbe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: A permanent cure for cancer has not been developed; the retrieved evidence indicates that researchers are actively exploring new treatments such as vaccines and gene editing, but no established universal cure currently exists

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_e502143179d6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: October 2022

### Sample freshqa_edf4ae4f32e5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: 1941

### Sample freshqa_ef3ad40c6540

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Lakers

### Sample freshqa_f11e8d758fc6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Slugs do not have lungs in the same sense as mammals; rather, they possess a single lung-like structure called a mantle cavity, which is a hollow space within the mantle lined with blood vessel-rich tissue for gas exchange and communicates with the outside through a small opening called the pneumostome

### Sample freshqa_f11e8d758fc6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: This cavity is considered analogous to a lung and is present in all pulmonate slug families except the veronicellids, which have no lung

### Sample freshqa_f11e8d758fc6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: Therefore, most slugs can be said to have one lung, though the term 'lung' refers to a simplified respiratory chamber rather than the complex organ found in higher animals

### Sample freshqa_f5d8e53958c6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Hawaii

### Sample freshqa_f5eef5c415e8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: 28

### Sample freshqa_f6ac249bdf53

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d2
- **Claim**: Ta-Nehisi Coates

### Sample freshqa_f6cc6071caa5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: A total of 893 Nazca geoglyphs had been discovered, according to research published in Popular Mechanics

### Sample freshqa_fc07d4525dbf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The youngest age eligible for COVID-19 vaccination in the United States varies by vaccine type

### Sample freshqa_fc07d4525dbf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Moderna's Spikevax is authorized for children as young as 6 months, while the FDA updated Pfizer's emergency use authorization to cover only children ages 5 and older Novavax is approved for ages 12 and older

### Sample freshqa_fc07d4525dbf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Earlier guidance from the CDC and other sources had referenced a broader 6-month minimum for all vaccines, but the most recent FDA action supersedes this, making 5 years old the current minimum age for Pfizer recipients

### Sample freshqa_fd00b29e848c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: February 18–March 19

### Sample freshqa_ff04793833cc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: Andrew Johnson was never elected as President of the United States by popular vote; he became president through succession after Abraham Lincoln's assassination in 1865

### Sample healthcontradict_0060

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: No, a tepid sponge bath does not reduce fever in children and may be uncomfortable

### Sample healthcontradict_0145

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Yes

### Sample hotpotqa_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d7
- **Claim**: World War II

### Sample hotpotqa_0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10
- **Claim**: Amy Jo Johnson

### Sample hotpotqa_0056

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10
- **Claim**: England

### Sample hotpotqa_0062

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9
- **Claim**: South Park

### Sample hotpotqa_0063

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d10, d7, d5, d6
- **Claim**: Boston College

### Sample hotpotqa_0064

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d5
- **Claim**: Victor Mature

### Sample hotpotqa_0070

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: 15

### Sample hotpotqa_0071

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d3, d1
- **Claim**: Keyshia Cole

### Sample hotpotqa_0073

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d3
- **Claim**: Golf Magazine is owned by Time Inc

### Sample hotpotqa_0079

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: Dennis Publishing

### Sample hotpotqa_0083

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7
- **Claim**: 1988

### Sample hotpotqa_0100

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9
- **Claim**: MedStar Washington Hospital Center

### Sample hotpotqa_0123

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d9, d1
- **Claim**: Lit's best known song is "My Own Worst Enemy," which became their number one rock hit and helped their album "A Place in the Sun" achieve platinum certification

### Sample hotpotqa_0134

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d9
- **Claim**: 1963 Pan American Games

### Sample hotpotqa_0153

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Danny Manning

### Sample hotpotqa_0155

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7
- **Claim**: 1984

### Sample hotpotqa_0186

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0192

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d7
- **Claim**: More than 1,600 German scientists, engineers technicians were recruited in post-Nazi Germany as part of Operation Paperclip, though the subset who became central to the U.S. space program was a smaller group that included Arthur Rudolph as a key developer of the Saturn V rocket

### Sample hotpotqa_0196

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Stuart period

### Sample misinformation_0023

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: No

### Sample qacc_0013

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d7, d5, d4
- **Claim**: The Fourteenth Amendment

### Sample qacc_0023

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d3, d7, d1, d5
- **Claim**: Pentheus was torn apart by the maenads at the end of The Bacchae

### Sample qacc_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Justin Timberlake

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d2, d7, d5, d4
- **Claim**: The most reliable sources report 506 instances of the f-word in The Wolf of Wall Street, according to Guinness World Records , The Guardian , Time Variety

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d2, d3, d7, d5, d6, d4
- **Claim**: This figure is corroborated by multiple outlets including Wikipedia and Slate , while a single source claims 569 instances ; the 506-count is the most widely recognized and authoritative tally

### Sample qacc_0156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d4, d2
- **Claim**: Sheldon Collins

### Sample qacc_08cf866bcb9b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0ac549afb037

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: The last name Hansen is of Scandinavian and Northern European origin, derived from the personal name Hans and used as a patronymic surname in Danish, Norwegian, Dutch, Flemish North German traditions

### Sample qacc_0ac549afb037

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: It is the most common surname in Norway and is carried by the greatest number of people in Denmark, making it one of the most prevalent family names in the region

### Sample qacc_0ac549afb037

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Research based on 23andMe DNA data further indicates that the most commonly observed ancestry for people with the surname Hansen is British & Irish (36.8%), followed by French & German (25.6%) and Scandinavian (19.9%), suggesting broader migratory and linguistic roots across Northern Europe

### Sample qacc_0b3b6452e224

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: The Statue of Liberty was designed by French sculptor Frédéric Auguste Bartholdi, who modeled the statue after his mother and drew inspiration from the Roman goddess of liberty, Libertas

### Sample qacc_0b75ed799d46

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The Screen Actors Guild Awards are being held at the Shrine Auditorium & Expo Hall in Los Angeles, California

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: Following the North Africa campaign, the Allies moved eastward across North Africa and subsequently invaded Sicily

### Sample qacc_0d85f1089c4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The Beti Bachao-Beti Padhao campaign has had multiple brand ambassadors across different states

### Sample qacc_0d85f1089c4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Parineeti Chopra was initially chosen as the brand ambassador for Haryana's version of the campaign , while Sakshi Malik was later announced as the brand ambassador for the Haryana government's specific initiative

### Sample qacc_0d85f1089c4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: At the national level, Madhuri Dixit became the brand ambassador for the campaign the Madhya Pradesh government appointed mountaineer Bhawna Dehariya Mishra and her daughter Siddhi Mishra as brand ambassadors for their state-specific effort

### Sample qacc_0d85f1089c4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Additionally, Avani Lekhara serves as the brand ambassador for the Rajasthan chapter of the campaign

### Sample qacc_1025b0681710

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Cassie Scerbo plays Lauren Tanner in Make It or Break It

### Sample qacc_132167e66120

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: India won the Cricket World Cup in 1983, 2007, 2024 2026

### Sample qacc_132167e66120

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: The 1983 victory was India's first, contested under ODI rules and held in England, led by captain Kapil Dev ; subsequent wins came in the T20 format in 2007 (South Africa), 2024 (Australia) 2026 (on home soil in Ahmedabad)

### Sample qacc_15ffab2466f7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: The Phantom of the Opera has played in Toronto at multiple venues

### Sample qacc_15ffab2466f7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The production opened at the Pantages Theatre in Toronto on September 20, 1989, running through September 26, 1999 , with the theatre itself restored specifically for this long-running residency

### Sample qacc_15ffab2466f7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: In addition, a later production closed at the Princess of Wales Theatre on June 30, 2018 the show has also been performed at the Ed Mirvish Theatre

### Sample qacc_160a528ae07e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: 3

### Sample qacc_17dc360eea55

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: 13

### Sample qacc_19ca08790764

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d2
- **Claim**: Oliver Stark plays Buck (Evan "Buck" Buckley) in the TV show 9-1-1

### Sample qacc_1a764b8b6cf5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The rule of the three rightly guided caliphs was called the Rashidun Caliphate

### Sample qacc_1b95727cc286

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: The film Paid in Full is loosely based on the real lives of drug dealers Azie Faison, Rich Porter Alpo Martinez, who controlled much of the drug trade in New York City in the 1980s

### Sample qacc_1b95727cc286

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: In the movie, Azie Faison is portrayed by Wood Harris as the character Ace, Rich Porter by Mekhi Phifer as Mitch Alpo Martinez by Cam'ron as Rico

### Sample qacc_1b95727cc286

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The film also drew inspiration from the experiences of additional figures like Kevin Carroll, though the specific real characters depicted in the film are most commonly associated with these three principals

### Sample qacc_213701765f94

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: January 15, 2009

### Sample qacc_2243f17ccc38

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: 1972

### Sample qacc_252987b8054c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: Tori Spelling

### Sample qacc_287da9f37864

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: Lionel Messi made his first appearance for Barcelona's first team on November 16, 2003, when he came on as a substitute in the 75th minute of a friendly match against Porto at the Estádio do Dragão

### Sample qacc_287da9f37864

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: This date is corroborated by multiple sources, including Barcelona's official website, which describes it as the moment that "goes down in FC Barcelona history"

### Sample qacc_287da9f37864

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: His official competitive debut followed on October 16, 2004, in a La Liga match against Espanyol, where he entered as a substitute for Deco at the age of 17 years and three months

### Sample qacc_290c939ed6e7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: February 9, 2018

### Sample qacc_292033e4b039

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: Muhammad

### Sample qacc_2a7f7e06e365

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: The first vertebrates to exist on earth were fish, which appeared around 480 million years ago

### Sample qacc_2a7f7e06e365

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: These early vertebrates belonged to a group called Sarcopterygians, which also gave rise to the first vertebrate land species

### Sample qacc_2a7f7e06e365

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: The broader context of vertebrate evolution shows that amphibians and reptiles later emerged around 360 million years ago, while birds and mammals appeared much later

### Sample qacc_2cbc9a53426f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Adrienne Barbeau played Oswald's mom (Kim Harvey) on The Drew Carey Show

### Sample qacc_2e1b5edb5e0d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The stratum lucidum

### Sample qacc_2ed872eb1114

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Louisiana

### Sample qacc_2f6d2647a424

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: Pete Rose

### Sample qacc_34cba3c71e06

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_367b09e4ed80

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: Jenny Slate

### Sample qacc_37fdedfe4478

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Susan Tedeschi

### Sample qacc_3c1297608017

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The practice of crossing fingers for good luck is generally traced to pre-Christian European traditions in which the cross-shaped gesture was seen as a powerful magical sigil capable of binding wishes or protecting against evil; it is also associated with early Christianity, as persecuted Christians reportedly used a variant of the gesture—the ichthys or fish symbol—to secretly recognize one another and invoke divine protection

### Sample qacc_3d4ebfa8b6dd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2, d1
- **Claim**: As a coach, Phil Jackson holds the record with 11 NBA championships; as a player, Bill Russell holds the record with 11 NBA championships

### Sample qacc_403a59870dc2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The Rams won the Super Bowl twice

### Sample qacc_403a59870dc2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Their first Super Bowl win was Super Bowl XXXIV on January 30, 2000, when the St. Louis Rams defeated the Tennessee Titans 23-16 at the Georgia Dome in Atlanta

### Sample qacc_403a59870dc2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Their second Super Bowl win was Super Bowl LVI on February 13, 2022, when the Rams defeated the Cincinnati Bengals 23-20 at SoFi Stadium in Inglewood, California, becoming the second NFL team to win the Super Bowl in its home stadium

### Sample qacc_41c44ecfd0f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d2
- **Claim**: The lymphatic vessels located in the small intestine are called lacteals

### Sample qacc_41c44ecfd0f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d2
- **Claim**: Lacteals are specialized lymphatic capillaries found in the intestinal villi, responsible for absorbing dietary fats (chyle) as well as transporting antigens and antigen-presenting cells to support the gut immune response

### Sample qacc_41c44ecfd0f0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: While Peyer's patches are also lymphoid structures in the ileum of the small intestine, they are not the lymphatic vessels themselves but rather clusters of lymphoid tissue associated with the villi

### Sample qacc_4387048ed24f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2, d1
- **Claim**: Anne Bancroft

### Sample qacc_44b315f6f4bb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: The retrieved evidence indicates that the Queen's crown jewels are kept in the Tower of London, specifically in the Jewel House within the Tower's grounds

### Sample qacc_44b315f6f4bb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: This is consistent with historical records showing that most crowns were moved to the Tower of London from Westminster Abbey in the 17th century that the Jewel House there served as the primary repository for state crowns and regalia

### Sample qacc_44b315f6f4bb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Additionally, the Queen's personal collection of jewels—distinct from the Crown Jewels—is reported to be stored 40 feet below Buckingham Palace in a converted air raid shelter

### Sample qacc_4fb90d57c274

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: December 1991

### Sample qacc_51b23ea15977

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The Soviet Union was leading the space race in April 1961, as Yuri Gagarin became the first human to travel into space on April 12, 1961, aboard the Vostok spacecraft

### Sample qacc_51b23ea15977

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: This achievement marked a significant milestone for the USSR in the space race against the United States, with the Soviets being first to send a person into space

### Sample qacc_51c89636151e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Manwë

### Sample qacc_531aff489b71

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: Kelly Reilly

### Sample qacc_54be882d5b58

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: The Italy episodes were filmed in Anguillara Sabazia, a town outside of Rome

### Sample qacc_5a9576fc5d8c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Jodie Sweetin

### Sample qacc_5eeb46719843

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Canada did not gain independence from Great Britain on a single date, as the transition was an evolutionary process rather than a momentous declaration

### Sample qacc_5eeb46719843

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The roots of Canadian independence trace back to 1867, when the British North America Act created the Dominion of Canada — a self-governing entity within the British Empire — marking the end of direct colonial rule ; this is why Canada Day is celebrated on July 1st, commemorating that year's confederation

### Sample qacc_5eeb46719843

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Further milestones included the Balfour Declaration of 1926, which recognized Canada as an autonomous community within the Empire the Statute of Westminster in 1931, which granted full legislative independence ; notably, the 1931 statute was retroactively applied to 1929, reflecting the gradual nature of this shift

### Sample qacc_5eeb46719843

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Some sources, particularly earlier educational materials, incorrectly identify 1867 as the year of full independence, but this conflates the creation of the Dominion with the completion of independence ; in fact, the final legal vestiges of colonialism were not fully removed until the Canada Act of 1982, which gave Canada the power to amend its own constitution without British parliamentary approval

### Sample qacc_5fb5c311d373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Lin-Manuel Miranda wrote "How Far I'll Go" for Moana

### Sample qacc_6485f021b694

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Carroll O'Conner and Jean Stapleton performed the theme song for All in the Family

### Sample qacc_66ba2af9c3b9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Soman Chainani

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_6837d86d03ac

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Prince William, the Prince of Wales, is first in line to the British throne

### Sample qacc_6969589d80c1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: Matt Monro

### Sample qacc_6af6e8cb8f34

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: Queen Charlotte, the German-born wife of George III, is credited with introducing the first Christmas tree to Britain in December 1800, when she set up a tree decorated with candles and sweets at Queen's Lodge, Windsor

### Sample qacc_6af6e8cb8f34

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: This predates the popularization of the tradition by Prince Albert and Queen Victoria in the 1840s, making Queen Charlotte the earliest known introducer of the Christmas tree to the UK

### Sample qacc_6b3b372cf27d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: Zooey Deschanel

### Sample qacc_6edf1477bd7e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Steve McEwan

### Sample qacc_6f8df54650a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: A U.S. passport provides visa-free or visa-on-arrival access to 180 countries and territories, making it among the most powerful passports in the world for travel freedom

### Sample qacc_6f8df54650a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: This figure is consistent across multiple sources, with the U.S. Department of State confirming that Americans can enter approximately 179 destinations visa-free, through visa-on-arrival via electronic travel authorization the U.S. Customs and Border Protection noting that reciprocal visa waiver arrangements with 42 countries further enhance this freedom

### Sample qacc_6f8df54650a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Older sources citing lower counts, such as 160 visa-free destinations, are outdated and superseded by more recent travel data

### Sample qacc_720d8b210357

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: Multiple — The number of DNA replication origins in eukaryotes varies by organism and chromosome size; d5 states the mechanism before enumerating specific counts across species

### Sample qacc_7916ffefdb99

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: John B. Watson

### Sample qacc_798b6853d20f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: glucose

### Sample qacc_7bf02a7deb69

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Charlie Day

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: October 1, 1968

### Sample qacc_7f5e5a4a4391

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The letter J was first used in the Middle Ages as a scribal variant of I and was formally established as a distinct letter after 1600 CE

### Sample qacc_7f5e5a4a4391

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: In English, the first books to clearly distinguish between I and J were the King James Bible 1st Revision Cambridge 1629 and an English grammar book published in 1633, making 1629–1633 a narrow window for J's practical introduction to English usage

### Sample qacc_7f5e5a4a4391

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: By the 16th and 17th centuries, scholars and printers had fully adopted J as a separate letter for words like 'Julius,' and it was finally acknowledged as a full-fledged distinct letter in the nineteenth century

### Sample qacc_883303a2d535

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Nana in Snow Dogs is a Border Collie

### Sample qacc_8882ab46be5d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: 38

### Sample qacc_899648874637

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: Kate Walsh plays Dr. Addison Shepherd (also known as Addison Forbes Montgomery) in Grey's Anatomy

### Sample qacc_8d7c14ed548f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Factor X (FX)

### Sample qacc_8daf80e943fa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: 5.88 trillion miles

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_8ef7b3cf5c3f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: The dominant ethnic group in southern South America (Argentina, Uruguay the Southern Cone region) are those of European descent, specifically Spanish and Italian origins, with Spanish heritage being the most prominent

### Sample qacc_9404250d756f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: The End of the F***ing World was primarily filmed in the United Kingdom, with Season 1 locations including Camberley, Surrey and the Isle of Sheppey (Kent), while Season 2 was shot entirely in Wales

### Sample qacc_940e6d9275f4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Billy Idol

### Sample qacc_946ecfb478b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Justin Timberlake

### Sample qacc_950881e7c998

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Boston Red Sox

### Sample qacc_9a9a28d7e159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Fairy Tail's final (third) season has already been released — it aired from October 7, 2018 to September 29, 2019

### Sample qacc_9a9a28d7e159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: This season completed the anime adaptation of Hiro Mashima's original manga, which had concluded earlier in 2017

### Sample qacc_9a9a28d7e159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: While the main series has ended, a sequel manga titled Fairy Tail: 100 Years Quest began serialization in 2018 and continues to publish new chapters bi-weekly, with Chapter 213 released on June 9, 2026

### Sample qacc_9b16fd6882f3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: Russ Ballard (written by/composed by)

### Sample qacc_9c2f95b14a78

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The Duluth Model is an intervention program that emphasizes a coordinated community response to domestic violence, placing accountability on offenders rather than victims focusing on stopping violence without repairing relationships

### Sample qacc_9fbf28f5786f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_a1c73439eca9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_a3c882e062c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: The planned completion date for the Sagrada Familia has been updated to the early 2030s, superseding the earlier target of 2026

### Sample qacc_a44267c115d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: Most of the water in the body is located within the intracellular space, accounting for approximately two-thirds of total body water, with the remaining one-third distributed in extracellular compartments such as interstitial fluid and blood plasma

### Sample qacc_a4dde83c35da

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: The Ming Dynasty had an autocratic government in which the emperor ruled personally after abolishing the prime minister's office, relying on a refined civil service system and trusted eunuchs to manage state affairs

### Sample qacc_a4dde83c35da

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: This system was described as excessively centralized and absolute, with the emperor taking direct control of the bureaucracy and establishing institutions such as the Grand Secretariat and Censorate to monitor official misconduct

### Sample qacc_a4dde83c35da

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: The Ming governmental structure persisted largely unchanged from 1368 to 1911, serving as a foundation for the subsequent Qing dynasty was characterized by the emperor's direct involvement in decision-making alongside a formal examination-based官僚体系。

### Sample qacc_a635c2fd4869

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: Roberta Flack

### Sample qacc_a6a2f8b1f0b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2, d1
- **Claim**: 233

### Sample qacc_a6b48b7accc4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: The first official T20 match was played between Sussex and Surrey in England in 2003 , while the first-ever T20 international was contested by New Zealand and Australia

### Sample qacc_a6df0af8c2ba

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d2
- **Claim**: Hosanna is a Hebrew expression derived from the phrase 'hoshi'a na' (הושיעה נא), meaning 'save us' or 'save us please!'

### Sample qacc_a6df0af8c2ba

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: In Greek, it is transliterated as 'hosanna' and used as an acclamation of praise, though it originally carried a supplicatory sense

### Sample qacc_a6df0af8c2ba

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: During the biblical account of Jesus' entry into Jerusalem, the crowd shouted 'Hosanna' as a joyful cry for salvation, making it a prominent element of Palm Sunday celebrations

### Sample qacc_a78a32b7b9a1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Atlanta Falcons

### Sample qacc_a91ae87c969d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Linda Davis

### Sample qacc_a927c4cccc6a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: 1977

### Sample qacc_aa94588b9477

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: 14 January 1960

### Sample qacc_aaf0f638e99b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2, d1
- **Claim**: A yellow 35 mph sign is an advisory speed plaque, not a regulatory speed limit sign

### Sample qacc_aaf0f638e99b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: These 'Horizontal Alignment Signs' indicate the measured safe speed for a specific curve or series of curves, but drivers can be ticketed for any speed if it is unsafe given current conditions

### Sample qacc_aaf0f638e99b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: In New Zealand, similarly styled yellow signs also denote suggested speeds and are treated as advisory rather than mandatory

### Sample qacc_ac5341df9a83

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: Troops for UN military actions come from Member States; the Security Council authorizes deployments via resolution UN Headquarters then liaises with countries to identify and deploy personnel

### Sample qacc_b0346f60b6ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: Celebrity Big Brother is available on CBS in the USA

### Sample qacc_b0ee06f2950d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Roanoke

### Sample qacc_b198a514fff8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: New Mexico was admitted to the Union as the 47th state

### Sample qacc_b1bd9518429b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Gibraltar

### Sample qacc_b281f09f0959

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Joseph McCarthy

### Sample qacc_ba7aaa9b36c8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: A fire broke out in the West Wing of the White House during a Christmas Eve party on December 24, 1929, destroying much of the wing

### Sample qacc_bc34664caee4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: California

### Sample qacc_bc7e9a7b4a83

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Usain Bolt

### Sample qacc_bf0bff050f03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: New Zealand

### Sample qacc_bfbb5f55a63f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_c264cb69676e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: A synovial saddle joint

### Sample qacc_c27400199055

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: Ghana

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: Seth MacFarlane

### Sample qacc_c675e6cd8ad6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: George Bruns composed the score for Disney's 1973 animated film Robin Hood

### Sample qacc_c69855566c76

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: Paul Reubens plays Pee-wee Herman in **Pee-wee's Big Holiday**

### Sample qacc_c731579bb51c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: 565

### Sample qacc_c88807a22775

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: .22 Long Rifle

### Sample qacc_c9b95dd57e73

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Peter Sarstedt

### Sample qacc_cb5bcdb1ef9b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: Elliott Gould played Trapper John in the 1970 MASH movie

### Sample qacc_cbddef47777e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Mishael Morgan

### Sample qacc_cbf25273f973

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: The last name Tavarez is of Hispanic origin, derived from the Portuguese and western Spanish surname Tavares

### Sample qacc_cbf25273f973

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: It is found mainly in the Dominican Republic and is also present in Cuba and Mexico, with people carrying the surname showing the highest ancestry proportion of Spanish and Portuguese descent

### Sample qacc_cbf25273f973

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The name's earliest recorded presence dates back to 13th century Portugal it is linked to notable Portuguese noble families involved in the Age of Exploration

### Sample qacc_ce4983c8a9c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Most effigy mounds were built between 750 and 1050 AD, though construction spanned a broader period from roughly 700 to 1200 AD

### Sample qacc_d00b0063e747

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: yes

### Sample qacc_d03e85bdc95a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Aristotle

### Sample qacc_d39801b5de65

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d2
- **Claim**: July 4, 1776

### Sample qacc_d3b85d857358

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The plane that dropped the atomic bomb on Hiroshima was the Enola Gay, a Boeing B-29 Superfortress bomber

### Sample qacc_d3b85d857358

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: On August 6, 1945, it became the first aircraft ever to drop an atomic weapon in warfare, releasing the bomb code-named 'Little Boy' over the city

### Sample qacc_d3b85d857358

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The Enola Gay was named after Enola Gay Tibbets, the mother of its pilot, Colonel Paul Tibbets is currently preserved and on display at the Smithsonian's National Air and Space Museum

### Sample qacc_d44802dc3c96

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: November 1936

### Sample qacc_d60bf850c4ff

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_d78d45c0e30f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Colombia and Japan qualified from Group H in the 2018 World Cup

### Sample qacc_d7c6682b5335

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_d7df0a1856b7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The Hubble classification of the Milky Way Galaxy is a barred spiral galaxy (Sb or SBb type)

### Sample qacc_d7df0a1856b7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: This means it features a central bar structure surrounded by spiral arms, fitting within the broader category of spiral galaxies defined by the Hubble sequence

### Sample qacc_d7df0a1856b7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The specific subclassification as Sb or SBb indicates a galaxy with moderately wound spiral arms and a noticeable but not dominant central bulge, characteristics consistent with observational evidence of the Milky Way's structure

### Sample qacc_d8b24beb2f90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: The balance sheet

### Sample qacc_d96b47272030

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: September 23, 1889

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: XXXTENTACION

### Sample qacc_e064a7a717ed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: The Glass Castle was primarily filmed in Montreal, Quebec, with additional filming in Welch, West Virginia on the To'hajiilee and Laguna Pueblo tribal lands near Albuquerque, New Mexico

### Sample qacc_e064a7a717ed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The production chose Montreal for most principal photography, while West Virginia provided authentic rural landscapes and New Mexico contributed desert scenery

### Sample qacc_e06ada156e0e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Nicole Gale Anderson

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: In Mexico, toll roads are called autopistas (or specifically numbered routes with the suffix 'D' for directo/cuota) the main federal operating agency is CAPUFE

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: The individual toll fees are referred to as 'cuotas,' and the primary payment methods include cash in Mexican pesos and electronic tags such as IAVE, Televia PASE

### Sample qacc_e6d89fce1b8e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Teddy Altman married Owen Hunt on Grey's Anatomy

### Sample qacc_e7318f6f3bbe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d2
- **Claim**: strengths

### Sample qacc_e87ffc07efd1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Franklin Delano Roosevelt

### Sample qacc_ea469c846404

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: 2025–26

### Sample qacc_eb6f14795c45

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Joan Cusack

### Sample qacc_eb7c676e133e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: The last time an astronaut went to the moon was December 14, 1972, during the Apollo 17 mission

### Sample qacc_eb7c676e133e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Apollo 17 Commander Eugene Cernan was the last astronaut to step on the lunar surface, with his bootprint marking the final human footprint on the moon

### Sample qacc_eb7c676e133e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: This mission remains the most recent human landing on the moon NASA's Apollo 17 records confirm that the crew splashed down back on Earth on December 19, 1972, after spending nearly 13 days in space

### Sample qacc_ec5b0067c29a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: Number One Observatory Circle

### Sample qacc_ecbc6adf8a48

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The retrieved evidence places the writing of 1 John between 70 and 110 AD, with scholars offering competing estimates: some favor 70–90 AD , others suggest 95–110 AD still others propose dates as early as before 70 AD or as late as 85–90 AD

### Sample qacc_ecbc6adf8a48

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Older sources, such as d4 and d5, further extend the range to the 90s AD, underscoring that no single date is universally accepted

### Sample qacc_ecd3d9c0ca11

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_f10c7ad4bb81

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d2
- **Claim**: Initialisms

### Sample qacc_f1776add7672

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2
- **Claim**: ICD-10 codes have a flexible length depending on the version and use case

### Sample qacc_f1776add7672

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: For procedure coding (ICD-10-PCS), each code is consistently seven characters long , while for diagnosis coding (ICD-10-CM), the length ranges from three to seven characters, with the first three characters forming the base code and additional characters providing greater specificity

### Sample qacc_f1776add7672

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The NHS data dictionary further clarifies that CM codes are minimally four characters long, with undivided three-character codes padded by 'X' and unused positions filled with '-' to maintain the structural flexibility

### Sample qacc_f2218f8c979e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2, d1
- **Claim**: The rib primal

### Sample qacc_f69c37496013

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: 1987

### Sample qacc_fbdae168fc6f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Sushma Swaraj

### Sample qacc_fbe562911999

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: In India's warrant of precedence, the Speaker of the Lok Sabha is placed at position 6 (Sl

### Sample qacc_fbe562911999

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2, d1
- **Claim**: No. 6), ranking above the Chief Justice of India and below only the President, Vice-President, Prime Minister Governor

### Sample qacc_fbe562911999

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This ranking is consistently confirmed across multiple sources, including the official Parliament of India publications and state-level tables of precedence

### Sample qacc_ff2cb00f4c03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: 7

### Sample situatedqa_geo_00f6ee9a6705

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: The Villages are located throughout Florida, spanning three counties: Lake, Sumter Marion

### Sample situatedqa_geo_082db791e263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: It depends on jurisdiction; federal U.S. law generally sets the minimum age at 18 for shotguns, while some states and other countries raise the threshold to 21

### Sample situatedqa_geo_09a6a048cfbf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: It depends on location; in the United States, the minimum legal drinking age is 21

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: In Ontario, red licence plates are used for both dealer plates and diplomat plates — dealer plates are white-background plates used by car dealerships, while diplomat plates have a red background and identify vehicles belonging to diplomats, consular officials foreign heads of mission

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: In Spain, red plates take on a different meaning, serving as temporary circulation plates for vehicles during registration processing, those temporarily out of service used for research and tests

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: Fleet vehicles , senior executive government vehicles specific statuses like Japanese diplomatic plates with red stripes , demonstrating that the meaning of red plates varies significantly by jurisdiction and context

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: 416,800 U.S. Army and Army Air Forces battle and non-battle dead, plus 382,700 UK dead — 800,000 total for the US and UK

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_362420ec2416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: As per the 2011 Census of India, Sikkim is the state with the lowest population, with approximately 6,10,577 residents

### Sample situatedqa_geo_362420ec2416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: This figure is confirmed across multiple sources, with Sikkim's population reported at around 607,688 to 6,10,577 depending on the specific dataset consulted

### Sample situatedqa_geo_362420ec2416

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that while Sikkim holds the record for the lowest state population, the Union Territory of Lakshadweep is the most minimally populated territory overall, with a population of just 64,429

### Sample situatedqa_geo_3815eef77a3c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The welfare state was introduced at different times across nations, making a single date difficult to provide

### Sample situatedqa_geo_3815eef77a3c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: In the United States, President Roosevelt established the American welfare state in the 1930s through the New Deal legislation, with key milestones such as the introduction of Social Security in 1935 ; in Europe, the German Empire under Otto von Bismarck pioneered social insurance programs in the late 19th century, starting around 1883 , while Britain's welfare state developed gradually from the Liberal reforms of 1906–1914, with major expansion occurring in the 1940s following the Beveridge Report

### Sample situatedqa_geo_48e19f6b37bc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: California is the 3rd largest U.S. state by area, with approximately 163,695 square miles (423,967 km²)

### Sample situatedqa_geo_48e19f6b37bc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: This is confirmed across multiple authoritative sources, including the U.S. Census Bureau data cited by Britannica, which ranks Alaska first, Texas second California third

### Sample situatedqa_geo_4cb699778b59

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: 6 years

### Sample situatedqa_geo_5f5c20228969

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_66684169f016

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: The Dandi March was led by Mahatma Gandhi and involved thousands of Indians, including notable figures such as Mithuben Petit and Pyare Lal Nayar

### Sample situatedqa_geo_66684169f016

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: The march began at Sabarmati Ashram with Gandhi accompanied by seventy-nine Ashramites and satyagrahis the document listing specific participants names members from Gujarat, Maharashtra UP

### Sample situatedqa_geo_6cca7fdb9b41

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The furthest point from the sea on Earth is the Eurasian pole of inaccessibility, located in northwestern China near the Kazakhstan border

### Sample situatedqa_geo_6cca7fdb9b41

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This location is approximately 2,645 km (1,643 miles) from the nearest coastline

### Sample situatedqa_geo_6cca7fdb9b41

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: For those interested in the furthest point from the sea specifically within Britain, the most cited candidate is Coton in the Elms, Derbyshire, with Church Flatts Farm being about 113 km (70 miles) from the nearest coast

### Sample situatedqa_geo_7222d6123c27

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Calcutta became the capital of British India in 1772, when Warren Hastings transferred all important offices there; the capital was later moved to Delhi in 1911

### Sample situatedqa_geo_74e7e677cae5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Social Security began on August 14, 1935, when the Social Security Act was signed into law by President Franklin D. Roosevelt

### Sample situatedqa_geo_74e7e677cae5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: The law established a federally administered system of social insurance for the aged, with payroll tax collection starting in 1937 and initial retirement benefit payments scheduled for 1942

### Sample situatedqa_geo_74e7e677cae5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: This makes 1935 the most commonly cited year for Social Security's official beginning, though the program developed in stages throughout the 1930s

### Sample situatedqa_geo_779fd84224fa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: Sydney Cove (also referred to as Sydney Harbour); the First Fleet arrived there in January 1788

### Sample situatedqa_geo_7f4d90dffb38

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: The federal excise tax on gasoline is 18.4 cents per gallon, though the overall tax rate varies by state

### Sample situatedqa_geo_80c72c2797bf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: The United States operates under a federal constitutional republic system, consisting of three coequal branches: legislative, executive judicial

### Sample situatedqa_geo_80c72c2797bf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: The legislative branch comprises Congress (the House of Representatives and Senate), the executive branch is headed by the President the judicial branch includes the Supreme Court and other federal courts

### Sample situatedqa_geo_80c72c2797bf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This structure is mandated by the U.S. Constitution, which also requires all states to uphold a 'republican form' of government, though the three-branch structure does not apply to state governments

### Sample situatedqa_geo_85af31651715

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: England, Wales Scotland each introduced smoking bans in pubs at different times — Scotland led the way on 26 March 2006, Wales followed on 2 April 2007 England completed the UK-wide ban on 1 July 2007

### Sample situatedqa_geo_85af31651715

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Northern Ireland also implemented a total ban in 2007, making the entire UK smoke-free in enclosed public spaces including pubs within a year of Scotland's pioneering legislation

### Sample situatedqa_geo_864fc7e36714

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_897e47478bbc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: 649,481

### Sample situatedqa_geo_8c889f8ce07a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Under U.S. constitutional design, treaty ratification is a joint process: the President negotiates and submits treaties to the Senate the Senate provides advice and consent — requiring a two-thirds majority to approve a resolution of ratification — after which the President formally proclaims entry into force

### Sample situatedqa_geo_8c889f8ce07a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: This constitutional balance means both branches must agree, with the Senate holding the power to withhold consent but not to formally ratify treaties itself

### Sample situatedqa_geo_932ca9fa3259

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_9f6c5f849b40

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: The most populous cities in the world are Tokyo (Japan), Shanghai (China) Jakarta (Indonesia), with respective 2025 population estimates of 33.4 million, 29.6 million 41.9 million

### Sample situatedqa_geo_9f6c5f849b40

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: These figures come from global rankings that distinguish clearly between city proper populations and broader urban agglomerations

### Sample situatedqa_geo_9f6c5f849b40

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: For reference, the three largest cities in the United States specifically are New York City, Los Angeles Chicago, with populations of approximately 8.8 million, 3.9 million 2.7 million respectively

### Sample situatedqa_geo_a1cde1787dab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: The Clean Air Act was passed in 1970, signed by President Richard Nixon on December 31, 1970

### Sample situatedqa_geo_a1cde1787dab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: This 1970 version superseded earlier federal air pollution laws passed in 1955 and 1963, making it the most current and comprehensive U.S. environmental statute

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Eisenhower

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: The California state flag features a California grizzly bear (Ursus arctos californicus), which is the official state animal of California

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: The bear on the flag is specifically identified as a grizzly bear, making it the most prominent symbol of the state

### Sample situatedqa_geo_c76a815bb275

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_d0532ee6a455

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Jordan

### Sample situatedqa_geo_d982055a66d8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_eec0ebc0dd88

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Scotland won the Calcutta Cup in 2026, making them the current holders

### Sample situatedqa_geo_eec0ebc0dd88

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: This supersedes older records showing England as the most recent winner, such as the 2018 match in which Scotland defeated England 25-13 at Murrayfield

### Sample situatedqa_geo_f2031e426cee

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Arjun Ram Meghwal is the present Minister of Law and Justice in India

### Sample situatedqa_geo_f26078ec6467

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Spain

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: The first form of national government after the Revolutionary War was the Articles of Confederation, adopted by the Second Continental Congress on November 15, 1777 ratified by the states in 1781

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Under this framework, the United States functioned as a loose confederation of semi-autonomous states, with a weak central government that lacked the power to tax, regulate commerce enforce laws uniformly

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This system was succeeded by the current U.S. Constitution, which was drafted at the 1787 Constitutional Convention and ratified in 1788, after influential groups found the Confederation government inadequate

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The shift from tea to coffee in the U.S. began during the American Revolutionary era: after the Boston Tea Party of 1773, drinking British tea became politically unfashionable coffee—imported through different supply chains—became the patriotic alternative

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: This cultural turning point persisted even after the Revolution coffee further eclipsed tea in 1865 when Union soldiers returned home from the Civil War having become accustomed to it as part of their standard rations ; by the 20th century, coffee dominated American daily life

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Broadly, the transition was shaped by political symbolism, immigration patterns industrial infrastructure—though regional and personal exceptions remained in some contemporary contexts, the trend has partially reversed

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The Federal Open Market Committee (FOMC) is the primary body that sets U.S. monetary policy

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: It is a Federal Reserve System entity consisting of the seven members of the Board of Governors and five regional Federal Reserve Bank presidents, who meet regularly to decide on tools like the federal funds rate and open market operations

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: The FOMC is considered the U.S. central bank its decisions are widely tracked as the key driver of monetary policy

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: in the United States, environmental policy is set at both the federal and state levels

### Sample situatedqa_temp_051502801f9c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: 1972

### Sample situatedqa_temp_05d714be23fd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: Ludacris

### Sample situatedqa_temp_0ad0871fcc33

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Wilt Chamberlain

### Sample situatedqa_temp_0c2289f57504

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Hamid Ansari

### Sample situatedqa_temp_14a587def215

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: 2025

### Sample situatedqa_temp_14f29fe7ff15

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: The British under General Howe defeated the Continental Army at the Battle of Brandywine on September 11, 1777

### Sample situatedqa_temp_14f70522567e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Lionel Messi

### Sample situatedqa_temp_180f238d8296

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Australia (5 titles), India (3 ODI titles), West Indies (3 ODI titles), England (2 ODI titles), Pakistan (1 title), Sri Lanka (1 title) — ODI Champions

### Sample situatedqa_temp_1987d35f994b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: October 27, 1986

### Sample situatedqa_temp_19badef7553b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: The Philadelphia Eagles won the Super Bowl twice in the retrieved evidence

### Sample situatedqa_temp_19badef7553b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Their first championship came on February 4, 2018, when they defeated the New England Patriots 41-33 in Super Bowl LII, marking their first title since 1960

### Sample situatedqa_temp_19badef7553b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Their second title was in 2024, when they defeated the Kansas City Chiefs in Super Bowl LIX, giving them their second Super Bowl win in an eight-year span

### Sample situatedqa_temp_1baff64de20e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Rumer Willis

### Sample situatedqa_temp_1c56e575f096

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: New South Wales last won the State of Origin series in 2024

### Sample situatedqa_temp_23d05dc2d7dc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: LeBron James

### Sample situatedqa_temp_2cafa66bbe5d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: 23 miles

### Sample situatedqa_temp_2f45ec399f17

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Novak Djokovic

### Sample situatedqa_temp_2f5dc02a5ce1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Cory Booker

### Sample situatedqa_temp_2f73dc6c2df1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: Mariah Carey

### Sample situatedqa_temp_301378915064

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Merritt Wever

### Sample situatedqa_temp_3026b0491e32

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: John Williams composed the music for the first three Harry Potter films

### Sample situatedqa_temp_32d33d503f69

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d2
- **Claim**: The new Henry Danger movie is confirmed to premiere on January 17, 2025, at 7 PM ET on Nickelodeon, with a simultaneous release on Paramount+

### Sample situatedqa_temp_35156c8be377

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The answer depends on the metric used to define 'richest.' By total GDP, Nigeria has historically been the largest economy in Africa , with a 2016 GDP of $411.966 billion and continued growth since then

### Sample situatedqa_temp_35156c8be377

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: By GDP per capita measured in current US dollars, South Africa ranks as the top performer with approximately $403 billion in 2024

### Sample situatedqa_temp_35156c8be377

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: However, by GDP per capita adjusted for purchasing power parity (PPP), Seychelles is the most affluent African nation with an estimated PPP of $42,110 in 2025

### Sample situatedqa_temp_3521e5dc831c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: Gagan Narang

### Sample situatedqa_temp_3781ac7b3ead

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Darren Criss

### Sample situatedqa_temp_3df0e6082901

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: LSU

### Sample situatedqa_temp_40e6764f611f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Mort is a mouse lemur (family Cheirogaleidae), a small primate species native to Madagascar is specifically identified as a Goodman's mouse lemur (Microcebus demidoff) within the fictional Madagascar franchise

### Sample situatedqa_temp_43c0aaae9828

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Hillsong Worship

### Sample situatedqa_temp_44d30ce08bf5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: UCLA has won the most NCAA Women's College World Series titles with 12 championships

### Sample situatedqa_temp_44d30ce08bf5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: The Bruins defeated Fresno State in the inaugural 1982 tournament and have continued to dominate, claiming titles in 1984, 1985, 1988, 1989, 1990, 1992, 1999, 2003, 2004, 2010 2019

### Sample situatedqa_temp_44d30ce08bf5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: This record is confirmed across multiple sources, with Arizona and Oklahoma trailing at 8 titles each the most recent 2025 tournament results still place UCLA at the top

### Sample situatedqa_temp_45cbc4dc3e28

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The current Chief Justice of the Sindh High Court is **Justice Zafar Ahmed Rajput**, who was appointed to the position on 6 December 2025 and has continued in office since that date

### Sample situatedqa_temp_45cbc4dc3e28

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: This is confirmed by the official Sindh High Court list of Chief Justices, which shows his tenure listed as 'from 06-12-2025 to Till Today'

### Sample situatedqa_temp_45cbc4dc3e28

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Wikipedia previously identified Muhammad Junaid Ghaffar as the Acting Chief Justice from 14 February 2025 Saadat Khan held the role temporarily in November 2023, but these appointments have since been superseded by Justice Rajput's 2025 appointment

### Sample situatedqa_temp_4ead2f4cd2d5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: Chrishell Stause played Bethany Bryant on The Young and the Restless

### Sample situatedqa_temp_4fbe76f974ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: The song became widely known through Judy Garland's 1939 film The Wizard of Oz, but Israel Kamakawiwo'ole's iconic version first appeared on his 1993 album Facing Future

### Sample situatedqa_temp_50748f92be3a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: The last FIFA World Cup was held in 2022 Argentina won the title after defeating France in the final

### Sample situatedqa_temp_50748f92be3a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: This result is confirmed across multiple sources, with Argentina's victory being the most recent outcome of the tournament

### Sample situatedqa_temp_557d04a28d7f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: LeBron James

### Sample situatedqa_temp_587e89bbcbe1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: A standard UNO deck contains 108 cards, though this has changed over time

### Sample situatedqa_temp_587e89bbcbe1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: Earlier sources and some current ones still report 108 cards as the standard, with the deck divided into numbered cards (0–9), action cards (Skip, Reverse, Draw Two) wild cards

### Sample situatedqa_temp_587e89bbcbe1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: However, in 2018, Mattel officially updated the deck by adding two new action cards — Wild Swap Hands and Wild Shuffle Hands — increasing the total count to 112 cards per deck

### Sample situatedqa_temp_587e89bbcbe1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: As a result, newer UNO sets and most authoritative sources now reflect the updated 112-card total, while older references or some special editions may still reference the prior 108-card figure

### Sample situatedqa_temp_5a59faf24972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: The latest stable version of Android is Android 16, released on June 10, 2025

### Sample situatedqa_temp_5a59faf24972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: This supersedes earlier reports identifying Android 15 as the latest version, as the Android project released a new major update in mid-2025

### Sample situatedqa_temp_5a59faf24972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Android 16 is the sixteenth major release of the Android operating system and was first made available to Pixel phones before rolling out to other manufacturers

### Sample situatedqa_temp_5e6b388e88aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The Colorado Avalanche last won the Stanley Cup on June 26, 2022, when they defeated the Tampa Bay Lightning 2-1 in Game 6 of the Stanley Cup Finals

### Sample situatedqa_temp_5e6b388e88aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: This was the team's third overall championship and first in 21 years, marking the most recent time the Avalanche have hoisted the Cup

### Sample situatedqa_temp_60095fdf39a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_603f0dc417ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_61a79d74d827

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: Düsseldorf, Germany

### Sample situatedqa_temp_61d9095f5b70

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: July 23, 1986

### Sample situatedqa_temp_657c130afab6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d2
- **Claim**: 1980

### Sample situatedqa_temp_6a683d0d04f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: five sharps in a key signature mean the key is B major

### Sample situatedqa_temp_6a8aed3e3d5f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: episode 245

### Sample situatedqa_temp_6c424fc78a69

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: PTI won the 2018 election in Pakistan

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: Todd Monken

### Sample situatedqa_temp_78c89bf81e8b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: On naval ships, SS most commonly stands for steamship, denoting a vessel powered by a steam engine

### Sample situatedqa_temp_78c89bf81e8b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: In modern US Navy hull classifications, however, SS specifically signifies a submersible ship — the prefix is used in codes such as SSN (nuclear-powered attack submarine), SSBN (nuclear-powered ballistic missile submarine) SSGN (nuclear-powered guided missile submarine)

### Sample situatedqa_temp_78c89bf81e8b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Additionally, some sources interpret SS as submarine ship, which is consistent with its current naval usage

### Sample situatedqa_temp_7a7a7f43e575

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Washington

### Sample situatedqa_temp_7cd18101326e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Indiana QB Fernando Mendoza was named the Offensive MVP and Indiana DL Mikail Kamara was named the Defensive MVP

### Sample situatedqa_temp_7d5f559b313d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: The most recent GDP data shows that the United States GDP reached 31.819 trillion dollars in the first quarter of 2026

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Australia's coastline is reported at approximately 25,000 kilometers (or about 15,535 miles), making it one of the longest in the world

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: However, high-precision measurements from scientific research and government surveys reveal a more detailed breakdown: the mainland coastline alone extends 35,821 km when island coastlines are included, the total length reaches 59,681 km

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: These differing figures reflect methodological variations in measurement scale, with smaller rulers capturing more coastline detail

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_7f07f7a6c607

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: Mohamed Salah

### Sample situatedqa_temp_8808c106a115

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: Tay-Sachs disease is an autosomal recessive genetic disorder

### Sample situatedqa_temp_8808c106a115

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: It is caused by a deficiency of the hexosaminidase A (HEX A) enzyme, which is necessary for breaking down GM2-ganglioside in cells, particularly in the brain and nervous system

### Sample situatedqa_temp_8808c106a115

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: The disease is inherited when an individual receives two variant copies of the HEXA gene — one from each parent — and it is estimated to occur in approximately 1 in 3,000 individuals of Ashkenazi Jewish, French Canadian Cajun descent, with a general population incidence of about 1 in 300,000

### Sample situatedqa_temp_88ef1cfab62e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Hunter Emery plays CO Rick Hopper in Orange Is the New Black

### Sample situatedqa_temp_8b66ca78c3d1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: 11,937

### Sample situatedqa_temp_8b8427cd7b6d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: The Cumberland River begins near Harlan, Kentucky, formed by the confluence of its headwater forks—Poor Fork, Clover Fork Martin's Fork—on the Cumberland Plateau

### Sample situatedqa_temp_8b8427cd7b6d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: It flows generally westward through the mountains of Kentucky before turning south into Tennessee, traveling through Nashville then bending northwest back into Kentucky

### Sample situatedqa_temp_8b8427cd7b6d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: The river ultimately ends by merging with the Ohio River at Smithland, Kentucky, northeast of Paducah

### Sample situatedqa_temp_8f57822ed7d2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2, d1
- **Claim**: 2020

### Sample situatedqa_temp_8fbdda192a13

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: September 1967

### Sample situatedqa_temp_901be1437bc7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Kent County, Maryland

### Sample situatedqa_temp_9049c197a579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: Drivers in California paid approximately $0.90 per gallon in taxes, consisting of $0.18 in federal tax and $0.72 in state taxes (including $0.60 excise tax, $0.10 sales tax $0.02 underground storage tank fee)

### Sample situatedqa_temp_9049c197a579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: By May 2026, the total gas tax had risen to approximately 70 cents per gallon, making it the highest in the country

### Sample situatedqa_temp_9049c197a579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The most recent excise tax rate for gasoline was $0.612 per gallon for the period of July 2025 through June 2026, with a 2.25% sales tax rate also applying

### Sample situatedqa_temp_956e1f6e518e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Apollo 17 (December 1972) — the last time humans were on the moon

### Sample situatedqa_temp_9a91e43d8136

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_a2624f6c031f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: 11,428,604

### Sample situatedqa_temp_a3a59c3f4b9b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Ramesh Kuntal Megh

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d2
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_a5e5db28902b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: 23 million

### Sample situatedqa_temp_a8024d95c6e1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Episode 10

### Sample situatedqa_temp_a8c74bced99a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: March 13, 624 CE

### Sample situatedqa_temp_ac0e6a4a7e32

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: Sun Yat-sen

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Shay Mitchell, who played Emily Fields in Pretty Little Liars, is 39 years old as of 2024

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: This makes her more than five years older than her character, who was originally depicted as a 16-year-old high school student in Rosewood

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Wikipedia confirms that Shay Mitchell was born on November 19, 1993, making her approximately 31 years old in 2024

### Sample situatedqa_temp_ae0882e48812

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The two largest deserts in China are the Gobi Desert and the Taklamakan Desert

### Sample situatedqa_temp_b3ad3248b7d4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The retrieved evidence supports 1438 as the founding of the Inca Empire under Pachacuti

### Sample situatedqa_temp_b6c75b99a2f4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: The longest wavelengths in the visible spectrum are approximately 700 nm, corresponding to the color red

### Sample situatedqa_temp_b6c75b99a2f4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: This is consistent across multiple authoritative sources, including Wikipedia, which notes that a typical human eye responds to wavelengths from about 380 to about 750 nanometers NASA's Chandra X-ray Observatory, which places visible light between infrared and ultraviolet while noting that red light represents the longest visible wavelengths

### Sample situatedqa_temp_b6c75b99a2f4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: Additionally, scientific references confirm that visible light spans roughly 400–700 nm, with red at the upper end of this range crossword evidence further corroborates that 670 nm is specifically associated with red as the longest visible wavelength

### Sample situatedqa_temp_b797de4c6610

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Cardiac biomarkers are substances that appear in the blood when the heart is stressed or damaged they are used to diagnose and monitor heart disease

### Sample situatedqa_temp_b797de4c6610

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: The most commonly used cardiac biomarker is cardiac troponin (troponin T or I), which enters the bloodstream shortly after a heart attack and remains elevated for days, making it the preferred marker per AHA guidelines ; other traditional enzymes previously used include creatinine kinase (CK) and its heart-specific subtype CK-MB, as well as myoglobin, though these are considered less specific

### Sample situatedqa_temp_b797de4c6610

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: Additional biomarkers used in clinical practice and research include natriuretic peptides (such as BNP or NT-proBNP), C-reactive protein (CRP) uric acid, each serving different diagnostic or prognostic purposes across conditions like heart failure, ischemia inflammation

### Sample situatedqa_temp_bb7a678b9ec9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Los Angeles (1932, 1984, 2028 Summer), Lake Placid (1932, 1980 Winter), Atlanta (1996 Summer), Palisades Tahoe/Squaw Valley (1960 Winter) St. Louis (1904 Summer)

### Sample situatedqa_temp_bb85e25a8159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: Florida Panthers

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: HMS Queen Elizabeth was commissioned on December 7, 2017 formally declared operational in 2020

### Sample situatedqa_temp_bdc0853f9a23

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: India's rank in the 2018 Global Peace Index was 136th

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: The last name Gerard is of French and Norman origin, derived from the Old French personal name Gérard, which itself traces to the ancient Germanic elements gēr ('spear') and hard ('hardy, brave, strong')

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: It is also found in Haiti and has cognates across Germanic and Romance languages

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: In addition to its French roots, the surname was historically used among the Anglo-Saxon tribes of Britain and is listed in the Domesday Book of 1086, making it a rich and ancient name with connections to both French and English heritages

### Sample situatedqa_temp_c92bc2a37a57

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: For the 2025-26 NBA season, LeBron James is the highest-paid player with total earnings of $132.6 million , while Shai Gilgeous-Alexander holds the top annual playing salary at $71.3 million per season starting in 2027-28

### Sample situatedqa_temp_c92bc2a37a57

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d5, d4
- **Claim**: Earlier records show Stephen Curry previously held the top annual salary for nine consecutive years prior analyses cited him and LeBron James as career leaders , but these rankings have since been superseded by Shai's recent contract extension

### Sample situatedqa_temp_ce10e6f883fb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: India and Pakistan

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The WTO has 166 member countries

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This figure is corroborated by earlier data showing the total reached 166 members by around 2022 , superseding older reports that cited 164 members as of 2016

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The most recent and accurate count, therefore, is 166 member countries

### Sample situatedqa_temp_d0579ca3907c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: The Battle of Kadesh is consistently dated to 1274 BCE

### Sample situatedqa_temp_d15849658c20

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: Oleksandr Usyk

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d2
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_d59edcfab87f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: Queen Charlotte of Mecklenburg-Strelitz, a German princess who became queen consort of Great Britain upon marrying King George III in 1761, is the namesake of Charlotte, North Carolina

### Sample situatedqa_temp_d59edcfab87f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: The city was officially incorporated in 1768 and has been known as the 'Queen City' ever since, reflecting its namesake's royal origins

### Sample situatedqa_temp_d59edcfab87f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by multiple authoritative sources, including the city of Charlotte's own official records and Britannica

### Sample situatedqa_temp_d63c04b198fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: 133

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: August 20, 1989

### Sample situatedqa_temp_dc4d0c4e24ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_dcdb77c2f3d9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: Paris, France

### Sample situatedqa_temp_df5975a9678a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: Saina Nehwal

### Sample situatedqa_temp_e3caa7edab1f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: 73

### Sample situatedqa_temp_ebd3c33e8be8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: Jonathan Bailey

### Sample situatedqa_temp_eed796e2fbed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Scottie Scheffler

### Sample situatedqa_temp_eeebc5ea5c4f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: The highest grossing movie in the Philippines is *Hello, Love, Again* (2024), which earned approximately ₱1.6 billion domestically

### Sample situatedqa_temp_eeebc5ea5c4f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This film surpassed the previous record holder, *Rewind* (2023), which grossed around ₱924 million

### Sample situatedqa_temp_eeebc5ea5c4f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Earlier sources, such as one citing *It Takes a Man and a Woman* (2013) with 405 million pesos , are outdated and superseded by these more recent box office figures

### Sample situatedqa_temp_efa79b1e7fb1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Stephen Curry (4,248 career 3-pointers as of April 2026); multiple sources including Wikipedia list and Yahoo Sports article confirming his all-time leadership

### Sample situatedqa_temp_f099d9d1452b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: The current US Director of the CIA is John L. Ratcliffe, who was officially sworn in on January 23, 2025

### Sample situatedqa_temp_f099d9d1452b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This is confirmed by the official CIA website, which notes that Vice President JD Vance administered the oath at a White House ceremony

### Sample situatedqa_temp_f099d9d1452b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: Director Ratcliffe is notable for being the first person ever to serve as both CIA Director and Director of National Intelligence, having previously held the latter role during President Trump's first term

### Sample situatedqa_temp_f196a847a496

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: 7

### Sample situatedqa_temp_f1e9ce4608b3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: Azzi Fudd

### Sample situatedqa_temp_f3b3face1b6b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: McDonald's Monopoly game pieces have appeared on various menu item packages and containers, including Big Macs, large fries breakfast sandwiches, as well as many other eligible items

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: 2021

### Sample situatedqa_temp_f971e49123a1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: 13

### Sample wikirevision_0001

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: Twitter is currently known as X. In April 2023, Twitter merged with X Holdings and ceased to be an independent company, becoming part of X Corp. This rebranding is confirmed across multiple sources, with Wikipedia redirecting 'Twitter' to the article on X (social network)

### Sample wikirevision_0001

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Prior to this change, Twitter had operated as a separate company since its founding in 2006

### Sample wikirevision_0002

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: Twitter is now known as X. In April 2023, Twitter merged with X Holdings and ceased to be an independent company, becoming part of X Corp. This rebrand was confirmed when Wikipedia's article on Twitter redirected to the article on X, indicating the name change as of 2023

### Sample wikirevision_0002

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: Prior to this merger, Twitter had been acquired by Elon Musk in October 2022, but the company was eventually rebranded to X

### Sample wikirevision_0003

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: Twitter is now known as X. In April 2023, Twitter merged with X Holdings and ceased to be an independent company, becoming part of X Corp. This rebranding is confirmed across multiple sources, with Wikipedia redirecting 'Twitter' to the article on X (social network), which notes the name change from 2006–2023

### Sample wikirevision_0004

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The current name of Facebook's parent company is Meta Platforms, Inc. This was confirmed when Facebook rebranded itself as Meta Platforms, Inc. in October 2022, officially changing its corporate identity

### Sample wikirevision_0004

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: The rebranding is further corroborated by additional context showing that Meta Platforms, Inc. is the parent company behind Facebook's products and initiatives

### Sample wikirevision_0007

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Google is owned by Alphabet Inc

### Sample wikirevision_0010

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Microsoft

### Sample wikirevision_0013

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Microsoft

### Sample wikirevision_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The latest President of India is Droupadi Murmu, who has held the office since July 2022

### Sample wikirevision_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: She is the 15th President of India and succeeded Ram Nath Kovind, making her the most recent holder of the position

### Sample wikirevision_0028

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The latest Prime Minister of India is Narendra Modi, who has served in office since 26 May 2014

### Sample wikirevision_0028

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: He is the Honourable Mr. Prime Minister and holds the highest office of the Government of India, being appointed by the President and responsible to the Lok Sabha

### Sample wikirevision_0032

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Emmanuel Macron is the current President of France, having been reelected in 2024 and taking office on 14 May 2017 for his second term

### Sample wikirevision_0032

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: He is the 25th holder of the office, succeeding François Hollande and being reelected ahead of his Socialist rival Marine Le Pen

### Sample wikirevision_0032

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across all available sources, including the high-credibility Wikipedia articles on both the President of France and the 2024 French presidential election

### Sample wikirevision_0033

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The current Chancellor of Germany is Friedrich Merz, who took office on May 6, 2025

### Sample wikirevision_0033

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: He is the 12th Chancellor of the Federal Republic of Germany and leads the Christian Democratic Union (CDU)

### Sample wikirevision_0033

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: This is consistent across multiple sources, including the current Wikipedia revision of the Chancellor of Germany article, which confirms his incumbency from May 6, 2025

### Sample wikirevision_0035

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: The latest Prime Minister of Japan is Sanae Takaichi, who assumed office on 21 October 2025

### Sample wikirevision_0035

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: She is the first female Prime Minister in Japanese history and has served in the role continuously since then

### Sample wikirevision_0035

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of the Prime Minister of Japan page, as well as the list of Japanese prime ministers

### Sample wikirevision_0040

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The latest President of Argentina is Javier Milei, who took office on 10 December 2023

### Sample wikirevision_0040

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: He is the incumbent President, serving as head of state and government at the Casa Rosada

### Sample wikirevision_0040

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple up-to-date sources, including the Wikipedia article on the President of Argentina, which also notes that the timestamp of the older revision (December 2024) and the newer revision (May 2026) both confirm his incumbency

### Sample wikirevision_0041

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The current President of Argentina is Javier Milei, who took office on 10 December 2023

### Sample wikirevision_0041

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: He is the 54th President of Argentina and belongs to the political party Unión por el Cambio

### Sample wikirevision_0041

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: This is consistent across multiple up-to-date sources, including the Wikipedia article on the President of Argentina, which also notes that the incumbent since December 10, 2023 is Javier Milei

### Sample wikirevision_0042

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Lee Jae Myung is the President of South Korea, serving as the country's head of state and government

### Sample wikirevision_0042

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by the official Wikipedia revision that superseded the older version in January 2026, which explicitly names him as incumbent with a detailed biography

### Sample wikirevision_0042

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Fourteen individuals have served as presidents of South Korea Lee Jae Myung is the most recent among them

### Sample wikirevision_0046

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Argentina (Argentina)

### Sample wikirevision_0047

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Argentina (defending 2022 champion, 3rd title) — the 2026 FIFA World Cup has not yet occurred, so the current champion remains Argentina

### Sample wikirevision_0049

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The current Indian Premier League champion is Royal Challengers Bengaluru (RCB), who won the 2026 IPL title — their first championship in the league's history

### Sample wikirevision_0049

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: This supersedes older information from the 2025 season, in which RCB was also listed as the champion , because the 2026 edition is the most recent completed season

### Sample wikirevision_0057

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: Google is owned by Alphabet Inc., a publicly traded company listed on Nasdaq

### Sample wikirevision_0057

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Alphabet was founded in 2015 as a reorganization of Google, making Google a wholly-owned subsidiary of Alphabet and Sundar Pichai the CEO of both companies

### Sample wikirevision_0057

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This structure is consistent across multiple sources, with d1 representing an older revision and d3 reflecting a more recent acquisition

### Sample wikirevision_0061

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: The current President of Mexico is Claudia Sheinbaum Pardo, who took office on 1 October 2024, making her the 66th President of Mexico

### Sample wikirevision_0061

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: She is the first woman and the first Jewish person to hold the office, serving a six-year term as the head of state and government

### Sample wikirevision_0061

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of the President of Mexico page, as well as her own Wikipedia biography

### Sample wikirevision_0064

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The current President of Turkey is Recep Tayyip Erdoğan, having served in office since 28 August 2014

### Sample wikirevision_0064

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: He is the 12th president in Turkey's republican history and also serves as the country's head of government and commander-in-chief

### Sample wikirevision_0064

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: This is consistent across multiple sources, including the older and newer Wikipedia revisions of the President of Turkey page, both of which confirm his incumbency since 2014

### Sample wikirevision_0065

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Facebook's parent company is Meta Platforms, Inc. This was confirmed when Facebook rebranded itself as Meta Platforms, Inc. in 2021 to reflect a strategic shift toward developing the metaverse

### Sample wikirevision_0065

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Meta Platforms is the current parent company of Facebook, owning and operating Facebook along with several other major social media platforms

### Sample wikirevision_0066

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: Facebook's parent company is Meta Platforms, Inc. This was confirmed when Facebook rebranded itself as Meta Platforms, Inc. in October 2022, officially changing its corporate identity

### Sample wikirevision_0066

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The rebranding was announced during the company's annual Connect developer conference the official press release stated that the change reflected Facebook's strategic focus on building the metaverse

### Sample wikirevision_0067

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: The current Ballon d'Or holder is Ousmane Dembélé, who won the 2025 edition (69th ceremony) for the 2024–25 season on 22 September 2025, marking his first win

### Sample wikirevision_0067

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: This is confirmed by the Wikipedia entry on the Ballon d'Or, which lists him as the holder and notes the 2026 ceremony as the next upcoming event

### Sample wikirevision_0067

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Earlier records show that the 2024 Ballon d'Or had gone to Spanish midfielder Rodri for the 2023–24 season, but that result has since been superseded by Dembélé's 2025 victory

### Sample wikirevision_0071

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Benjamin Netanyahu is the current Prime Minister of Israel, having assumed office on 29 December 2022

### Sample wikirevision_0071

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of the Prime Minister of Israel page, as well as the list of Israeli prime ministers

### Sample wikirevision_0072

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: Twitter is currently known as X. In April 2023, Twitter merged with X Holdings and ceased to operate as an independent company, becoming part of X Corp. This rebranding is confirmed across multiple sources, with Wikipedia redirecting 'Twitter' to the article on X (formerly Twitter) and noting the name change took effect in 2023

### Sample wikirevision_0074

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: The current Vice President of the United States is JD Vance, who assumed office on January 20, 2025

### Sample wikirevision_0074

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: This is confirmed across multiple authoritative sources, including the newer Wikipedia revision and the list of vice presidents of the United States

### Sample wikirevision_0074

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The Office of the Vice President further corroborates his tenure, noting that it is headed by the chief of staff to Vice President JD Vance

### Sample wikirevision_0076

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The latest Prime Minister of Pakistan is Shehbaz Sharif, who took office on 4 March 2024

### Sample wikirevision_0076

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by the official Wikipedia entries on both the Prime Minister of Pakistan and the Deputy Prime Minister of Pakistan, as well as corroborated by additional high-credibility sources

### Sample wikirevision_0076

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Shehbaz Sharif's appointment is further supported by the fact that he also served as Deputy Prime Minister before assuming the top role

### Sample wikirevision_0082

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: The current Ballon d'Or holder is Ousmane Dembélé, who won the 2025 award at the 69th ceremony

### Sample wikirevision_0082

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This makes him the most recent winner, superseding older information that had described the 2024 ceremony or referenced outdated seasonal models

### Sample wikirevision_0085

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The current Prime Minister of France is Sébastien Lecornu, who assumed office on 9 September 2025

### Sample wikirevision_0085

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: He is the 32nd Prime Minister of the French Republic and serves under President Emmanuel Macron

### Sample wikirevision_0085

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple up-to-date sources, including the high-credibility Wikipedia article on the Prime Minister of France, which also notes that the role has been known as 'Prime Minister' since 1959 when Michel Debré became the first officeholder under the Fifth Republic

### Sample wikirevision_0086

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The current Prime Minister of Pakistan is Shehbaz Sharif, who took office on 4 March 2024

### Sample wikirevision_0086

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: He is the 22nd Prime Minister of Pakistan and serves as the leader of the House and head of the executive branch of the Government of Pakistan

### Sample wikirevision_0086

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This is consistent across multiple up-to-date sources, including the Wikipedia article on the Prime Minister of Pakistan, which also notes that the office has been held by 20 different individuals since 1947

### Sample wikirevision_0088

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The latest Leader of the Labour Party in the UK is Keir Starmer, who was elected to the position on 4 April 2020

### Sample wikirevision_0088

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: This is confirmed by the official Wikipedia entries on both the Leader of the Labour Party and the 2020 Labour leadership election

### Sample wikirevision_0088

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Keir Starmer also serves as the Prime Minister of the United Kingdom

### Sample wikirevision_0089

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Calcutta is officially called Kolkata

### Sample wikirevision_0089

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The city officially changed its name from Calcutta to Kolkata in 2001 this is confirmed across multiple sources including the newer Wikipedia revision of Calcutta and the main Kolkata article

### Sample wikirevision_0089

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: While Calcutta is still commonly used, especially outside of India, the official civic body of the city is the Government of Kolkata

### Sample wikirevision_0090

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: The latest President of Indonesia is Prabowo Subianto, who took office on 20 October 2024

### Sample wikirevision_0090

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: He is the eighth president of Indonesia and served as the 26th minister of defence under President Joko Widodo from 2019 to 2024

### Sample wikirevision_0090

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This is consistent across multiple up-to-date sources, including the Wikipedia article on the President of Indonesia, which also notes that the presidency was established in 1945 with Sukarno as the first president

### Sample wikirevision_0093

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The current US Open men's singles champion is Carlos Alcaraz, who defeated Jannik Sinner in the 2025 US Open final

### Sample wikirevision_0093

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The 2025 US Open was the 145th edition of the tournament, held at the USTA Billie Jean King National Tennis Center in New York City

### Sample wikirevision_0093

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Seed ten, Jannik Sinner, was the defending champion but lost in the final to Carlos Alcaraz

### Sample wikirevision_0096

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The current Chief Justice of India is Surya Kant, who became the incumbent on 24 November 2025

### Sample wikirevision_0096

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: He is the 53rd Chief Justice to serve since the Supreme Court of India was established in 1950

### Sample wikirevision_0097

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Bengaluru

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The latest Cricket World Cup champion is Australia, who defeated India by six wickets in the 2023 final held in Ahmedabad on 19 November 2023

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This was Australia's sixth Cricket World Cup title the 2023 edition was the 13th overall, hosted entirely in India

### Sample wikirevision_0099

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The next scheduled tournament is the 2027 ICC Men's Cricket World Cup, co-hosted by South Africa, Zimbabwe Namibia

### Sample wikirevision_0100

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The current Prime Minister of Pakistan is Shehbaz Sharif, who took office on 4 March 2024

### Sample wikirevision_0100

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: He is the 22nd Prime Minister of Pakistan and serves as the leader of the House and head of the Government

### Sample wikirevision_0100

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This is consistent across multiple up-to-date sources, including the Wikipedia article on the Prime Minister of Pakistan, which also notes that Pakistan has had 20 prime ministers since 1947

### Sample wikirevision_0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The current Leader of the Labour Party in the UK is Keir Starmer, who was elected to the position on 4 April 2020 following his victory in the party's 2020 leadership election

### Sample wikirevision_0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of the Labour Party leadership page, both of which confirm his election on the same date

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This change was confirmed by the Haryana Government in 2016 the city officially changed its name from Gurgaon to Gurugram

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: As a result, Gurgaon is now known as Gurugram

### Sample wikirevision_0105

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: Bengaluru

### Sample wikirevision_0109

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: The current Prime Minister of Canada is Mark Carney, who assumed office on 14 March 2025

### Sample wikirevision_0109

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: He is the 24th person to serve as Canada's head of government and holds the highest office of the Government of Canada

### Sample wikirevision_0109

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of the Prime Minister of Canada page, as well as the list of prime ministers of Canada

### Sample wikirevision_0111

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Facebook's parent company is currently called Meta Platforms, Inc. This was confirmed when Facebook rebranded itself as Meta Platforms in 2021, officially changing its corporate identity

### Sample wikirevision_0111

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The rebranding was announced during the company's October 2021 earnings call Meta Platforms subsequently filed a certificate with the State of Delaware to legally change its name

### Sample wikirevision_0111

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: As a result, Facebook is now owned by Meta Platforms, which also operates Instagram, WhatsApp other subsidiaries

### Sample wikirevision_0112

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: The current President of Indonesia is Prabowo Subianto, who took office on 20 October 2024

### Sample wikirevision_0112

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: He is the eighth president of Indonesia and serves as both head of state and head of government

### Sample wikirevision_0112

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of the President of Indonesia page, as well as his own Wikipedia article

### Sample wikirevision_0115

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The current Leader of the Conservative Party in the UK is Kemi Badenoch, who was elected to the position on 2 November 2024, following her victory against Robert Jenrick in the party's 2024 leadership election

### Sample wikirevision_0115

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: This is confirmed by the official Wikipedia entries on both the Conservative Party leadership and the Conservative Party in the House of Lords, as well as corroborated by the Guardian's coverage of the 2024 leadership result

### Sample wikirevision_0119

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Jannik Sinner

### Sample wikirevision_0120

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: The current French Open men's singles champion is Carlos Alcaraz, who defended his title by defeating world No. 1 Jannik Sinner in the 2025 final, marking his second French Open title and fifth major overall

### Sample wikirevision_0120

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This result is corroborated across multiple sources, with the 2026 edition page also listing Alcaraz as the current singles champion despite his withdrawal from that tournament

### Sample wikirevision_0121

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The current President of Argentina is Javier Milei, who took office on 10 December 2023

### Sample wikirevision_0121

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: He is the incumbent President of the Argentine Nation, serving as both head of state and head of government

### Sample wikirevision_0121

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple up-to-date sources, including the Wikipedia article on the President of Argentina, which also notes that the country has had many different types of heads of state throughout its history

### Sample wikirevision_0123

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The current US Open men's singles champion is Carlos Alcaraz, who defeated Jannik Sinner in the 2025 US Open final

### Sample wikirevision_0123

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The 2025 US Open was the 145th edition of the tournament, held at the USTA Billie Jean King National Tennis Center in New York City

### Sample wikirevision_0123

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: 2025 is the most recent US Open, with 2026 listed as the upcoming (but not yet completed) edition

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The current President of Germany is Frank-Walter Steinmeier, who has been in office since 19 March 2017

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: He is the Federal President of the Federal Republic of Germany, serving as the country's head of state, with Bellevue Palace in Berlin as his official residence

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: This is consistent across multiple sources, including the current Wikipedia revision of the President of Germany article, which confirms his incumbency from 19 March 2017

### Sample wikirevision_0125

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The latest Prime Minister of Australia is Anthony Albanese, who has served in office since 23 May 2022

### Sample wikirevision_0125

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: He is the 31st person to hold the office since its creation in 1901

### Sample wikirevision_0129

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Madras is officially called Chennai

### Sample wikirevision_0129

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Chennai is the capital and largest city of Tamil Nadu, the southernmost state of India the city is coterminous with Chennai district, which together forms the Chennai Metropolitan Area

### Sample wikirevision_0129

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Historically, the region was part of the Madras Presidency, which was officially called the Presidency of Fort St. George until 1937 and later the Madras Province, but this refers to an administrative division, not the current official city name, which remains Chennai

### Sample wikirevision_0132

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: The current Prime Minister of Japan is Sanae Takaichi, who assumed office on 21 October 2025

### Sample wikirevision_0132

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: She is the country's first female prime minister and has held the position continuously since then

### Sample wikirevision_0132

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of the Prime Minister of Japan page, as well as the list of Japanese prime ministers

### Sample wikirevision_0134

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The current Prime Minister of Australia is Anthony Albanese, who has served in office since 23 May 2022

### Sample wikirevision_0134

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: He is the 31st person to hold the office since its creation in 1901

### Sample wikirevision_0135

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Jannik Sinner

### Sample wikirevision_0137

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Calcutta was officially called Calcutta until 2001, when the city officially changed its name to Kolkata

### Sample wikirevision_0137

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This change was confirmed by the Calcutta High Court in 2001 since then Kolkata has been the official name of the city

### Sample wikirevision_0137

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: As a result, Calcutta is no longer the official name of the city, which is now officially called Kolkata

### Sample wikirevision_0141

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The latest Wimbledon men's singles champion is Jannik Sinner, who won the 2025 Wimbledon Championships

### Sample wikirevision_0141

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: This is confirmed by the official Wikipedia entries on the Wimbledon Championships, which list Sinner as the current men's singles champion with the 2025 edition being the most recent completed tournament

### Sample wikirevision_0141

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Sinner is also listed as the current men's singles holder, indicating that no newer champion has superseded his 2025 title

### Sample wikirevision_0142

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: The latest Vice President of the United States is JD Vance, who assumed office on January 20, 2025

### Sample wikirevision_0142

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: This is confirmed across multiple authoritative sources, including the newer Wikipedia revision and the list of vice presidents of the United States

### Sample wikirevision_0142

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The Office of the Vice President further corroborates his tenure, noting that it is headed by the chief of staff to Vice President JD Vance

### Sample wikirevision_0145

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Emmanuel Macron is the current President of France, having been reelected in 2024 and taking office on 14 May 2017 for his second term

### Sample wikirevision_0145

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: He is the 25th holder of the office, succeeding François Hollande and being reelected ahead of his Socialist rival Marine Le Pen

### Sample wikirevision_0145

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across all available sources, including the high-credibility Wikipedia articles on both the President of France and the 2024 French presidential election

### Sample wikirevision_0149

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: The latest President of the Philippines is Bongbong Marcos, who assumed office on June 30, 2022

### Sample wikirevision_0149

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: He is the 17th and current President, succeeding Rodrigo Duterte

### Sample wikirevision_0149

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This is consistent across multiple high-credibility sources, including the Wikipedia article on the President of the Philippines, which also notes that the incumbent Senate President as of May 2026 is Alan Peter Cayetano

### Sample wikirevision_0150

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The latest US Open men's singles champion is Carlos Alcaraz, who defeated Jannik Sinner in the 2025 US Open final

### Sample wikirevision_0150

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The 2025 US Open was the 145th edition of the tournament, held at the USTA Billie Jean King National Tennis Center in New York City

### Sample wikirevision_0150

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: This result is corroborated by the Wikipedia pages on the US Open, which identify the 2025 edition as the most recent completed tournament

### Sample wikirevision_0151

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Australia

### Sample wikirevision_0152

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: The latest Ballon d'Or winner is Ousmane Dembélé, who claimed the 2025 award at the 69th ceremony in September 2025

### Sample wikirevision_0152

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: This is confirmed by the Wikipedia entry on the Ballon d'Or, which lists Dembélé as the holder of the 2025 award and notes that the 2026 ceremony is next

### Sample wikirevision_0152

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: While the 2024 Ballon d'Or had previously recognized Rodri as the top male player, that result has since been superseded by Dembélé's 2025 victory

### Sample wikirevision_0153

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The latest President of Germany is Frank-Walter Steinmeier, who has been in office since 19 March 2017

### Sample wikirevision_0153

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: He is the Federal President of the Federal Republic of Germany, serving as the country's head of state, with Bellevue Palace in Berlin as his official residence

### Sample wikirevision_0153

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: This is consistent across multiple sources, including the current Wikipedia revision of the President of Germany article, which confirms his incumbency from 19 March 2017

### Sample wikirevision_0154

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: The latest President of Mexico is Claudia Sheinbaum Pardo, who took office on 1 October 2024, making her the 66th President of Mexico

### Sample wikirevision_0154

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: She is the first woman and the first Jewish person to hold the office, serving as President until 2030

### Sample wikirevision_0154

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: This is consistent across multiple sources, including the high-credibility Wikipedia article on the President of Mexico, which also confirms her incumbency from 1 October 2024

### Sample wikirevision_0155

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Facebook's parent company is Meta Platforms, Inc. This was confirmed when Facebook rebranded itself as Meta Platforms, Inc. in October 2021, officially changing its corporate identity

### Sample wikirevision_0155

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The rebranding was announced during the company's Connect conference and was intended to reflect a strategic shift toward building the metaverse—an interconnected digital ecosystem spanning virtual and augmented reality technologies

### Sample wikirevision_0155

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: As a result of this rebranding, Facebook is no longer its own parent company but is instead owned by Meta Platforms, which also operates Instagram, WhatsApp other subsidiaries

### Sample wikirevision_0156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: The current President of the Philippines is Bongbong Marcos, who assumed office on June 30, 2022

### Sample wikirevision_0156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: He is the 17th President of the Philippines and serves as both head of state and head of government

### Sample wikirevision_0156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of the President of the Philippines page, as well as the list of presidents of the Philippines

### Sample wikirevision_0157

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The current President of India is Droupadi Murmu, who has held the office since July 2022

### Sample wikirevision_0157

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: She is the 15th President of India and succeeded Ram Nath Kovind

### Sample wikirevision_0157

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple high-credibility sources, including the newer Wikipedia revision of the President of India article

### Sample wikirevision_0160

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: The current President of Indonesia is Prabowo Subianto, who took office on 20 October 2024

### Sample wikirevision_0160

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: He is the eighth president of Indonesia and serves a five-year term

### Sample wikirevision_0160

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of the President of Indonesia page, as well as his own Wikipedia biography

### Sample wikirevision_0161

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Gurgaon is officially called Gurugram

### Sample wikirevision_0161

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This change was confirmed by the Haryana Government in 2016 the city is now officially known by this name

### Sample wikirevision_0162

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Argentina (defending 2022 champion)

### Sample wikirevision_0165

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: The current President of the United States is Donald Trump, who assumed office on January 20, 2025

### Sample wikirevision_0165

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is confirmed across multiple authoritative sources, including the newer Wikipedia revision of the President of the United States article, which supersedes the older revision

### Sample wikirevision_0165

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: The list of presidents of the United States also explicitly identifies Trump as the incumbent, noting this is his second non-consecutive term in office

### Sample wikirevision_0166

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The current Prime Minister of India is Narendra Modi, who has served in office since 26 May 2014

### Sample wikirevision_0166

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: He is the Honourable Mr. Prime Minister and holds the highest office of the Government of India, being appointed by the President and responsible to the Lok Sabha

### Sample wikirevision_0167

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: The current President of Mexico is Claudia Sheinbaum, who took office on 1 October 2024, making her the 66th President of Mexico

### Sample wikirevision_0167

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: She is the first woman and first Jewish person to hold the office, serving a non-renewable sexenio (six-year term) established under the Constitution of Mexico

### Sample wikirevision_0167

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of the President of Mexico article, as well as her own Wikipedia biography

### Sample wikirevision_0170

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The 2025 French Open men's singles champion was Carlos Alcaraz, who defeated Jannik Sinner in the final to win his second French Open title

### Sample wikirevision_0170

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: This result is corroborated across multiple sources, including the Wikipedia article on the 2025 French Open and the main French Open article listing him as the current champion

### Sample wikirevision_0170

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: However, it is worth noting that the 2026 French Open took place with Alcaraz as the defending champion the query asks for the 'current' champion — if the tournament has been played and a new champion crowned, that would be the most up-to-date answer

### Sample wikirevision_0171

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The current Australian Open men's singles champion is Carlos Alcaraz, who defeated Novak Djokovic in the 2026 final to win his first Australian Open title and seventh major overall

### Sample wikirevision_0171

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The 2026 Australian Open was the 114th edition of the tournament, held at Melbourne Park from 18 January to 1 February 2026, with Jannik Sinner as the defending champion who lost in the semifinals

### Sample wikirevision_0171

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: 2026 is the most recent Australian Open held Alcaraz's victory there makes him the current champion

### Sample wikirevision_0172

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The latest French Open men's singles champion is Carlos Alcaraz, who defended his title in 2025 by defeating world No. 1 Jannik Sinner in the final

### Sample wikirevision_0172

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This result is corroborated across multiple sources, including the high-credibility Wikipedia article on the 2025 French Open , which also notes that Alcaraz subsequently withdrew from the 2026 tournament due to a wrist injury


================================================================================

*Report generated by CATS v2.0*
