# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 127 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.970 (over 736 samples)

**GR F1** *(used in CATS)*: 0.982

**Behavior Adherence**: 0.816 (over 609 applicable samples)

**Factual Grounding**: 0.004 (over 609 applicable samples)

**Single-Truth Recall**: 0.772 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.643

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.982
- **Precision**: 0.998
- **Recall**: 0.965
- **Accuracy**: 0.970
- TP=587, FP=1, FN=21, TN=127

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.858
- **Abstain Recall**: 0.992
- **Abstain F1**: 0.920
- **Specificity**: 0.965
- Abstain TP=127, FP=21, FN=1, TN=587


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (56 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.986
- **GR F1** *(used in CATS)*: 0.990
- **Behavior**: 0.948 (n=155)
- **Grounding**: 0.006 (n=155)
- **Recall**: 0.854 (n=154)
- **CATS**: 0.700

### Type 2: Complementary Info

- **Samples**: 221 (45 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.959
- **GR F1** *(used in CATS)*: 0.974
- **Behavior**: 0.943 (n=176)
- **Grounding**: 0.009 (n=176)
- **Recall**: 0.715 (n=156)
- **CATS**: 0.660

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.972
- **GR F1** *(used in CATS)*: 0.984
- **Behavior**: 0.583 (n=96)
- **Grounding**: 0.000 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.522

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.987
- **GR F1** *(used in CATS)*: 0.993
- **Behavior**: 0.662 (n=145)
- **Grounding**: 0.000 (n=145)
- **Recall**: 0.743 (n=140)
- **CATS**: 0.599

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.865
- **GR F1** *(used in CATS)*: 0.928
- **Behavior**: 0.865 (n=37)
- **Grounding**: 0.000 (n=37)
- **Recall**: 0.784 (n=37)
- **CATS**: 0.644


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2118

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
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: Nematodes can increase soil fertility indirectly through nutrient cycling, but the effect depends heavily on nematode type and functional group — not all nematodes are beneficial

### Sample conflictingqa_05b33f4ca156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: The answer depends on the salamander species; most are poisonous to touch, but a few are considered safe to handle gently

### Sample conflictingqa_060e5f26c453

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Yes — the Great Pacific Garbage Patch is larger than Texas

### Sample conflictingqa_0717d0e62f3b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Fashion designs can receive copyright protection, but only for specific design elements rather than the clothing items themselves

### Sample conflictingqa_0717d0e62f3b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Graphic designs, textile patterns logos on garments are generally protected under U.S. copyright law when they possess a sufficient degree of creativity and can be identified separately from the functional aspects of the clothing

### Sample conflictingqa_0717d0e62f3b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: However, copyright does not extend to colors, fashion sketches the overall configuration or shape of a garment apparel is broadly classified as a functional product in most countries, which further limits copyright protection

### Sample conflictingqa_0717d0e62f3b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: In cases where full protection is sought, designers may explore alternative intellectual property frameworks, such as trademark law for brand elements or industrial design rights for the clothing's form

### Sample conflictingqa_07e290a954ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The evidence on St. John's wort is mixed and depends on the severity of depression

### Sample conflictingqa_07e290a954ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: Multiple clinical studies, including a systematic review cited by the Psychopharmacology Institute, suggest that St. John's wort is more effective than placebo and approximately equivalent to low-dose tricyclic antidepressants or standard SSRIs, particularly for mild to moderate depression

### Sample conflictingqa_07e290a954ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: A peer-reviewed study on patients with mild-to-moderate depression found comparable efficacy and safety when compared to SSRIs another source notes that clinical evidence supports its use specifically for mild depression

### Sample conflictingqa_07e290a954ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: However, WebMD and other sources point to two large studies—including one sponsored by the NCCAM—that failed to find St. John's wort more effective than placebo for moderately severe major depression user surveys cited by the Black Dog Institute similarly rated it as less effective than conventional antidepressants for clinical depression overall

### Sample conflictingqa_07e290a954ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: In summary, St. John's wort may be effective for mild to moderate depression, but the evidence does not support its use as a treatment for severe depression

### Sample conflictingqa_08757808c000

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: Weight lifting does not cause high blood pressure; rather, it may reduce blood pressure over time

### Sample conflictingqa_08757808c000

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5
- **Claim**: Research indicates that long-term strength training (more than eight weeks) can lower blood pressure and improve cardiovascular health, while the temporary spike during lifting is considered normal and not dangerous for most healthy individuals

### Sample conflictingqa_08757808c000

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5
- **Claim**: However, people with preexisting hypertension or cardiovascular risks should exercise caution, as blood pressure can rise temporarily during heavy lifts — sometimes exceeding 200/100 mmHg — and modifying techniques such as lifting lighter loads or avoiding the Valsalva maneuver may be advisable

### Sample conflictingqa_0875b5f3262a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1
- **Claim**: No, Allen Ginsberg's *Howl* was not judged to be obscene

### Sample conflictingqa_0875b5f3262a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: The U.S. The U.S. Customs seized the first shipment of the book in 1956 in 1957 Judge Clayton Horn ruled that the work was not obscene because it had 'redeeming social value.' This ruling set an important precedent for literary expression and remains widely cited

### Sample conflictingqa_0a05aabca56a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: Yes — anime is a specific type of cartoon originating in Japan, while cartoons include animations from many countries and styles

### Sample conflictingqa_0ad05303220b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Judaism is a religion, not a race; anyone can become Jewish by converting the Jewish people are better described as an ethnoreligion — a nation or tribe that shares a common land, history religious beliefs — rather than a genetic race

### Sample conflictingqa_0c3c7b487766

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: Yes, iodine supplementation can cause thyroid problems — specifically hyperthyroidism, hypothyroidism, goiter autoimmune thyroiditis — particularly in susceptible individuals such as those with preexisting thyroid disease, the elderly pregnant women; excess iodine intake may also disrupt thyroid homeostasis even in healthy adults

### Sample conflictingqa_0dba017da71c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Yes — the world's largest known organism is a fungus, specifically *Armillaria solidipes* (Honey Fungus), which scientists discovered in the Pacific Northwest spanning approximately 5.5 kilometres

### Sample conflictingqa_0dba017da71c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This underground network is estimated to be over 2000 years old and covers roughly 2,384 acres, surpassing the size of notable contenders like the Pando Aspen tree colony in Utah

### Sample conflictingqa_0dba017da71c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: This claim, with some citing *Armillaria ostoyae* (also known as the 'Humongous Fungus') as the record holder, but the underlying consensus remains the same: the world's largest organism is indeed a fungus

### Sample conflictingqa_111bef268aa6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Yes — apple peels contain far more fiber and antioxidants than the flesh peeling removes those specific nutrients; however, peeling does not reduce the apple's vitamin C content, which is found both in the flesh and the skin

### Sample conflictingqa_114c06976f62

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The Church of the Flying Spaghetti Monster is a subject of genuine disagreement

### Sample conflictingqa_114c06976f62

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Some sources emphasize its legal recognition as a religion in several countries and its organized structure, while others characterize it primarily as a satirical protest against intelligent design or a social movement lacking sincere religious belief

### Sample conflictingqa_11c5ef7c4545

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The retrieved evidence presents competing views

### Sample conflictingqa_11c5ef7c4545

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Some sources argue that anyone can become an entrepreneur if they are willing to learn, work hard face risks , while others contend that entrepreneurship is not for everyone because it requires rare skills, mindset risk tolerance — characteristics that not all individuals possess

### Sample conflictingqa_11c5ef7c4545

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: The debate thus hinges on whether entrepreneurship is a universal opportunity or a specialized calling reserved for those with particular aptitudes

### Sample conflictingqa_151865dc414b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: Yes, pulsatile tinnitus can often be successfully treated and cured once its underlying cause is identified

### Sample conflictingqa_151865dc414b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: The Institute for Neurology and Neurosurgery at Northwell Health confirms that conditions such as venous sinus stenosis, tumors, arteriovenous malformations high blood pressure — all common causes of pulsatile tinnitus — can be addressed with treatments ranging from venous sinus stenting to surgery to medication resolving the cause typically eliminates the tinnitus symptom

### Sample conflictingqa_151865dc414b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: However, the degree of relief depends heavily on the specific cause: some cases — such as those rooted in anemia, hyperthyroidism venous stenosis — respond well to focused treatment, while others where no curable cause is found may require management through sound therapy, hearing aids lifestyle changes

### Sample conflictingqa_1d741779a31e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: The evidence is mixed

### Sample conflictingqa_1d741779a31e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Some sources indicate that artificial sweeteners are safe for diabetics and may even help improve blood sugar control, while others suggest that certain artificial sweeteners could worsen glycemic control, alter the gut microbiome potentially increase the risk of type 2 diabetes and cardiovascular disease

### Sample conflictingqa_1d741779a31e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: The FDA has generally recognized artificial sweeteners as safe when consumed within acceptable daily limits, but some researchers argue that long-term use deserves further study the optimal use of artificial sweeteners in diabetes management remains a subject of ongoing scientific debate

### Sample conflictingqa_1e9c6cd8e30b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: Yes — palm oil production causes serious environmental harm through deforestation, loss of biodiversity pollution; however, some sources note that sustainably-produced palm oil may mitigate these impacts

### Sample conflictingqa_21f33954c8af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Dog breeding is not universally considered unethical, but opinions and research outcomes are divided

### Sample conflictingqa_21f33954c8af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Some argue that breeding is unethical because it contributes to overpopulation, exploits dogs' reproductive systems causes inherited health problems; others argue that breeding is not inherently unethical if regulations are improved and awareness is spread

### Sample conflictingqa_220ec09fbb2c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: Cows are often described as having four stomachs, but technically they have one stomach divided into four distinct compartments: the rumen, reticulum, omasum abomasum

### Sample conflictingqa_237adb87065f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The retrieved evidence indicates that the Silurian period marks an important milestone in the evolution of land plants, though not necessarily the absolute birth of the first land plants

### Sample conflictingqa_237adb87065f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: Several sources identify the Silurian as the debut of the first confirmed land plants, with Cooksonia described as the most famous of these early pioneers simple vascular plants emerging alongside moss forests during this period ; one source even frames the Silurian as the first period with fossils of extensive non-microscopic life on land

### Sample conflictingqa_237adb87065f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: However, high-credibility research pushes the origins back further, noting that the earliest radiation of land plants (embryophytes) actually began in the Middle Ordovician, making the Silurian a continuation rather than the birth of land plant evolution

### Sample conflictingqa_237adb87065f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: In summary, while the Silurian period is correctly celebrated for its rich fossil record of early land plants, it represents a peak of terrestrialization that built upon earlier Ordovician developments rather than marking the very first appearance of land plants

### Sample conflictingqa_2395695f1604

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The scientific evidence on whether dairy products increase mucus production is conflicting

### Sample conflictingqa_2395695f1604

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5
- **Claim**: Some sources report that excessive milk consumption has been associated with increased respiratory tract mucus production and asthma a 2004 study cited by the American Review of Respiratory Disease found that 58.5% of parents believed milk increases mucus ; however, a 2012 study by BC Children's Hospital stated that 'studies have not been able to provide a definitive link' between milk and increased mucus production a 2005 review published in the Journal of the American College of Nutrition concluded that milk consumption does not lead to mucus production or asthma occurrence

### Sample conflictingqa_2395695f1604

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: Overall, while dairy products may alter the sensation or viscosity of existing mucus, the predominant clinical consensus is that they do not actually increase mucus production the perceived coating effect in the mouth and throat is due to oral enzymes interacting with milk rather than true mucus formation

### Sample conflictingqa_24c25ef3a801

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: The retrieved evidence is mixed

### Sample conflictingqa_24c25ef3a801

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: Some sources argue that money can buy happiness, but usually only up to a point — for example, one source says money buys happiness only up to an annual income of about $75,000–$100,000, while others say money can buy happiness if spent strategically on experiences and other people

### Sample conflictingqa_24fa0020a521

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: Most healthy children do not need multivitamins if they are growing normally and eating a varied, well-balanced diet

### Sample conflictingqa_24fa0020a521

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5
- **Claim**: The American Academy of Pediatrics (AAP) does not recommend a daily multivitamin for children eating a well-rounded diet, cautioning that supplements should never replace actual food intake

### Sample conflictingqa_24fa0020a521

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: However, certain children may benefit from supplementation — for example, vitamin D drops for exclusively breastfed infants, iron for infants born prematurely or at low birth weight vitamin B12 for those on strict vegan diets

### Sample conflictingqa_2500bfd10608

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: he evidence is mixed

### Sample conflictingqa_2500bfd10608

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Some research links fluoride exposure to neurobehavioral problems and lowered IQ in children high levels of fluoride can cause fluorosis and concentrate in bone; however, fluoride is generally considered safe at the 0.7 mg/L concentration used in U.S. drinking water the CDC and AAP continue to support water fluoridation as a public health benefit

### Sample conflictingqa_264e97231363

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1
- **Claim**: The retrieved evidence consistently states that chlorine is not the direct cause of green hair; rather, copper (often from algaecides or tap water) oxidizes and attaches to hair, turning it green

### Sample conflictingqa_287debe5bb90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Philosophers and researchers hold conflicting views on whether we can know anything beyond our minds

### Sample conflictingqa_287debe5bb90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Some argue that formal systems run into fundamental limitations when applied to themselves, making it impossible for thought alone to fully grasp the nature of mind , while others contend that true understanding requires transcending conceptual reasoning and becoming aware of deeper, non-thought-based mental processes

### Sample conflictingqa_287debe5bb90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: Conversely, some sources suggest that genuine knowledge of the external world may be possible by quieting mental noise and opening oneself to direct experience by considering theories such as transparency, which propose that self-knowledge can be gained by observing external phenomena rather than relying solely on introspection

### Sample conflictingqa_287debe5bb90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: However, these perspectives remain debated, with some philosophers questioning whether any mind exists externally at all or whether the concept of 'knowing beyond the mind' is coherent given the inherent limitations of human cognition

### Sample conflictingqa_288cd1b45aab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Yes

### Sample conflictingqa_29f69e16a0c3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Yes, flowers do communicate with bees through multiple channels including electric fields, fragrances, colors possibly sound-induced nectar changes — though the exact mechanisms and evolutionary purposes are still under investigation

### Sample conflictingqa_2c0ea18839df

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Yes

### Sample conflictingqa_311fca0928d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The retrieved evidence presents conflicting views

### Sample conflictingqa_311fca0928d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Some sources argue that IPv6 is not fundamentally more secure than IPv4 — for example, IPv6's main security mechanism is IPsec, which can also be used with IPv4 but is not mandatory the absence of NAT in IPv6 does not meaningfully improve security ; moreover, security incidents with IPv6 protocols still arise primarily from design and implementation issues rather than the protocol itself

### Sample conflictingqa_311fca0928d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: On the other hand, other sources argue that IPv6 does offer concrete security advantages over IPv4, such as built-in IPSec support, improved data integrity a larger address space that can make scanning attacks more difficult

### Sample conflictingqa_311fca0928d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Overall, whether IPv6 is fundamentally more secure than IPv4 depends significantly on how the protocol is configured and maintained, as well as on which specific security dimensions are being compared

### Sample conflictingqa_34610226ee3f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: No

### Sample conflictingqa_34fef928d452

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: Yes — Archaeopteryx was capable of powered flight, according to a 2025 study that confirmed the animal could generate lift and fly short distances

### Sample conflictingqa_35491baf4f4b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Yes, the moon does have an atmosphere, though it is very thin and technically classified as an exosphere rather than a conventional atmosphere

### Sample conflictingqa_35491baf4f4b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: This exosphere consists of a tenuous mixture of gases including helium, argon, neon, ammonia, methane, carbon dioxide some sodium, potassium rubidium was confirmed during the Apollo missions of the 1960s and 1970s

### Sample conflictingqa_35491baf4f4b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Additionally, NASA research has shown that the moon once possessed a thicker, transient atmosphere approximately 3 to 4 billion years ago, formed when intense volcanic eruptions released gases faster than they could escape to space, with this ancient atmosphere persisting for about 70 million years before being lost

### Sample conflictingqa_3601f7480501

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Research and expert opinion present conflicting findings on unlimited vacation time

### Sample conflictingqa_3601f7480501

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Some sources argue it reduces stress, improves productivity enhances job satisfaction , while others note that employees often take fewer days than under traditional accrual systems and may feel pressure to limit their time off, potentially leading to burnout

### Sample conflictingqa_3601f7480501

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: Additional factors such as communication practices, managerial oversight individual workload further complicate the picture, suggesting that unlimited PTO's effectiveness varies significantly by context and implementation

### Sample conflictingqa_37ab7146eb9b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: Robots can be programmed to detect and respond to stimuli analogous to pain, but most researchers believe this simulation falls short of actual feeling

### Sample conflictingqa_37ebad668bb7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: The retrieved evidence consistently indicates that data is nearly always required for machine learning, as models depend on training examples to generalize and make accurate predictions

### Sample conflictingqa_37ebad668bb7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: However, some sources introduce important nuances: certain ML approaches can perform adequately with smaller, structured datasets the specific volume needed may vary significantly based on the algorithm's complexity and the problem's requirements

### Sample conflictingqa_37ebad668bb7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Overall, while data is not strictly required in all theoretical scenarios, practical machine learning almost always involves some form of training data

### Sample conflictingqa_39fe5c441657

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The retrieved evidence presents competing views

### Sample conflictingqa_39fe5c441657

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: Some sources argue that astral projection is a real experience — vivid, frequently reported associated with measurable brain activity — while others argue that it is better explained as a type of lucid dream or hallucination and lacks physical evidence

### Sample conflictingqa_3afd7f725cb4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Opinions differ — 41% of adults surveyed by NPR-Ipsos do not consider audiobooks real reading, while others argue they are equally legitimate

### Sample conflictingqa_3bd13d25098b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The Moon is generally considered geologically inactive compared to Earth, but there is emerging evidence that it may still host limited recent activity

### Sample conflictingqa_3c835387fe6d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Yes, though indirectly: the Komodo dragon evolved in Australia and lived there until around 300,000 years ago before dispersing to Indonesia, where it survives today

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: It depends on whether the artificial tree is used for approximately 20 years or more

### Sample conflictingqa_3f3c3399259a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1
- **Claim**: The relationship between fish oil and heart disease risk is genuinely contested in the evidence

### Sample conflictingqa_3f3c3399259a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Some clinical trials and expert statements argue that fish oil supplements do not significantly reduce the risk of heart attack or stroke and that the benefit of reducing triglycerides comes with trade-offs such as increased atrial fibrillation at high doses

### Sample conflictingqa_3f3c3399259a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: On the other hand, some systematic reviews and observational research note that fish oil may still have potential cardiovascular benefits, particularly for specific high-risk conditions, though individual study results remain inconsistent

### Sample conflictingqa_411445406724

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Cycads were abundant and diverse during the Mesozoic era and are often called the 'age of cycads,' but they did not fully dominate the plant kingdom; flowering plants eventually replaced them as ecologically dominant land species

### Sample conflictingqa_42d60ecaee9f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Emoji are not currently considered a new form of language by most linguists, who note that they lack a formal grammar, fail tests for mutual intelligibility function more as visual supplements to existing languages

### Sample conflictingqa_42d60ecaee9f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: However, some researchers argue that emoji are reviving ancient visual communication traditions—such as hieroglyphs—and may eventually evolve into a more complete language system, particularly as they gain wider adoption and standardization

### Sample conflictingqa_4317242e485c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Yes — the IUCN's 2016 report cited by the NRA Hunters' Leadership Forum describes trophy hunting as 'the most effective way to save wildlife populations,' while d3 (Royal Society Publishing) notes it can provide benefits to wildlife conservation; however, d1 (Conservation Frontlines) cautions that a ban may currently be detrimental and calls for reform rather than abolition d5 (Discover Wildlife/Oxford WildCRU) warns that blanket bans could increase poaching

### Sample conflictingqa_4786f87b62be

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The retrieved evidence presents competing views

### Sample conflictingqa_4786f87b62be

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Some researchers argue that the gender wage gap is real and measurable, while others argue that it is largely explained by factors like occupation and parenting choices and therefore does not reflect discrimination

### Sample conflictingqa_517b918aa677

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The constitutional status of prayer in U.S. public schools is nuanced and contested

### Sample conflictingqa_517b918aa677

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The Supreme Court has ruled that school-led or endorsed prayer — including participation by faculty and staff — is unconstitutional under the Establishment Clause, as it creates a coercive religious environment even when described as 'voluntary'; by contrast, the First Amendment also protects students' right to engage in personal, non-disruptive religious expression or prayer on school property federal guidance requires schools to adopt a stance of religious neutrality

### Sample conflictingqa_52181cd092aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The query's claim that the trash island (Great Pacific Garbage Patch) is as large as Texas is true, though some sources exaggerate the size

### Sample conflictingqa_5233eab573e5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Yes — there are more tigers kept as pets than in the wild

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
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The evidence on whether bicarbonate supplementation prevents chronic kidney disease (CKD) progression is mixed and depends on disease stage and dose

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Some studies suggest a benefit: a prospective study found that sodium bicarbonate slowed the rate of creatinine clearance decline in stage 4 CKD the KDIGO 2024 guidelines recommend oral sodium bicarbonate to normalize blood bicarbonate levels when serum bicarbonate is below 18 mEq/L ; additionally, a peer-reviewed study noted that bicarbonate supplementation reduced markers of fibrosis and may preserve eGFR in early CKD stages

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: However, other research has produced negative or inconclusive results: a randomized trial found no effect of bicarbonate on kidney failure progression after a mean follow-up of 1.35 years while supplementation may reduce urinary TGF-β in early stages, a low dose (0.5 mEq/kg/day) did not significantly reduce it in advanced diabetic CKD with normal bicarbonate levels

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: Overall, the data support a conditional role for bicarbonate in earlier stages of CKD—particularly when acidosis is present—but the evidence does not support universal supplementation in all CKD patients the optimal dose and population remain subjects of ongoing investigation

### Sample conflictingqa_56fd6bf22253

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Adenoids can grow back after removal, although it is generally considered uncommon and rarely causes significant problems

### Sample conflictingqa_56fd6bf22253

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: The 2000 study cited in d2 found that adenoids rarely regrow enough to cause nasal obstruction symptoms, with most patients showing either no or only trace amounts of adenoidal tissue on follow-up examination a 2009 PubMed study similarly noted that adenoidal regrowth occurred in only 19.1% of cases and did not manifest clinically

### Sample conflictingqa_56fd6bf22253

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: However, some sources indicate that regrowth is possible under certain conditions — such as when surgery is performed at a very early age or if small portions of tissue are left behind — and some patients may experience partial regrowth without necessarily experiencing symptoms

### Sample conflictingqa_57190bca6f7a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: The 1815 Mount Tambora eruption was the most powerful and destructive volcanic event in recorded history , but it is not explicitly confirmed as the deadliest in terms of total casualties

### Sample conflictingqa_5e7a6a2debfd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: Male bees do not perform active tasks like gathering nectar or maintaining the hive; they are drones whose primary role is to mate with the queen, though they are often expelled from the colony before winter

### Sample conflictingqa_613a0093714b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The phrase is popularly associated with 17th century England, but scholars and experts disagree on its precise origin

### Sample conflictingqa_62b1aff6586d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: The ozone layer is healing, but the process is gradual and not yet complete

### Sample conflictingqa_62b1aff6586d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: A 2025 MIT-led study confirmed with 95% statistical confidence that the Antarctic ozone layer is recovering as a direct result of global reductions in ozone-depleting substances, making it one of the most successful environmental remediation efforts in history

### Sample conflictingqa_62b1aff6586d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: However, some sources note that a hole still exists over New Zealand the healing is further complicated by factors such as rocket launches, illustrating that full restoration will take time

### Sample conflictingqa_63fde268aa8c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: Religious and philosophical traditions such as Sanatana Dharma and Cartesian dualism assert that the mind is separate from the body, while contemporary embodied cognition and neuroscience argue that the mind-body distinction is a fiction — that thoughts, sensations movements arise from the same psychobiology and are inseparably interconnected

### Sample conflictingqa_63fde268aa8c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Scientifically, there is currently no evidence supporting the existence of any aspect of an individual separate from its body, though the mind-body problem remains an active philosophical dilemma without a definitive resolution

### Sample conflictingqa_676188bf8139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: Yes — the Lantern Festival does honor and commemorate deceased ancestors, though it also has broader symbolism and multiple competing origin theories

### Sample conflictingqa_6988dd820a61

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The retrieved evidence is conflicting

### Sample conflictingqa_6988dd820a61

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1
- **Claim**: Some researchers, including a Nature-published study and a Scientific American-reported University of Tokyo study, argue that full moons may increase the probability of large earthquakes through tidal stress mechanisms

### Sample conflictingqa_6988dd820a61

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Others, including a USGS researcher who analyzed 204 magnitude 8+ earthquakes, found that earthquake incidence showed no relationship to lunar phases and described the data as 'completely random.' Currently, the scientific consensus remains divided on whether full moons represent a meaningful trigger for earthquake activity

### Sample conflictingqa_6d7279b0a8ed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: No, the Gutenberg Bible was not the first book printed with movable type

### Sample conflictingqa_6d7279b0a8ed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: While it is recognized as the earliest major European book printed using mass-produced metal movable type and revolutionized printing in the West, it was preceded by the Jikji — a Korean Buddhist text printed in 1377 using wooden movable type — which predates the Gutenberg Bible by approximately 78 years

### Sample conflictingqa_6d7279b0a8ed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: Furthermore, Chinese and Korean inventors had been producing printed books using movable type for centuries before Gutenberg was even born, making the Gutenberg Bible neither the first printed book nor the first made with movable type

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: The retrieved evidence is conflicting

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: Some sources argue that split ends cannot be permanently repaired because hair is dead tissue and the damage cannot be regeneratively healed, though products containing proteins, quaterniums acidic bonding agents may temporarily mask the damage by coating the cuticle or holding fibers together

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Other sources argue that certain bond-building products can repair the chemical bonds in hair that are broken by damage, making the repair permanent rather than temporary

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The difference is that true repair requires restoring the disulfide bonds in hair, which most commercial products cannot do

### Sample conflictingqa_6fe31cd2ef65

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Whether rolling /r/ in Spanish is strictly necessary depends on the context: rolling is required for words with double RR and for single R at the beginning of a word or after certain consonants, but a softer tap (alveolar tap) is used for single R in the middle of words

### Sample conflictingqa_6fe31cd2ef65

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: While native speakers and some linguists consider the distinction important, fluent communication is still possible without perfect rolling, especially for single middle Rs the severity of the requirement varies by context and speaker preference

### Sample conflictingqa_734859937e46

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: It depends on jurisdiction; some ISPs can sell data without consent where laws permit it, while others face state-level restrictions requiring opt-in or opt-out consent

### Sample conflictingqa_73560dfab1ae

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: The evidence on whether high doses of vitamin C alleviate common cold symptoms is mixed and contested

### Sample conflictingqa_73560dfab1ae

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: High-quality sources indicate that vitamin C does not reliably prevent colds, though some research suggests it may modestly reduce cold duration and severity: a peer-reviewed meta-analysis found that vitamin C decreased the severity of common colds by approximately 15% and shortened the duration of severe cold symptoms specifically , while Mayo Clinic experts note that extra vitamin C has not clearly proven preventive value but may potentially speed recovery slightly

### Sample conflictingqa_73560dfab1ae

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: On the other hand, some sources present opposing user opinions and anecdotal claims that vitamin C can significantly reduce cold duration or severity others caution that most people already obtain sufficient vitamin C from their diet and that high-dose supplementation carries potential risks such as increased kidney stone risk — highlighting the ongoing debate between clinical research and broader public beliefs regarding vitamin C's therapeutic benefits

### Sample conflictingqa_747727772a30

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1
- **Claim**: Bees generally avoid flying in the rain, but the answer depends heavily on the type and intensity of precipitation

### Sample conflictingqa_76956c2fba7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The relationship between saturated fats and heart disease risk is genuinely contested in the scientific literature

### Sample conflictingqa_76956c2fba7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Some research, including a British Heart Foundation-funded study, suggests that a diet high in saturated fat can adversely affect cardiovascular disease risk factors such as LDL cholesterol and liver fat, potentially increasing heart disease risk even without weight gain

### Sample conflictingqa_76956c2fba7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5
- **Claim**: However, meta-analyses of both observational studies and randomized controlled trials have failed to consistently confirm a strong association between saturated fat intake and actual heart disease outcomes, with some studies showing no significant link and others reporting mixed results depending on the type of fat used for replacement

### Sample conflictingqa_76956c2fba7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Overall, while saturated fats are widely advised against due to their cholesterol-raising effects, the evidence for their direct causal role in heart disease remains inconclusive and is the subject of ongoing debate

### Sample conflictingqa_7998e59d9b97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Yes

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: The retrieved evidence presents competing views

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: Some sources (e.g., ) assert that the Catholic Church is the one true church by claiming apostolic succession, scripture divine attributes as proof

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Other sources (e.g., ) argue that the Bible alone should determine which church is true that the Catholic Church's claim lacks biblical support

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: This divergence reflects conflicting theological interpretations rather than a single authoritative resolution

### Sample conflictingqa_7cf85109a70d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: No — bronze is more durable than brass

### Sample conflictingqa_80857a692531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: Farmed and wild salmon are broadly similar in nutritional value, with both containing comparable amounts of protein, omega-3 fatty acids vitamins

### Sample conflictingqa_80857a692531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: However, some sources note that wild salmon tends to have higher levels of certain vitamins — such as vitamin D and vitamin A — and lower fat content, while farmed salmon can accumulate higher levels of environmental contaminants such as PCBs ; d1 reports some natural mineral differences in favor of wild salmon

### Sample conflictingqa_80857a692531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Overall, the consensus is that farmed salmon is a valuable and healthy substitute when wild salmon is unavailable or too expensive

### Sample conflictingqa_80baf25496cd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Multiculturalism may hinder unity in cases where deep cultural affiliations prevent assimilation or integration where politicians exploit cultural differences to fuel divisions — a perspective rooted in empirical research showing a correlation between multiculturalism and reduced civic unity

### Sample conflictingqa_80baf25496cd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: However, alternative theoretical and empirical viewpoints contend that multiculturalism can serve as a pathway to unity, particularly within spiritual or civic frameworks that actively embrace diversity and challenge the assumption that cultural differences are insurmountable barriers

### Sample conflictingqa_80baf25496cd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Additional evidence further complicates the picture by highlighting the negative consequences of monocultural approaches, such as ethnocentrism and homogeneity pressures, which may themselves hinder the very unity that multiculturalism is often accused of obstructing

### Sample conflictingqa_8848765fc18a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5
- **Claim**: Spelunking and caving are generally considered the same activity — the exploration of caves — as directly stated by sources that define them interchangeably

### Sample conflictingqa_8848765fc18a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: However, some communities draw a distinction: d2 suggests that 'caving' is used by enthusiasts as a more technically committed version of the hobby, while 'spelunking' is seen as a casual or amateur pursuit, though this distinction is not universally accepted

### Sample conflictingqa_894f4a4b9552

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: The majority of scientific evidence strongly suggests that dark matter exists

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: The retrieved evidence indicates that bird calls are not universally unique to each individual but vary by species and context

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Some birds, such as songbirds, do develop individually recognizable songs through learning, while others are born with innate calls that are not unique to each bird

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Additionally, many species use calls to communicate specific messages—such as alarms—that are understood across unrelated species, further suggesting calls are not exclusively individual identifiers

### Sample conflictingqa_8cf8ebb94554

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Research and clinical opinion on knee brace effectiveness for injury prevention is divided

### Sample conflictingqa_8cf8ebb94554

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Some studies suggest prophylactic knee braces may reduce the risk of certain injuries (such as MCL strain) in contact sports the American Academy of Orthopaedic Surgeons notes they are designed to prevent or reduce the severity of knee injuries in such settings

### Sample conflictingqa_8cf8ebb94554

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: However, the American Academy of Family Physicians concludes there is no conclusive evidence supporting routine use of knee braces for prevention Cleveland Clinic similarly notes that no studies can definitively prove a brace prevented an injury

### Sample conflictingqa_8cf8ebb94554

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: Effectiveness appears to depend on the type of brace, the context of use the specific injury being addressed

### Sample conflictingqa_8efa53ba7c60

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Partially — birds are descended from theropod dinosaurs and T-Rex was a theropod, but T-Rex is not the direct ancestor of modern birds

### Sample conflictingqa_9251c7a33ec5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: he evidence is mixed

### Sample conflictingqa_9251c7a33ec5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Some research suggests that spaying or neutering can increase the risk of certain conditions, such as some cancers, joint disorders urinary tract issues, while other studies conclude that the overall health benefits (such as preventing mammary tumors and prostatic disease) outweigh these risks

### Sample conflictingqa_9251c7a33ec5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The effects depend on multiple factors, including the sex, breed, size age of the animal at the time of surgery the specific outcome of interest

### Sample conflictingqa_9261438d6ee2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Research is divided on whether fish feel pain in the same way humans do

### Sample conflictingqa_9261438d6ee2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5
- **Claim**: Some studies, such as those cited by Australian Museum, show that fish possess nerve receptors (nociceptors) similar to humans and exhibit behavior modification in response to noxious stimuli, leading researchers like Dr. Lynne Sneddon and Victoria Braithwaite to argue that fish do experience pain

### Sample conflictingqa_9261438d6ee2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: However, other scientists note anatomical differences — such as the absence of dense neocortical folds in fish brains — and argue that fish may only respond to noxious stimuli (nociception) rather than experiencing true subjective pain , a view echoed by researcher Dr. J. Rose who states that fish pain perception is very different from human pain

### Sample conflictingqa_9275911a2961

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Yes, certain antacids can cause kidney stones — specifically, antacids containing calcium can lead to calcium kidney stones antacids containing magnesium can contribute to magnesium-ammonium-phosphate stones, particularly with excessive use

### Sample conflictingqa_9275911a2961

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Research has documented a case of kidney stones formed from magnesium-ammonium-phosphate crystals in a patient taking large amounts of magnesium-containing antacid studies also warn that calcium-containing antacids may increase the risk of kidney stones when used in excess or alongside calcium supplements

### Sample conflictingqa_9275911a2961

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: It is worth noting that the risk is generally considered higher with excessive or prolonged use some sources distinguish by subtype, noting that calcium-based antacids are the primary concern for kidney stone formation

### Sample conflictingqa_962d8f5d5574

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: Yes — all snakes are able to swim, though most spend little time in water

### Sample conflictingqa_9b11b8e571aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1
- **Claim**: Gonorrhea is almost always transmitted sexually, but rare non-sexual transmission routes exist

### Sample conflictingqa_9b73cb6cce52

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Giant African Land Snails can make acceptable pets in some countries, but come with significant drawbacks

### Sample conflictingqa_9b73cb6cce52

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: They are legal in the UK and are among the most popular invertebrate pets due to relatively simple care requirements, making them suitable for beginners and educational settings ; however, they are illegal to own in the US because they can spread disease and cause agricultural damage

### Sample conflictingqa_9b73cb6cce52

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Caring for them requires careful management of temperature, humidity diet to prevent health issues owners must also handle them safely to avoid disease transmission

### Sample conflictingqa_9b73cb6cce52

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Some sources note that while they are low-maintenance and long-lived, they may not be ideal for children since they can live up to 10 years and boredom with them is a common reason for rehoming

### Sample conflictingqa_9ceca2645833

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Affirmative Action is not per se reverse discrimination

### Sample conflictingqa_a1e36a8db854

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: The question of whether glyphosate is harmful to humans is genuinely contested

### Sample conflictingqa_a1e36a8db854

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5
- **Claim**: The EPA has concluded that glyphosate does not pose a risk to humans when used according to directions and that it is not likely to be carcinogenic, though the Agency also found no risks of concern for children exposed to glyphosate residues on food ; similarly, Health Canada determined that proper use does not cause harmful effects

### Sample conflictingqa_a1e36a8db854

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: On the other hand, the Seattle Statement—a consensus document from Washington University scientists—concluded that evidence linking glyphosate to cancer, kidney and liver disease, reproductive issues neurological harm is now so strong that regulatory action is urgently justified peer-reviewed research has documented glyphosate crossing the blood-brain barrier and contributing to neuroinflammation

### Sample conflictingqa_a1e36a8db854

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: In summary, while government agencies and some researchers consider glyphosate safe when used as directed, others argue that the evidence of harm is compelling and warrants stricter regulation

### Sample conflictingqa_a25014a5c5b5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Not all plants can survive without any light; many die within weeks of complete darkness

### Sample conflictingqa_a2f06d54b240

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Stalactites can form underwater, as evidenced by a published giant underwater encrusted stalactite found approximately 30 meters below modern sea level in the Blue Hole of Lighthouse Reef Atoll

### Sample conflictingqa_a2f06d54b240

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: However, some sources distinguish between true stalactites, which require dripping water and typically form in dry caves analogous underwater structures such as rimstone dams or marine stalactites that grow via different mechanisms in wet environments

### Sample conflictingqa_a3264277980a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The idea that Orson Welles's 1938 War of the Worlds broadcast caused mass panic is widely questioned by scholars and historians

### Sample conflictingqa_a3264277980a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: While newspapers at the time reported widespread hysteria, later research — including surveys cited by Slate and American University's W. Joseph Campbell — suggests that few listeners actually believed the broadcast was real, that the press exaggerated the reaction to discredit radio as a competing news medium that the notion of tens of thousands fleeing in panic was largely a media-driven myth

### Sample conflictingqa_a3980a2921cf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Yes — hair oil is beneficial for all hair types

### Sample conflictingqa_a507c5b61631

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: Volcanic activity is among the leading hypotheses proposed as a trigger for the Paleocene-Eocene Thermal Maximum (PETM), but the evidence remains inconclusive and contested

### Sample conflictingqa_a507c5b61631

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Some researchers argue that volcanism was dominant in releasing the initial carbon disturbance, pointing to isotopic evidence and mercury proxies as support , while others emphasize that the PETM onset coincides with a mercury low, suggesting at least one additional carbon reservoir was involved in the initial warming

### Sample conflictingqa_a507c5b61631

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Further complicating the picture, alternative or complementary mechanisms—such as methane release from ocean sediments or permafrost—are frequently cited alongside volcanic activity as top candidates for the PETM's cause , reflecting ongoing scientific debate about the relative contributions of different carbon sources

### Sample conflictingqa_a7ff288bc615

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The retrieved evidence is mixed

### Sample conflictingqa_a7ff288bc615

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Some sources argue that AI has already passed the Turing test: one IE business article states that "as of 2025, AI has passed the Turing test" a peer-reviewed arXiv paper titled "Large Language Models Pass the Turing Test" reports that LLMs exhibit human-like tone and fallibility

### Sample conflictingqa_a7ff288bc615

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: However, expert opinion and critical analysis push back on these claims: cognitive scientist Gary Marcus argues that the 2014 declaration of passing was similarly overblown that recent claims are premature because the bar was set too low , while a Popular Mechanics analysis of the GPT-4.5 study notes that passing the test does not constitute genuine artificial general intelligence

### Sample conflictingqa_a7ff288bc615

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Overall, whether AI has truly passed the Turing test remains a subject of ongoing debate, with researchers and commentators holding differing views on the test's meaning and the evidence available

### Sample conflictingqa_a864ff85e648

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: The evidence on whether Growth Hormone (GH) treatment reverses aging effects is genuinely divided

### Sample conflictingqa_a864ff85e648

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Some sources suggest that restoring optimal HGH levels can help reverse certain signs of aging, such as reduced muscle mass, decreased bone density diminished skin elasticity studies have indicated youth-like benefits including improved cognitive function and boosted immunity ; however, other high-credibility sources note that while GH may increase muscle mass and reduce body fat in healthy older adults, these changes do not clearly translate to increased strength or other definitive age-related improvements experts broadly agree that the existing research remains mixed and insufficient for a decisive conclusion

### Sample conflictingqa_a864ff85e648

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Overall, the scientific consensus leans toward caution, emphasizing that much stronger evidence is needed before GH can be reliably recommended as an age-reversal therapy

### Sample conflictingqa_a8ec1c0d2b92

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The retrieved evidence presents conflicting opinions

### Sample conflictingqa_a8ec1c0d2b92

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Some sources argue that green tea does not cause kidney stones and may even reduce the risk due to its antioxidant content and diuretic effects , while others note that green tea contains oxalates and recommend moderation for those at high risk

### Sample conflictingqa_a8ec1c0d2b92

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: A 2019 study cited by one source found that daily green tea consumption did not increase the risk of kidney stone formation , but a urologist quoted by the same source warned that iced tea (which includes green tea) is 'one of the worst things to drink' for those prone to the most common type of kidney stones

### Sample conflictingqa_a994724a28e7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: The4 evidence is conflicting

### Sample conflictingqa_a994724a28e7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Some sources argue that cold water closes the cuticle, potentially reducing frizz and improving shine, while others argue that the effect is negligible, that cold water damages hair that cold water rinsing is ineffective

### Sample conflictingqa_a9bed39d234d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: The retrieved evidence is divided on this question

### Sample conflictingqa_a9bed39d234d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Some sources argue that certain foods, particularly low-calorie vegetables and fruits, may burn more calories during digestion than they provide, a concept known as 'negative-calorie' foods

### Sample conflictingqa_a9bed39d234d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5
- **Claim**: However, high-credibility sources note that this is unlikely to be true for any food, as even low-calorie foods contain more calories than it takes to break them down and absorb them

### Sample conflictingqa_a9bed39d234d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5
- **Claim**: The mainstream scientific view, therefore, is that negative-calorie foods are unlikely to exist, though certain foods may still support weight loss by being low in calories and high in fiber or water content

### Sample conflictingqa_b2524e4883ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1
- **Claim**: The retrieved evidence presents competing perspectives

### Sample conflictingqa_b2524e4883ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Some sources argue that meteor showers primarily pose a threat to spacecraft and satellites rather than to people or Earth's surface

### Sample conflictingqa_b2524e4883ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Others argue that certain meteor streams may contain chunks large enough to cause significant damage upon impact

### Sample conflictingqa_b2524e4883ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: Currently, the dominant scientific view is that major meteor showers do not represent a significant direct threat to life on Earth, though smaller impacts and indirect effects (such as atmospheric pollution or satellite damage) are recognized

### Sample conflictingqa_b323dd4b5820

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Current CO2 levels are not unprecedented in Earth's history in an absolute sense, as levels were similarly high during the Pliocene 3.3 million years ago and potentially much higher still earlier; however, the recent rate of increase—100–200 times faster than any natural increase since the last ice age—is itself unprecedented, as is the magnitude when viewed against the full 66-million-year record

### Sample conflictingqa_b7fd50f9f980

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: Yes — 'alright' is a widely accepted alternate spelling of 'all right', used primarily in informal or casual contexts, while 'all right' is generally preferred in formal writing

### Sample conflictingqa_b9854bd5a19e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: Yes — the human brain has decreased in size by approximately 10–12.7% over the past 10,000–30,000 years, with some researchers attributing this to improved metabolic efficiency, others to declining body size still others to reduced cognitive load as humans transitioned to complex societies

### Sample conflictingqa_bac0f4d62f96

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The retrieved evidence is conflicting

### Sample conflictingqa_bac0f4d62f96

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Some sources argue that meteorites can come from comets, while others argue that few, if any, large meteorites come from comets

### Sample conflictingqa_bd2e652cd64d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Yes — electric toothbrushes are consistently found to be more effective than manual ones at removing plaque, cleaning along the gumline reducing gum recession

### Sample conflictingqa_bdee100fa8e5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The question of whether Orson Welles' 1938 War of the Worlds broadcast caused a real-life panic is genuinely contested among scholars and historians

### Sample conflictingqa_bdee100fa8e5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: While the broadcast is legendary for allegedly triggering mass hysteria—supposedly causing suicides, heart attacks widespread flight—modern research casts serious doubt on these claims

### Sample conflictingqa_bdee100fa8e5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: Studies cited by Wikipedia note that newspapers exaggerated rare cases of fear to discredit radio as a competing news medium surveys found that most listeners who panicked interpreted the drama as a German invasion rather than an alien one ; similarly, the BBC reports that Professor W. Joseph Campbell of American University argues the panic was always exaggerated that the majority of listeners understood the program was fiction from the start

### Sample conflictingqa_bdee100fa8e5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: On the other side, some sources insist the panic was real if limited to specific regions scattered eyewitness accounts describe genuine confusion and alarm —though academic researchers like Michael Socolow counter that these anecdotes were distorted by the press and that very few people actually heard the broadcast or took it as genuine

### Sample conflictingqa_be17259fe5c0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: Not according to a 2020 genomic study, which found that penguins first evolved in Australia and New Zealand and then spread to Antarctica — though a 2006 paleontological analysis argued an Antarctic origin was more likely

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The environmental comparison between paper and plastic straws is nuanced and depends on the metric considered

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: On the one hand, paper straws are biodegradable and avoid the persistent pollution associated with plastic, making them appear more environmentally friendly in terms of end-of-life impact ; additionally, some studies suggest paper straws may generate lower overall emissions than plastic across their full lifecycle

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: On the other hand, some research presents a contrasting picture: a UK government assessment found that paper straws actually emit more greenhouse gases when they decompose in landfills than plastic straws do one source notes that paper straws require significant energy to produce and emit greenhouse gases during manufacture

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Some critics further argue that paper straws may be less environmentally friendly than plastic when considering production costs alone experts widely agree that refusing straws altogether is the most sustainable choice

### Sample conflictingqa_bf6e9339f2c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: Yes, nutritional yeast is a complete protein source for vegans

### Sample conflictingqa_bf6e9339f2c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: Nutritional yeast is an 'excellent source of highly digestible complete protein' and contains all essential amino acids ; research published via NIH similarly confirms yeast single-cell protein (SCP) contains all essential amino acids in the required quantity, classifying it as a complete protein

### Sample conflictingqa_bf6e9339f2c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: While most sources agree on this point , some nuance exists: unfortified nutritional yeast contains just a few nutrients and lower vitamin levels the broader context is that combining multiple plant-based protein sources throughout the day is generally recommended to ensure full nutritional coverage

### Sample conflictingqa_bfbbc2c7a1af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Yes

### Sample conflictingqa_c1119b945459

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: The retrieved evidence indicates that Hindus do not believe in a single god in the strict monotheistic sense; rather, Hinduism is often described as henotheistic or mono-polytheistic

### Sample conflictingqa_c1119b945459

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: Many Hindus believe in one supreme being or divine force (such as Brahman) while also recognizing numerous deities as manifestations or aspects of that singular reality

### Sample conflictingqa_c1119b945459

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: At the same time, it is also true that no two Hindus may believe exactly the same thing, as the religion places a high degree of tolerance and individual choice in spiritual matters

### Sample conflictingqa_c34991d9897e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Yes, copyright can protect logos — but only if the logo contains artistic or creative elements; plain text or generic designs typically do not qualify

### Sample conflictingqa_c34991d9897e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: In the UK, a logo will almost always attract automatic copyright protection from the moment it is created, though this protection is limited to direct copying and does not prevent independent creation of similar designs

### Sample conflictingqa_c34991d9897e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5
- **Claim**: In the US and Australia, the standard is similarly conditional: a logo must meet the threshold of original artistic merit to receive copyright protection

### Sample conflictingqa_c3e66630928c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1
- **Claim**: Research and user opinions are divided on whether coffee grounds deter slugs and snails; laboratory tests found that caffeine solutions above 0.1% concentration deter snails, but dry grounds have generally insufficient caffeine strength to reliably stop them

### Sample conflictingqa_c418fecfc1e2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: Yes

### Sample conflictingqa_c574530da7a3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Religious and scholarly opinions differ on whether Adam and Eve were real historical figures; there is no settled scientific consensus on the question

### Sample conflictingqa_cc71318e5853

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The retrieved evidence presents conflicting views on whether death remains a taboo topic in modern society

### Sample conflictingqa_cc71318e5853

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: Some sources argue that death is still highly taboo, particularly in American culture, where a 1991 Gallup poll found that Americans almost never think about death children are shielded from it while the media exposes them to its most violent aspects ; similarly, one source notes that before the pandemic, death was among the most taboo topics in society, causing unprecedented fear

### Sample conflictingqa_cc71318e5853

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: On the other hand, d1 cites Blauner's thesis as arguing that death in modern society is not actually taboo d5 observes that death becomes less uncomfortable when discussed personally or professionally , suggesting the degree of taboo may vary by context or population

### Sample conflictingqa_cd661c2c20b6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Gwen Stacy's death is commonly cited as a symbolic or definitive end of the Silver Age

### Sample conflictingqa_d295f9ea94b3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Botox is not considered plastic surgery; it is classified as a non-surgical cosmetic procedure

### Sample conflictingqa_d9a36fe4c135

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Religious authorities and scholars hold differing views on biblical infallibility; there is no single authoritative answer accepted by all traditions

### Sample conflictingqa_dc6f972e8447

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Yes, cryptocurrency markets can be manipulated, although the ease of doing so is debated

### Sample conflictingqa_dd426f7706e8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: No, werewolves cannot be created by a full moon

### Sample conflictingqa_dd426f7706e8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: The full moon is commonly associated with werewolf transformations in popular culture, but this is largely a cinematic trope rooted in French folklore and Greek legends rather than a universal mythological rule

### Sample conflictingqa_dd426f7706e8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Werewolves were traditionally created through means such as curses, bites pacts with dark forces their transformations were not exclusively bound to the lunar cycle

### Sample conflictingqa_dd426f7706e8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Some fictional interpretations, such as Reddit user kylemichaelsmith's headcanon, do restrict bite-turned werewolves to transforming only during full moons, but this is one of many conflicting literary takes rather than a factual or universally accepted origin mechanism

### Sample conflictingqa_e2e2361dc28b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: Yes

### Sample conflictingqa_e60bdcf75ba7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Yes, organic yields are generally lower than conventional yields, though the magnitude varies considerably by crop type, region management practices

### Sample conflictingqa_e93e708d49a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Over their lifetime, typical rooftop solar panels produce more energy than they consume in their manufacture, installation recycling

### Sample conflictingqa_ece626a6cba9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Yes

### Sample conflictingqa_f1932b75ace7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Bee stings have been traditionally used to treat arthritis some studies suggest their venom contains anti-inflammatory compounds that could theoretically provide relief

### Sample conflictingqa_f1932b75ace7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: However, there is significant scientific disagreement: modern medicine does not formally recognize bee sting therapy (apitherapy) research remains largely inconclusive, with most evidence coming from animal studies, case reports user testimonials rather than controlled clinical trials

### Sample conflictingqa_f1932b75ace7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: Additionally, bee stings pose serious risks, including severe allergic reactions and potentially worsening joint conditions in some individuals, which must be carefully weighed against any potential benefits

### Sample conflictingqa_f22b389be1d6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Yes

### Sample conflictingqa_f39c966c2ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The retrieved evidence presents competing views

### Sample conflictingqa_f39c966c2ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: Some sources assert that Macbeth was cursed from its first performance, citing folklore that a coven of witches objected to Shakespeare's use of real incantations and caused disasters including the death of the actor playing Lady Macbeth around 1606

### Sample conflictingqa_f39c966c2ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: However, statistical analysis of production records has suggested that Macbeth does not experience significantly more mishaps than other Shakespearean plays, leading some scholars to question whether the curse is anything more than a theatrical legend

### Sample conflictingqa_f39c966c2ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The weight of evidence leans toward the curse being a folktale rather than a documented historical fact, though the story remains widely told in theater culture

### Sample conflictingqa_f3b163170581

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: No, humans did not evolve from modern apes such as chimps or gorillas, although we do share a common ancestor with them

### Sample conflictingqa_f3ba8599e370

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Yoga is not formally classified as a religion in and of itself, but it does contain spiritual elements and draws on Hindu scriptural traditions

### Sample conflictingqa_f43b2c51deea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Yes

### Sample conflictingqa_f4693bea2c31

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Emoji serve as a visual supplement to written language but do not constitute a distinct written language themselves

### Sample conflictingqa_f4811561af0c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Not strictly — Australia was first charted and landed upon by the Dutch in the early 17th century, but formal discovery/claim attribution is contested among scholars and sources

### Sample conflictingqa_f777a43ba278

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The link between yerba mate and cancer is nuanced and subject to ongoing scientific debate

### Sample conflictingqa_f777a43ba278

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: The NIH cites research indicating that almost all epidemiological studies on the topic found an association between mate consumption and cancers of the esophagus, larynx oral cavity, with risk increasing according to daily quantity, duration temperature

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Verywell Health similarly reports that population studies have demonstrated elevated rates of esophageal, head and neck bladder cancers among yerba mate users, though the evidence from animal studies regarding cancer prevention is described as not easily applicable to humans

### Sample conflictingqa_f777a43ba278

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d5
- **Claim**: However, GoodRx argues that the association with esophageal cancer is primarily attributable to the high temperature at which yerba mate is traditionally consumed—not the tea itself—and notes that laboratory research has also identified anti-cancer properties in yerba mate that led to cancer cell death ; Healthline and Dr. Axe further clarify that while the tea contains PAHs, a known carcinogen, the consensus is that consuming it in moderation at reasonable temperatures is generally considered safe, with the major risk factor being excessive, prolonged use alongside smoking and alcohol consumption

### Sample conflictingqa_f7fec8c0688b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1
- **Claim**: The retrieved evidence presents competing views

### Sample conflictingqa_f7fec8c0688b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: The official military explanation, cited by multiple sources, attributes the Phoenix Lights to LUU-2B/B rescue flares dropped by A-10 aircraft during a training mission, but many witnesses found this explanation unconvincing due to discrepancies in timing, motion volume described during the sighting

### Sample conflictingqa_f7fec8c0688b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Some researchers and former officials, including Arizona's former Governor Fife Symington, have argued that the lights were not flares because flares do not fly in formation and the object Symington witnessed was clearly solid and silent

### Sample conflictingqa_f7fec8c0688b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: As a result, the incident remains unresolved in public opinion, with many continuing to believe the lights represented a UFO or possibly a classified military aircraft rather than standard flares

### Sample conflictingqa_f8da23d84ecc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Once considered the same dinosaur, Apatosaurus and Brontosaurus were reclassified as distinct genera in a 2015 study

### Sample conflictingqa_f970957c5e52

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The retrieved evidence presents competing views

### Sample conflictingqa_f970957c5e52

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Some sources argue that the Oxford comma is optional but beneficial for clarity, while others contend that it is necessary in specific contexts to prevent ambiguity

### Sample conflictingqa_f97fef94decc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Studies and expert opinion are divided on this question

### Sample conflictingqa_f97fef94decc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5
- **Claim**: High-credibility sources note that modern VR headsets are designed with motion tracking and high-resolution displays that while there is no evidence VR causes permanent eye damage, prolonged use can lead to temporary discomfort — including eye strain, dryness blurred vision — similar to what one might experience from staring at a phone or computer screen for too long

### Sample conflictingqa_f97fef94decc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Some individuals, particularly those with pre-existing conditions like amblyopia or strabismus, may find the experience less tolerable children under 13 are generally advised against use ; one developer reported a convergence problem after prolonged use, though clinical experts stated there is no reliable evidence of permanent deterioration

### Sample conflictingqa_f97fef94decc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5
- **Claim**: Overall, most sources agree that moderation is key, as excessive screen time of any kind is linked to digital eye strain the 20-20-20 rule — looking away from the screen every 20 minutes to focus on something 20 feet away for 20 seconds — is recommended to mitigate these effects

### Sample conflictingqa_fa98c00bd697

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Black holes are not directly visible with a telescope because their gravity is so strong that nothing, including light, can escape

### Sample conflictingqa_fa98c00bd697

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: However, astronomers can detect black holes indirectly by observing phenomena such as gravitational lensing, accretion disk radiation jet emissions — and history's first direct black hole image was captured in 2019 by the Event Horizon Telescope

### Sample conflictingqa_fad0d30903d2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Yes, the evidence clearly supports that Woodstock promoted peace and love

### Sample conflictingqa_fbedb688b1d9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: Religious scholars and commentators hold differing views on whether Mormons are Christian

### Sample conflictingqa_fbedb688b1d9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: Some argue that because Mormons believe in Jesus Christ and accept much of the New Testament, they should be considered Christians, while others argue that key doctrinal differences — such as the rejection of the Trinity and the belief in ongoing scriptural revelation — make Mormonism fundamentally distinct from traditional Christian theology

### Sample conflictingqa_fcdb9e210683

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: The retrieved evidence presents conflicting views on whether viruses belong in the phylogenetic tree of life

### Sample conflictingqa_fcdb9e210683

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Some sources argue that viruses cannot be placed on the standard phylogenetic tree because they do not encode ribosomal RNA and are considered non-living entities under the classic definition of life ; Nature Reviews Microbiology counters that the phylogenetic tree is rooted in genomic sequence data rather than physical manifestation, meaning viral genomes do belong in the tree

### Sample conflictingqa_fcdb9e210683

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Other research supports viral inclusion through phylogenomic analysis, showing that modern viruses likely descended from ancient cellular organisms and that their genomes cluster with specific lineages ; meanwhile, some evolutionary researchers maintain that viruses do not truly fit the tree because they lack essential cellular machinery and exhibit evolutionary rates far exceeding those of cellular life

### Sample freshqa_0293a11bd364

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Hindi

### Sample freshqa_02b3ba89ebd0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: No Republican was elected Speaker of the House on the ninth ballot in January 2023

### Sample freshqa_02b3ba89ebd0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: The ninth ballot actually resulted in Kevin McCarthy receiving 200 votes, which was 18 short of the 218 needed for election, according to The Guardian

### Sample freshqa_02b3ba89ebd0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The New York Times similarly confirms that the election dragged well beyond the ninth ballot, with McCarthy finally securing the speakership only on the 15th vote after extensive negotiations

### Sample freshqa_02b3ba89ebd0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: This outcome reflects a broader narrative across multiple sources indicating that initial reports of a potential resolution on the ninth ballot were inaccurate the deadlock persisted far longer than originally anticipated

### Sample freshqa_0436c0b3a9d7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: Aryna Sabalenka and Amanda Anisimova

### Sample freshqa_047057d22309

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: King Charles III has not stripped Prince Harry's title as the Duke of Sussex

### Sample freshqa_047057d22309

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Harry and Meghan agreed to stop using their HRH titles when they stepped down as working royals in 2020 the official Royal Family website was later updated to reflect this change

### Sample freshqa_047057d22309

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Harry retains the title of Duke of Sussex, with his conduct and relationship with the monarchy remaining a subject of ongoing discussion rather than a formal title revocation

### Sample freshqa_049cc3f14d5e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: St. Petersburg State University

### Sample freshqa_1009f5c49e12

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Paris

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1
- **Claim**: August 16, 1977

### Sample freshqa_122352ad92e3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This year's Passover (Pesach) began on April 1, 2026

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: Hillary Clinton has not enacted any executive orders as a U.S. President

### Sample freshqa_1ef881d26e2e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: Maryam Mirzakhani (1977–2017) was the first female recipient of the Fields Medal, winning it in 2014 for her outstanding contributions to the dynamics and geometry of Riemann surfaces and moduli spaces

### Sample freshqa_1ef881d26e2e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: She was awarded the medal for her work while a professor at Stanford University, making history as the first woman and the first Iranian to receive the prestigious prize

### Sample freshqa_1ef881d26e2e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: However, d3 and d1 note that Maryna Viazovska later became the second female Fields Medalist in 2022, meaning Mirzakhani is no longer the only female recipient

### Sample freshqa_1fe3ba1b1cba

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_2130eea851fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Geoffrey Hinton has accumulated approximately 1,035,072 total citations across 776+ publications on Google Scholar as of June 2026, with an h-index of 190 and an i10-index of 527

### Sample freshqa_2130eea851fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Earlier reports, such as those from 36Kr in 2024, noted that Hinton had just become the second computer scientist to surpass one million Google Scholar citations, a milestone since surpassed as his citation count continues to grow

### Sample freshqa_25b286cb2af1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Venus has no moons

### Sample freshqa_263eca8e024e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: The name of the worldwide highest grossing Bollywood movie is Dangal

### Sample freshqa_2877cf4bd00f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: Donald Trump was born on June 14, 1946, making him 79 years old as of January 20, 2025

### Sample freshqa_2877cf4bd00f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: This is confirmed by his Wikipedia entry, which lists his date of birth and current age

### Sample freshqa_2877cf4bd00f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: His cardiac age, as assessed by his doctor following a medical evaluation, is approximately 65 — meaning his chronological age is roughly 79

### Sample freshqa_28e155139ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: The latest stable Android version is Android 16, released on June 10, 2025, with Android 17 Beta 1 also available for testing

### Sample freshqa_2b9ba7e192e2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Dina Boluarte (Dina Elisa Boluarte Zavaleta), who became Peru's first female president on December 7, 2022, after being sworn in during a turbulent political crisis that followed her predecessor Pedro Castillo's impeachment

### Sample freshqa_2d7eb41139aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: 6

### Sample freshqa_2e51f51132ee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_31ad09b9cd22

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The latest Grammy Award for Best Jazz Performance was won by Chick Corea, Christian McBride Brian Blade at the 68th Annual Grammy Awards in 2026

### Sample freshqa_31ad09b9cd22

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Their winning recording was "Windows - Live," honoring albums released between August 31, 2024 and August 30, 2025

### Sample freshqa_31ad09b9cd22

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This supersedes earlier reports citing Samara Joy as the 2025 winner, as the 2026 ceremony represented the most recent awards ceremony

### Sample freshqa_3227ea6c6056

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The latest major .NET version is .NET 10.0.8

### Sample freshqa_354e7097602d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The first atomic bomb test in the United States took place at the Trinity Site, located on the U.S. Army's Alamogordo Bombing and Gunnery Range in New Mexico

### Sample freshqa_354e7097602d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: Known as the Trinity test, this historic detonation occurred on July 16, 1945, at 5:30 a.m. the site is now part of the White Sands Missile Range, approximately 210 miles south of Los Alamos, New Mexico

### Sample freshqa_354e7097602d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The test was conducted as part of the Manhattan Project and resulted in a yield equivalent to approximately 18.6 kilotons of TNT

### Sample freshqa_35bf342002aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: 7

### Sample freshqa_3847b5cb9b42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1
- **Claim**: Russia's invasion of Ukraine (2022–present)

### Sample freshqa_39dcd7b38c39

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: Maya Angelou

### Sample freshqa_3ad16f379533

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Russia

### Sample freshqa_3cd0b514193b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: Tokyo's minimum wage is ¥1,226 per hour

### Sample freshqa_3dc3cf00bce6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: Queen Elizabeth II of England was famous for keeping Pembroke Welsh Corgis

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
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Joe Biden did not visit Russia as president of the United States

### Sample freshqa_4a98eba95e97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5
- **Claim**: His only meeting with Russian President Vladimir Putin took place in Geneva, Switzerland, on June 16, 2021, during a summit that gave a temporary positive impulse to U.S.-Russia relations before being derailed by Putin's war on Ukraine

### Sample freshqa_4a98eba95e97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: This Geneva meeting is confirmed across multiple authoritative sources the Institute of New Europe explicitly notes that Biden did not visit Russia due to the ongoing conflict in Ukraine

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
- **Claim**: The youngest passenger on board the Titanic was Millvina Dean, who was approximately two months old at the time of departure

### Sample freshqa_5574b1447bdb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: Wuhan

### Sample freshqa_5d6e5db69928

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Greenland (Peary Land / Kap København formation); older claims cite Siberian permafrost, but Greenland record stands as of 2022

### Sample freshqa_5eb89aae15f3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: KGF: Chapter 1

### Sample freshqa_5ecee1c55713

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Portugal

### Sample freshqa_64c12116affc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: Donald J. Trump is the President of the United States, having served two non-consecutive terms: first from January 20, 2017 to January 20, 2021 currently from January 20, 2025 to the present

### Sample freshqa_64c12116affc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: This is confirmed by the official White House website, which identifies him as the 45th and 47th President

### Sample freshqa_64c12116affc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: His first term was marked by significant domestic and foreign policy initiatives his current administration is noted for record economic growth and border security achievements

### Sample freshqa_6927f007d7cc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: The Voice US season 29 was won by Alexia Jayy, who topped the Battle of Champions finale on April 14, 2026

### Sample freshqa_6927f007d7cc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: She performed "Lady Marmalade" and "One and Only" during the finale, earning the most votes from an in-studio audience of past contestants and superfans

### Sample freshqa_6927f007d7cc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This supersedes earlier reports from 2025, which had cited Adam David as the winner of a prior season

### Sample freshqa_6a45fadeb16b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Based on recent data, Costco's Executive membership costs $130 annually , providing a 2% cashback reward on qualified purchases

### Sample freshqa_6a45fadeb16b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Older sources citing $120 or $45 reflect outdated pricing or promotional offers rather than the current standard rate

### Sample freshqa_6f42c128eb6c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Harry Maguire has never won the Ballon d'Or

### Sample freshqa_7bc7bb2dde20

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: The latest Academy Award for Best Picture was won by **One Battle After Another** (Paul Thomas Anderson, 2025), which claimed the prize at the 98th Academy Awards

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by multiple sources reporting on the ceremony, with the film also winning awards for Best Director and Best Adapted Screenplay

### Sample freshqa_7bc7bb2dde20

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Earlier records show a temporal progression of winners — including *Anora* (2025), *Oppenheimer* (2024) *Everything Everywhere All at Once* (2023) — but the most current information places *One Battle After Another* at the top

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
- **Supporting Docs Found**: d4, d5
- **Claim**: The first animal to circle the Moon was the Soviet spacecraft Zond 5, which carried two Russian tortoises in September 1968

### Sample freshqa_7e63fcff2dea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Unlike the first animal to orbit Earth (Laika the dog aboard Sputnik 2 in 1957), Zond 5 did not land on the Moon but represented the earliest lunar loop with living cargo

### Sample freshqa_7e63fcff2dea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: For the record, no dog has ever landed on the Moon while some sources incorrectly cite monkeys as the first lunar travelers , the Zond 5 tortoises hold the accurate distinction for the first animals to circle the Moon

### Sample freshqa_7f1c3aae61a5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Luke Littler

### Sample freshqa_80642f637dc7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Lionel Messi

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: Bayonne, New Jersey

### Sample freshqa_8eca5bd62ae0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: Beijing

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
- **Claim**: Frank Rosenblatt, the inventor of the Perceptron, died in a boating accident

### Sample freshqa_a47283064972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: No, the Raptors do not have a winning record in the latest NBA season

### Sample freshqa_a47283064972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The Britannica document explicitly lists the 2023–24 season record as 25–57, which is well below .500 and confirms a losing record

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: While the Raptors did advance to the playoffs in the 2025–26 season , the available evidence does not support a winning regular-season record in the most recent NBA season

### Sample freshqa_a50d0f1f3cdf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: Queen Elizabeth II of England died on 8 September 2022

### Sample freshqa_a5492f36ca23

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: January 10, 2016

### Sample freshqa_a8b908895e11

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1
- **Claim**: San José

### Sample freshqa_ab11b5dce00e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1
- **Claim**: USA

### Sample freshqa_b0ffe73c1789

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Colleen Hoover has published at least 26 books, though sources differ on the exact total due to outdated information

### Sample freshqa_b3264b37f54b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Yes, Arsenal is on top of the Premier League standings

### Sample freshqa_b99c189f2222

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Jeff Bezos did not sell Amazon; he sold Amazon shares

### Sample freshqa_c3f10dc1632d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5
- **Claim**: Jiangsu

### Sample freshqa_c479e83e408f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: 15

### Sample freshqa_c4976f8629cb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The heaviest reptile in the world is the saltwater crocodile (Crocodylus porosus), according to Quora

### Sample freshqa_c4976f8629cb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: While the Komodo dragon is often cited as the largest lizard, it is significantly smaller than the saltwater crocodile, weighing only up to 365 pounds compared to the crocodile's much greater mass

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
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: Vincent van Gogh

### Sample freshqa_cf331ed7d09f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: macOS 26 Tahoe

### Sample freshqa_d4d59d75b206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5
- **Claim**: Drake did not top Spotify's list of most-streamed artists in three consecutive years

### Sample freshqa_d4d59d75b206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5
- **Claim**: He was the most-streamed artist in 2015, 2016 2018, but these years are not consecutive — there is a gap between 2016 and 2018

### Sample freshqa_d4d59d75b206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5
- **Claim**: Seed answer "2016–2018" is incorrect because 2017 was not a consecutive year of topping the list

### Sample freshqa_d510972df578

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: By nominal production budget, Star Wars: The Rise of Skywalker (2019) is currently the most expensive completed film, with analysts placing its net budget at roughly $490 million

### Sample freshqa_d510972df578

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This figure includes principal photography, reshoots post-production but excludes global marketing costs

### Sample freshqa_d510972df578

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Earlier record-holders include Pirates of the Caribbean: On Stranger Tides (2011), which had a reported budget of $378.5 million Star Wars: The Force Awakens, which ranks highest when budgets are adjusted for inflation at approximately $552 million

### Sample freshqa_dd85dcbc2262

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Aryna Sabalenka

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5
- **Claim**: 12

### Sample freshqa_ddd643091cbe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: A permanent cure for cancer has not been developed

### Sample freshqa_ddd643091cbe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: The retrieved evidence shows that while some cancer types can be cured—such as through complete surgical removal of a malignant tumor or the first complete chemotherapy cure of choriocarcinoma in 1953—most cancers cannot be permanently cured once they have metastasized researchers are still exploring newer treatments like vaccines and gene editing that could one day change the landscape of cancer care

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The game resumed approximately 21 minutes after Damar Hamlin suffered cardiac arrest on the field

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
- **Supporting Docs Found**: d3
- **Claim**: Slugs have one lung

### Sample freshqa_f5d8e53958c6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Hawaii

### Sample freshqa_f5eef5c415e8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: 28

### Sample freshqa_f6ac249bdf53

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: Ta-Nehisi Coates

### Sample freshqa_f6cc6071caa5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: A total of 893 Nazca geoglyphs have been discovered, comprising 248 additional geoglyphs found through AI-supported field surveys conducted in 2023 and 2024, plus the 645 previously known figures

### Sample freshqa_f6cc6071caa5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: This figure is further broken down by glyph type, with the 248 new discoveries including 160 figurative geoglyphs, raising the known total of figurative glyphs to 893

### Sample freshqa_f6cc6071caa5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Earlier counts reported lower totals because each new survey round superseded prior records: a 2022 update had raised the total to approximately 358 after 168 glyphs were found , while a 2024 AI analysis had nearly doubled the then-known total of 430 subsequent continuous discoveries pushed the cumulative tally to the current 893

### Sample freshqa_fc07d4525dbf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: 6 months

### Sample freshqa_fd00b29e848c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: Ramadan 2026

### Sample freshqa_ff04793833cc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: Andrew Johnson was not elected as President of the United States in any year; he became President in 1865 after Abraham Lincoln's assassination

### Sample healthcontradict_0060

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: No, a tepid sponge bath does not reduce fever in children and is not recommended by NHS guidelines

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
- **Supporting Docs Found**: d10, d5, d2, d7, d6
- **Claim**: Boston College

### Sample hotpotqa_0064

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d5
- **Claim**: Victor Mature

### Sample hotpotqa_0070

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Tom Daley

### Sample hotpotqa_0071

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d10
- **Claim**: Keyshia Cole

### Sample hotpotqa_0073

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d10
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
- **Supporting Docs Found**: d9, d5, d1
- **Claim**: Lit's best known song is "My Own Worst Enemy," a number one rock hit that helped their album A Place in the Sun go platinum

### Sample hotpotqa_0123

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: It was released in March 1999 as the lead single from that album and went on to win the Billboard Modern Rock Track of the Year award

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
- **Supporting Docs Found**: d7
- **Claim**: More than 1,600 German scientists, engineers technicians were recruited in post-Nazi Germany as part of Operation Paperclip, though the subset directly linked to Arthur Rudolph's work on the U.S. space program is unspecified

### Sample hotpotqa_0196

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Stuart

### Sample misinformation_0023

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: No, drinking bleach does not cure infections and is extremely dangerous

### Sample qacc_0013

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d7
- **Claim**: The Fourteenth Amendment

### Sample qacc_0023

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d1, d7, d8
- **Claim**: Pentheus was torn apart by the maenads at the end of The Bacchae

### Sample qacc_0023

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1, d8
- **Claim**: He was lured to the woods by Dionysus and killed there, with his own mother Agave bearing his head on a pike back to Thebes

### Sample qacc_0023

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d1, d7, d8
- **Claim**: This outcome, noting that the maenads scattered his body parts across the hillside and that his mother held his decapitated head

### Sample qacc_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Justin Timberlake

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d8, d5, d7
- **Claim**: The movie contains 506 instances of the word "fuck", according to multiple sources including Guinness World Records , The Guardian , Time Variety

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1, d7, d8, d6
- **Claim**: This figure is further corroborated by Guinness World Records and Slate , while Wikipedia similarly reports 569 f-words Collider confirms the film's status as one of the most profane in cinema history with a similar count

### Sample qacc_0156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d6
- **Claim**: Sheldon Collins

### Sample qacc_08cf866bcb9b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: The documentary Anne_Bancroft

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0ac549afb037

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: The last name Hansen is of Scandinavian and Germanic origin, derived from the personal name Hans and used as a patronymic surname in Danish, Norwegian, Dutch, Flemish North German cultures

### Sample qacc_0ac549afb037

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: It is the most common surname in Norway and is also found in variant forms such as Hanssen, Hansson Hanson

### Sample qacc_0ac549afb037

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: While the name itself has these linguistic roots, 23andMe ancestry data indicates that people with the surname Hansen have the highest concentration of British & Irish ancestry (36.8%), followed by French & German (25.6%) and Scandinavian (19.9%) origins, suggesting the name has spread beyond its Nordic birthplace

### Sample qacc_0b3b6452e224

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: The Statue of Liberty was designed by French sculptor Frédéric Auguste Bartholdi, who modeled the statue's face after his own mother and drew inspiration from the Roman goddess of liberty, Libertas

### Sample qacc_0b75ed799d46

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Shrine Auditorium & Expo Hall

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: Following the Allied victory in North Africa, the document indicates the Allies moved eastward across North Africa and ultimately into Europe via Italy

### Sample qacc_0d85f1089c4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The Beti Bachao-Beti Padhao campaign has had multiple brand ambassadors across different states

### Sample qacc_0d85f1089c4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Parineeti Chopra was chosen as the brand ambassador for Haryana's version of the campaign , while Sakshi Malik was appointed as the brand ambassador for the Haryana government's initiative

### Sample qacc_0d85f1089c4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: At the national level, Madhuri Dixit became the brand ambassador the campaign also selected Avani Lekhara for Rajasthan and Bhawna Dehariya Mishra and her daughter Siddhi Mishra for Madhya Pradesh

### Sample qacc_1025b0681710

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Cassie Scerbo plays Lauren Tanner in Make It or Break It

### Sample qacc_132167e66120

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: India won the Cricket World Cup in 1983, 2007, 2024 2026

### Sample qacc_132167e66120

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: The 1983 ODI victory was India's first, led by Kapil Dev at Lord's in England , while the 2007 T20 triumph was led by MS Dhoni in South Africa

### Sample qacc_132167e66120

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: In 2024, India defeated South Africa in a nail-biting final led by Rohit Sharma in 2026 India defended the title by defeating New Zealand by 96 runs in the Ahmedabad final, becoming the first team to win three T20 World Cups and do so on home soil

### Sample qacc_15ffab2466f7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: The Pantages Theatre

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
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: Oliver Stark

### Sample qacc_1a764b8b6cf5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The rule of the four rightly guided caliphs was called the Rashidun Caliphate (Arabic: الخلافة الراشدة, al-Khilafah al-Rashidah)

### Sample qacc_1a764b8b6cf5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: This term is used in Sunni Islam to denote the period from 632 to 661 CE, when Abu Bakr, Umar, Uthman Ali ruled as the first caliphs following the death of Muhammad

### Sample qacc_1a764b8b6cf5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: The word Rashidun (راشدون) means 'rightly guided,' signifying that their rule served as a model to be followed and emulated from a religious standpoint

### Sample qacc_1b95727cc286

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Azie Faison, Rich Porter Alpo Martinez

### Sample qacc_213701765f94

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: January 15, 2009

### Sample qacc_2243f17ccc38

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: Leeds United won the FA Cup in 1968 and again in 1972

### Sample qacc_252987b8054c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5
- **Claim**: Tori Spelling

### Sample qacc_287da9f37864

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: Lionel Messi made his first appearance for Barcelona's first team on November 16, 2003, when he came on as a substitute in the 75th minute of a friendly match against Porto during the inauguration of the Estádio do Dragão

### Sample qacc_287da9f37864

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: This debut, under coach Frank Rijkaard, occurred when Messi was just 16 years, four months 23 days old

### Sample qacc_287da9f37864

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: His official competitive debut followed on October 16, 2004, in a La Liga match against Espanyol, where he again came off the bench to replace Deco in the 82nd minute

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
- **Claim**: The first vertebrates to exist on earth were fish — specifically jawless fishes — which are recognized as the earliest group to have possessed a vertebral column

### Sample qacc_2a7f7e06e365

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: These early vertebrates emerged around 480 million years ago during the Early Ordovician period, making them the most ancient group among all vertebrates

### Sample qacc_2a7f7e06e365

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The fossil record further confirms that these jawless fishes gave rise to all subsequent vertebrate lineages, including those that eventually transitioned to land

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
- **Supporting Docs Found**: d2, d1
- **Claim**: Isle de Jean Charles, a sinking island off the coast of New Orleans; filming also took place in the swamps and rural areas of southern Louisiana

### Sample qacc_2f6d2647a424

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: Pete Rose

### Sample qacc_34cba3c71e06

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Missi Hale

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
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: The practice of crossing fingers for good luck is generally traced to pre-Christian European traditions in which the cross was a potent magical sigil associated with binding and securing wishes , though it was later absorbed into Christian practice as a covert recognition symbol among early persecuted believers

### Sample qacc_3c1297608017

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Historians note that the modern solo gesture likely evolved from the original two-person ritual, in which one person's index finger crossed over another's, forming a cross to anchor a wish until fulfillment ; this evolution may have been further popularized during the Hundred Years' War the gesture remains most widespread in historically Christian nations

### Sample qacc_3d4ebfa8b6dd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: As a coach, Phil Jackson holds the record with 11 NBA championships (six with the Chicago Bulls and five with the Los Angeles Lakers); among players, Bill Russell holds the record with 11 championships

### Sample qacc_403a59870dc2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The Rams have won the Super Bowl twice

### Sample qacc_403a59870dc2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Their first victory was on January 30, 2000, when the St. Louis Rams defeated the Tennessee Titans 23-16 in Super Bowl XXXIV at the Georgia Dome in Atlanta

### Sample qacc_403a59870dc2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Their second win came on February 13, 2021, when the Rams defeated the Cincinnati Bengals 23-20 in Super Bowl LVI at their home stadium, becoming the second NFL team to win the Super Bowl in their own facility

### Sample qacc_41c44ecfd0f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: Lacteals

### Sample qacc_4387048ed24f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1
- **Claim**: Anne Bancroft won the Oscar for Best Actress for The Miracle Worker at the 35th Academy Awards in 1963, beating Bette Davis who was nominated for her role in What Ever Happened to Baby Jane?

### Sample qacc_44b315f6f4bb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: The retrieved evidence indicates that the Queen's crown jewels are kept in the Tower of London (specifically at the Jewel House within the Tower's grounds), though some sources note that the medieval coronation regalia were historically kept at Westminster Abbey until 1649

### Sample qacc_4fb90d57c274

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: 1991

### Sample qacc_51b23ea15977

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The Soviet Union was leading the space race in April 1961, as evidenced by Yuri Gagarin's historic flight aboard Vostok 1 on April 12, 1961, which made him the first human to travel into space

### Sample qacc_51b23ea15977

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This achievement left the United States trailing, as NASA responded by accelerating its own manned spaceflight efforts, including the eventual Apollo program aimed at reaching the Moon

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
- **Claim**: Anguillara Sabazia

### Sample qacc_5a9576fc5d8c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Jodie Sweetin

### Sample qacc_5eeb46719843

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Canada did not gain independence from Great Britain on a single date, as the process was evolutionary rather than abrupt

### Sample qacc_5eeb46719843

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5
- **Claim**: The Statute of Westminster in 1931 is commonly cited as when Canada became fully independent, but some sources argue that full legal and constitutional independence was not complete until the passage of the Canada Act in 1982

### Sample qacc_5eeb46719843

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Earlier milestones include the Balfour Declaration of 1926, which recognized Canada as an autonomous community within the British Commonwealth the granting of partial legislative independence in 1919

### Sample qacc_5fb5c311d373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Lin-Manuel Miranda wrote "How Far I'll Go" for Moana

### Sample qacc_6485f021b694

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Carroll O'Conner and Jean Stapleton

### Sample qacc_66ba2af9c3b9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Soman Chainani

### Sample qacc_67b35f41ba84

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Alice Kremelberg

### Sample qacc_6837d86d03ac

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Prince William, the Prince of Wales, is first in line to the British throne

### Sample qacc_6969589d80c1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1
- **Claim**: Matt Monro

### Sample qacc_6af6e8cb8f34

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Queen Charlotte, the German-born wife of George III, is credited with introducing the first Christmas tree to Britain by decorating one with candles and sweets at Queen's Lodge, Windsor in December 1800

### Sample qacc_6af6e8cb8f34

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: This makes her the earliest known introducer of the tradition, predating Prince Albert's later popularization by approximately four decades

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
- **Supporting Docs Found**: d5, d1
- **Claim**: 179

### Sample qacc_720d8b210357

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: The number of DNA replication origins in eukaryotes varies by organism and chromosome size

### Sample qacc_720d8b210357

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: High-credibility sources report approximately 30,000–50,000 origins in human cells , while other complex eukaryotes may have around 20 identified origins the general principle across all eukaryotic species is that multiple origins are required due to the large genome size

### Sample qacc_7916ffefdb99

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: John B. Watson

### Sample qacc_798b6853d20f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: glucose

### Sample qacc_7bf02a7deb69

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Charlie Day

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: October 1968

### Sample qacc_7f5e5a4a4391

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: The letter J was first used in the Middle Ages as a scribal variant of I and was formally established as a distinct letter after 1600 ; by the 16th and 17th centuries, scholars and printers had fully adopted it as a separate character , with the first English language books clearly distinguishing between I and J appearing in 1629 and 1633

### Sample qacc_7f5e5a4a4391

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: The evidence is nuanced: d3 states J did not exist in English until 1633, while d4 notes it was used in Spanish prior to 1600 and was finally acknowledged as a full letter in the nineteenth century

### Sample qacc_883303a2d535

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Nana in Snow Dogs is identified as an Australian Shepherd

### Sample qacc_8882ab46be5d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: 38

### Sample qacc_899648874637

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: Kate Walsh plays Dr. Addison Shepherd in Grey's Anatomy

### Sample qacc_8d7c14ed548f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Factor X

### Sample qacc_8daf80e943fa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5
- **Claim**: 5.88 trillion miles

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_8ef7b3cf5c3f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: The dominant ethnic group in southern South America, including Argentina and Uruguay, are those of European descent

### Sample qacc_9404250d756f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1
- **Claim**: The End of the F***ing World was primarily filmed in the UK, with Season 1 locations including Camberley, Guildford the Isle of Sheppey (Kent), while Season 2 was shot entirely in Wales

### Sample qacc_940e6d9275f4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Billy Idol

### Sample qacc_946ecfb478b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Justin Timberlake

### Sample qacc_950881e7c998

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Boston Red Sox

### Sample qacc_9a9a28d7e159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The retrieved evidence indicates that the Fairy Tail anime has already concluded, with the final season airing from October 7, 2018 to September 29, 2019

### Sample qacc_9a9a28d7e159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: However, a 2026 sequel manga titled Fairy Tail: 100 Years Quest is currently being published, featuring new chapters released bi-weekly in the United States

### Sample qacc_9b16fd6882f3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: Russ Ballard (Argent)

### Sample qacc_9c2f95b14a78

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1
- **Claim**: The Duluth Model is an intervention program that emphasizes a coordinated community response to domestic violence, holding batterers accountable while keeping victims safe

### Sample qacc_9c2f95b14a78

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: It recognizes domestic violence as a pattern of power and control exerted by an abuser over an intimate partner and incorporates a gender-based analysis that examines societal norms contributing to violence against women

### Sample qacc_9c2f95b14a78

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The model promotes a collaborative approach involving multiple community stakeholders — including law enforcement, criminal justice professionals, social service providers advocacy organizations — to ensure victim safety, hold abusers accountable provide comprehensive support services

### Sample qacc_9c2f95b14a78

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1
- **Claim**: Unlike a traditional treatment program, the Duluth Model functions as a Coordinated Community Response (CCR) that places the primary responsibility for controlling abusers on the community and the individual abuser, rather than on the victim

### Sample qacc_9fbf28f5786f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: The International Space Station (ISS) was conceived in 1993 and its construction began in the late 1980s , with the first elements launched in 1998

### Sample qacc_9fbf28f5786f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: The first crew arrived in 2000, marking the station's official occupation , though the question of a single 'launch date' is complicated because the ISS was built incrementally over time

### Sample qacc_a1c73439eca9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_a3c882e062c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: The planned completion date for the Sagrada Familia has been updated to the early 2030s, with the final towers of the Glory Façade expected to be completed during that period

### Sample qacc_a3c882e062c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: This revised schedule is due to pandemic-related delays, superseding the earlier target of 2026, which was already facing uncertainty because only the main spire was scheduled to be finished by that year

### Sample qacc_a3c882e062c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: The construction authority has declined to commit to a more precise date given the ongoing nature of the work

### Sample qacc_a44267c115d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: Most of the water in the body is located within the intracellular space (approximately two-thirds), with the remaining one-third found in the extracellular space, which includes interstitial fluid and blood plasma

### Sample qacc_a44267c115d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This distribution is consistent across multiple authoritative sources, including academic references that confirm water constitutes roughly 50–70% of total body weight in adults, with the brain and heart each containing around 73–83% water

### Sample qacc_a4dde83c35da

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: The Ming Dynasty had an autocratic government

### Sample qacc_a635c2fd4869

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: Roberta Flack

### Sample qacc_a6a2f8b1f0b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1
- **Claim**: 233

### Sample qacc_a6b48b7accc4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: The first official T20 match was played between Sussex and Surrey in England in 2003 , marking the debut of this revolutionary cricket format

### Sample qacc_a6b48b7accc4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The first official Twenty20 matches were contested on 13 June 2003 between English counties in the inaugural Twenty20 Cup , with the first Lord's venue match following on 15 July 2004

### Sample qacc_a6b48b7accc4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Video evidence further supports that the landmark New Zealand vs. Australia T20 International was played in 2005 , confirming the sequential evolution of the game from domestic to international stages

### Sample qacc_a6df0af8c2ba

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Hosanna is a Hebrew expression meaning “save us” or “save us now,” derived from the phrase hoshi'a na

### Sample qacc_a6df0af8c2ba

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: It functioned as both a prayer and an exclamation of praise, having lost its original supplicatory sense and becoming a joyful acclamation by the time of Jesus' entry into Jerusalem

### Sample qacc_a6df0af8c2ba

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The word is also used in Jewish feasts, such as the Feast of Tabernacles, where the seventh day was known as 'Hosanna Day'

### Sample qacc_a78a32b7b9a1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Atlanta Falcons

### Sample qacc_a91ae87c969d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Linda Davis

### Sample qacc_a927c4cccc6a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: 1977

### Sample qacc_aa94588b9477

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: 14 January 1960

### Sample qacc_aaf0f638e99b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1
- **Claim**: A yellow 35 mph sign is generally an advisory speed sign, not a regulatory speed limit sign

### Sample qacc_aaf0f638e99b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: It suggests reducing speed to 35 mph for safety, but exceeding the advised speed is not enforceable under general traffic statutes

### Sample qacc_aaf0f638e99b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: In North Carolina, for example, only black-on-white signs are considered regulatory and enforceable, while yellow horizontal alignment signs with speed advisories are solely used to advise motorists of a curve or roadway change

### Sample qacc_aaf0f638e99b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Similarly, in New Zealand, a 35 km/h advisory sign covers a range of approximately 31–41 km/h and is similarly unenforceable

### Sample qacc_aaf0f638e99b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: That said, some jurisdictions (such as California) do treat yellow advisory signs as enforceable speed limits, so drivers should be aware of local rules

### Sample qacc_ac5341df9a83

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: Troops for UN military actions come from Member States; the Security Council authorizes deployments via resolution UN Headquarters then liaises with countries to identify and deploy personnel

### Sample qacc_b0346f60b6ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5
- **Claim**: Celebrity Big Brother is broadcast on CBS in the USA

### Sample qacc_b0ee06f2950d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Season 6 of American Horror Story is titled _American Horror Story: Roanoke_

### Sample qacc_b198a514fff8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: 47th

### Sample qacc_b1bd9518429b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Gibraltar

### Sample qacc_b281f09f0959

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Joseph McCarthy

### Sample qacc_ba7aaa9b36c8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: A fire broke out in the West Wing of the White House on Christmas Eve 1929, during a party for the children of presidential aides, destroying much of the wing

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

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Sri Lanka

### Sample qacc_bfbb5f55a63f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_c264cb69676e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: A synovial joint; specifically, the joint between the malleus and incus is a synovial saddle joint (also called the incudomalleolar joint), not a hinge joint

### Sample qacc_c27400199055

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: Ghana

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Seth MacFarlane voices Carter Pewterschmidt, Lois's father, on Family Guy

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: While Mike Henry is known for his work on Family Guy, he portrays Cleveland Brown rather than Lois's dad

### Sample qacc_c675e6cd8ad6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: The retrieved evidence suggests that multiple composers were involved depending on which Disney Robin Hood production is meant

### Sample qacc_c675e6cd8ad6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: For the 1952 live-action film, Elton Hayes composed the music and wrote original songs including 'Whistle, My Love,' drawing on medieval English melodies for inspiration

### Sample qacc_c675e6cd8ad6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: For the 1973 Disney animated version, George Bruns served as the composer for the majority of the soundtrack, while individual songs like 'Oo-De-Lally' featured music by Roger Miller 'Love' was composed by Floyd Huddleston

### Sample qacc_c69855566c76

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: Paul Reubens plays Pee-wee Herman in *Pee-wee's Big Holiday*

### Sample qacc_c731579bb51c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Hallmark Movies and Mysteries is on channel 565 HD on DIRECTV

### Sample qacc_c88807a22775

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: .22 Long Rifle

### Sample qacc_c9b95dd57e73

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Peter Sarstedt

### Sample qacc_cb5bcdb1ef9b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1
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
- **Claim**: The last name Tavarez is of Spanish and Portuguese origin, derived from the habitational name Tavares found in Portugal and the Azores

### Sample qacc_cbf25273f973

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: It is also associated with the Dominican Republic and has variants such as Tavares and Tavares across different regions

### Sample qacc_cbf25273f973

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Based on 23andMe data, people with the surname Tavarez show recent ancestry links to Cuba and Mexico their top paternal haplogroup is Q-M3, which is predominantly found among people with East Asian and Indigenous American ancestry

### Sample qacc_ce4983c8a9c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Most effigy mounds were built between approximately 700 and 1200 CE , with a peak construction period estimated at roughly 750–1050 CE ; the broader Woodland period context places their origins as early as 650 CE some sources cite up to 1200 CE as a terminal date

### Sample qacc_d00b0063e747

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: yes

### Sample qacc_d03e85bdc95a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Aristotle

### Sample qacc_d39801b5de65

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: July 4, 1776

### Sample qacc_d3b85d857358

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1
- **Claim**: The plane that dropped the atomic bomb on Hiroshima was the Enola Gay, a Boeing B-29 Superfortress bomber

### Sample qacc_d3b85d857358

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: On August 6, 1945, it became the first aircraft ever to drop an atomic weapon in warfare, releasing the bomb code-named 'Little Boy' over the city

### Sample qacc_d3b85d857358

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The Enola Gay was named after Enola Gay Tibbets, the mother of its pilot, Colonel Paul Tibbets

### Sample qacc_d44802dc3c96

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: November 1936

### Sample qacc_d60bf850c4ff

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Cadbury sells its products in over 50 countries

### Sample qacc_d78d45c0e30f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Colombia and Japan

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
- **Supporting Docs Found**: d3
- **Claim**: It is also frequently described more broadly as simply a spiral galaxy some sources note it may be classified as Sc or SBc depending on the degree of central bar prominence and arm looseness

### Sample qacc_d8b24beb2f90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: The balance sheet

### Sample qacc_d96b47272030

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Nintendo was founded in 1889

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: XXXTENTACION

### Sample qacc_e064a7a717ed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: The Glass Castle was primarily filmed in Montreal, Quebec, with additional shooting in West Virginia and New Mexico

### Sample qacc_e064a7a717ed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The movie's pre-production and principal photography took place in Montreal during May–June 2016, capturing both interior sets and exterior New York-inspired street scenes

### Sample qacc_e064a7a717ed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: In West Virginia, the production filmed several scenes in Welch and surrounding areas, including at Vic Nystrom Stadium and Mount View High School also captured desert landscapes on the To'hajiilee and Laguna Pueblo tribal lands near Albuquerque, New Mexico

### Sample qacc_e06ada156e0e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Nicole Gale Anderson

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: In Mexico, toll roads are called autopistas (federal highways with the suffix "D" for directo) the toll fee is specifically referred to as a cuota

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: These tolled routes are also known as libramientos (bypass ring roads) and that their toll plazas are commonly called casetas

### Sample qacc_e6d89fce1b8e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Owen Hunt

### Sample qacc_e7318f6f3bbe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: strengths

### Sample qacc_e87ffc07efd1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Franklin D. Roosevelt

### Sample qacc_ea469c846404

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Rangers' last appearance in the Champions League group stages was the 2025–26 season, where they finished bottom of their group with 6 points from 6 matches in the play-off round

### Sample qacc_ea469c846404

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This result means they did not qualify for the main group stage but participated in the play-offs

### Sample qacc_ea469c846404

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: Earlier records show Rangers also competed in the group stage during the 2022–23 season, finishing third in their group previously in the 2011–12 season under manager Ally McCoist

### Sample qacc_ea469c846404

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The club's most recent qualification for the group stages occurred in the 2024–25 season, where they were eliminated in the third qualifying round

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
- **Supporting Docs Found**: d2, d5
- **Claim**: Apollo 17 Commander Eugene Cernan was the last astronaut to walk on the lunar surface, having returned to the spacecraft around 5:40 a.m. This mission marked the final human lunar landing of the Apollo program NASA subsequently terminated the Apollo 17 astronauts' stay on the moon

### Sample qacc_ec5b0067c29a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: Number One Observatory Circle

### Sample qacc_ecbc6adf8a48

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Scholars place the writing of the First Epistle of John between 70 and 110 AD, with one source specifically suggesting the 90s AD as the most probable period

### Sample qacc_ecd3d9c0ca11

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Guy Norris

### Sample qacc_f10c7ad4bb81

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: Initialisms

### Sample qacc_f1776add7672

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d5
- **Claim**: ICD-10 codes have a flexible length depending on the version and use case

### Sample qacc_f1776add7672

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: For inpatient care (ICD-10-CM), codes range from three to seven characters — starting with a letter, followed by numbers and optional additional specifiers — while for procedural coding (ICD-10-PCS), each code is fixed at seven characters

### Sample qacc_f2218f8c979e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1
- **Claim**: The rib primal

### Sample qacc_f69c37496013

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: 1987

### Sample qacc_fbdae168fc6f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Indira Gandhi

### Sample qacc_fbe562911999

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1
- **Claim**: In the Indian Warrant of Precedence, the Speaker of the Lok Sabha is placed at Sl

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
- **Claim**: It depends on jurisdiction; federal US law generally requires 18 to buy a shotgun, though individual states and nations like the UK have different rules

### Sample situatedqa_geo_09a6a048cfbf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: It depends on where you are; in the United States (federal minimum legal drinking age is 21 years old

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: The meaning of a red license plate varies by jurisdiction and context

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: In many countries, red plates commonly indicate vehicles belonging to a fleet (such as rental, city commercial fleets), though specific rules differ by region

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: In Spain, red license plates are used for vehicles in circulation during registration processing, those temporarily out of service employed for research and testing , while in Ontario, Canada, red plates are restricted to authorized dealerships and diplomats

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: In Turkey, red text on a white background signifies vehicles belonging to senior executives such as security directors or university rectors in Japan, a red stripe on a license plate carries its own specific significance

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: United States casualties in World War II included approximately 416,800 military deaths and 1,700 civilian deaths , making a combined total of roughly 418,500

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This figure is consistent across multiple authoritative sources, including the National World War II Museum and a research starter from EBSCO , which together provide comprehensive casualty data for all participating nations

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: For a broader WE (Western Allies) total, one source claims there were approximately 407,000 American military casualties , while UK casualties stood at about 382,700 military and 67,100 civilian deaths , reflecting the conflict's staggering human cost

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The minimum age to drive a transport vehicle varies by jurisdiction and vehicle type, but federal employment rules provide a useful reference point

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Employees under 16 may not drive motor vehicles on public roads as part of their jobs, while 17-year-olds may drive in limited circumstances — offering a de facto lower bound for transport drivers

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: For commercial transport specifically, many companies require drivers to be at least 21 years old, with some state and federal regulations echoing this threshold for operating larger vehicles , while international rules such as the North American Free Trade Agreement require drivers to be at least 18 to cross borders

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: For those under 18, nearly all U.S. states restrict driving privileges through graduated licensing programs, requiring anywhere from 15 to 17 years of age to obtain a full license

### Sample situatedqa_geo_362420ec2416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: As per the 2011 Census of India, Sikkim is the state with the lowest population

### Sample situatedqa_geo_362420ec2416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Multiple sources consistently report Sikkim's approximate population as 6,10,000 (six lakh ten thousand) inhabitants, making it the least populated state in the country

### Sample situatedqa_geo_362420ec2416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: This figure is confirmed across various references, including the official Census 2011 data, which places Sikkim at the bottom when compared to all other states

### Sample situatedqa_geo_3815eef77a3c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: The welfare state was introduced at different times across nations depending on the country and definition

### Sample situatedqa_geo_3815eef77a3c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: A major consensus point is that Germany pioneered social insurance in the 1880s, with Chancellor Otto von Bismarck enacting the world's first compulsory health and unemployment insurance laws ; Britain later followed with its own landmark 1911 National Insurance Act, which consolidated earlier liberal reforms of 1906–1914 , while the United States formally established its welfare state architecture during the 1930s via President Roosevelt's New Deal legislation

### Sample situatedqa_geo_48e19f6b37bc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: California is the 3rd largest U.S. state by area, with approximately 163,695 square miles

### Sample situatedqa_geo_48e19f6b37bc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: This is confirmed across multiple authoritative sources, including the U.S. Census Bureau data cited by Britannica and Infoplease , as well as supplementary rankings from Swedish Nomad and Vedantu

### Sample situatedqa_geo_48e19f6b37bc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Globally, the United States itself ranks as the third largest country by land area, at roughly 3.8 million square miles

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
- **Supporting Docs Found**: d3
- **Claim**: The Dandi March was led by Mahatma Gandhi and involved thousands of participants, including Ashramites, students people from various Indian states

### Sample situatedqa_geo_66684169f016

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Notable individuals who accompanied Gandhi on the march included Pyare Lal Nayar, who served as his personal secretary Mithuben Petit, a Parsi woman who joined him from Sabarmati

### Sample situatedqa_geo_66684169f016

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: A partial list of persons who accompanied Gandhi identifies members from Gujarat, Maharashtra Uttar Pradesh, providing further specificity on the march's composition

### Sample situatedqa_geo_6cca7fdb9b41

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The global point on Earth farthest from any ocean is the Eurasian pole of inaccessibility, located in northwestern China near Kazakhstan

### Sample situatedqa_geo_6cca7fdb9b41

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5
- **Claim**: For Britain specifically, the village of Coton in the Elms in Derbyshire is often cited as the furthest point from the sea, with Church Flatts Farm nearby measured at approximately 113km (70 miles) from the nearest coast ; alternatively, some sources claim this title belongs to Ashby de la Zouch, Cross-in-Hand Tring depending on the definition used

### Sample situatedqa_geo_7222d6123c27

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Calcutta became the capital of British India in 1772, when Warren Hastings transferred all important offices there

### Sample situatedqa_geo_7222d6123c27

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d5
- **Claim**: The capital was then moved from Calcutta to Delhi in 1911, making Delhi the capital of India under British rule

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
- **Claim**: This makes 1935 the most commonly cited year for Social Security's inception, though the program developed in stages throughout the 1930s

### Sample situatedqa_geo_779fd84224fa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Sydney Cove

### Sample situatedqa_geo_7f4d90dffb38

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: It depends on location; taxes vary by state and country

### Sample situatedqa_geo_7f4d90dffb38

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: Federally, the U.S. imposes a flat excise tax of 18.4 cents per gallon on gasoline, while Ohio specifically charges $0.385 per gallon

### Sample situatedqa_geo_7f4d90dffb38

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Average state-level excise taxes range from 8 to 61 cents per gallon, with California having the highest at approximately 61 cents per gallon

### Sample situatedqa_geo_80c72c2797bf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: U.S. federal government

### Sample situatedqa_geo_85af31651715

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: England: 1 July 2007 (Health Act 2006); Scotland: 26 March 2006; Wales: 2 April 2007

### Sample situatedqa_geo_864fc7e36714

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Historical data shows that the bulk of immigrants coming to the United States originated from Europe, with nearly 90% coming from Europe during the late 19th and early 20th centuries the 1965 Immigration and Nationality Act radically changed this composition by opening doors to Latin America and Asia

### Sample situatedqa_geo_864fc7e36714

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: More recent data reflects a temporal shift: Pew Research data shows that since 1965, about half of U.S. immigrants have come from Latin America—with Mexico alone accounting for about 25% of new arrivals in the 1965–2007 wave and approximately 11% between 2021 and 2023 —while Brookings Institution data from 2023 indicates that nearly half of recent immigrants originate from South and Central America and the Caribbean, with Mexico, India China as the top three countries of origin

### Sample situatedqa_geo_897e47478bbc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: 649,481

### Sample situatedqa_geo_8c889f8ce07a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Under U.S. constitutional law, treaty ratification is a joint process involving both the executive and legislative branches: the President negotiates and submits treaties to the Senate the Senate provides advice and consent — requiring a two-thirds majority for approval — but the President is ultimately responsible for signing and depositing the instrument of ratification

### Sample situatedqa_geo_8c889f8ce07a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: This constitutional design is reflected in the practical process described by the Senate's own documentation, which notes that the Senate's role is to consider resolutions of ratification rather than ratify treaties directly that formal ratification occurs when the instruments are exchanged between the United States and the foreign party

### Sample situatedqa_geo_932ca9fa3259

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Levee maintenance responsibilities vary by ownership and jurisdiction

### Sample situatedqa_geo_932ca9fa3259

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: The U.S. Army Corps of Engineers (USACE) is responsible for building and maintaining USACE-owned levees, inspecting those structures overseeing levee safety nationwide ; additionally, local levee boards, landowners federal agencies like the Natural Resources Conservation Service may own and maintain other levees depending on the region

### Sample situatedqa_geo_932ca9fa3259

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5
- **Claim**: Ownership and specific roles can be confirmed through the National Levee Database, which lists the responsible entity for each levee , while levee owners and operators generally handle the everyday care, repairs emergency response

### Sample situatedqa_geo_9f6c5f849b40

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: The most populous cities in the world are Jakarta (Indonesia), Dhaka (Bangladesh) Tōkyō (Japan), with populations estimated at 41.9M, 36.6M 33.4M respectively

### Sample situatedqa_geo_9f6c5f849b40

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: In the United States specifically, the three largest cities are New York, Los Angeles Chicago, with populations of approximately 8.8M, 3.9M 2.7M

### Sample situatedqa_geo_9f6c5f849b40

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Regional and definitional differences produce varying rankings depending on the scope and methodology used to define a 'city.'

### Sample situatedqa_geo_a1cde1787dab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5
- **Claim**: The Clean Air Act was passed in 1970, signed by President Richard Nixon on December 31, 1970

### Sample situatedqa_geo_a1cde1787dab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: This 1970 version superseded earlier federal air pollution laws passed in 1955 and the Clean Air Act of 1963, making it the most current and comprehensive U.S. environmental statute addressing air quality

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Eisenhower

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5
- **Claim**: The California grizzly bear (Ursus arctos californicus)

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
- **Supporting Docs Found**: d4, d5
- **Claim**: Scotland

### Sample situatedqa_geo_f2031e426cee

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The current Law Minister of India is Arjun Ram Meghwal, who serves as the Minister of Law and Justice

### Sample situatedqa_geo_f2031e426cee

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: He is not to be confused with Kiren Rijiju, who formerly held the portfolio and is sometimes incorrectly identified as the current Law Minister ; d3 explicitly names Meghwal as the Minister of the Ministry of Law and Justice

### Sample situatedqa_geo_f2031e426cee

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: At the union level, the Ministry of Law and Justice is headed by Arjun Ram Meghwal, while the Attorney-General is R. Venkataramani

### Sample situatedqa_geo_f26078ec6467

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: Spain

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: The first form of national government after the Revolutionary War was the Articles of Confederation

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The shift from tea to coffee in the US began during the American Revolution when the Boston Tea Party made tea-drinking politically risky coffee became the patriotic alternative

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Coffee would completely eclipse hot tea by 1865 — when Union soldiers returned home from the Civil War having become accustomed to it as part of their standard rations — and by around 2025, approximately 75% of American adults drink coffee daily compared to a smaller fraction who drink tea

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The Federal Open Market Committee (FOMC) is the primary body that sets U.S. monetary policy

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

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
- **Claim**: Mohammad Hamid Ansari

### Sample situatedqa_temp_14a587def215

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: 2026

### Sample situatedqa_temp_14f29fe7ff15

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: The British under General Howe defeated the Continental Army at the Battle of Brandywine on September 11, 1777, opening the way for the British conquest of Philadelphia

### Sample situatedqa_temp_14f29fe7ff15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This result is corroborated by the American Battlefield Trust, which notes that the battle was the largest of the Revolutionary War in terms of manpower and that the British victory left the Continental Army intact despite Washington's defense

### Sample situatedqa_temp_14f29fe7ff15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Encyclopedia Britannica further confirms that the battle resulted in a British defeat of the Americans, though the Continental Army remained intact, directly contributing to the eventual American victory at Saratoga

### Sample situatedqa_temp_14f70522567e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Lionel Messi

### Sample situatedqa_temp_180f238d8296

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Australia (5 titles), India (3 titles), West Indies (2 titles), Pakistan (1 title), Sri Lanka (1 title)

### Sample situatedqa_temp_1987d35f994b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: Great Basin became a national park in 1986

### Sample situatedqa_temp_19badef7553b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: The Philadelphia Eagles won their first Super Bowl on February 4, 2018, defeating the New England Patriots 41-33 in Super Bowl LII

### Sample situatedqa_temp_19badef7553b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: This victory gave the Eagles their first NFL championship since 1960 and marked the franchise's breakthrough moment after decades of near-misses

### Sample situatedqa_temp_19badef7553b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The Eagles made two additional Super Bowl appearances in 2023 and 2025, winning their second title on February 9, 2025, again against the Kansas City Chiefs

### Sample situatedqa_temp_1baff64de20e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: Rumer Willis guest starred in Pretty Little Liars season four, playing the character Zoe — a charity organizer who appears in an episode scheduled to air in late July

### Sample situatedqa_temp_1c56e575f096

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: 2024

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
- **Claim**: Cory A. Booker

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
- **Claim**: John Williams

### Sample situatedqa_temp_32d33d503f69

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: Henry Danger: The Movie is coming out on January 17, 2025

### Sample situatedqa_temp_35156c8be377

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: The answer depends on the metric used to define 'richest.' By total GDP (nominal), Nigeria has historically been the largest economy in Africa, with a 2016 GDP of approximately $411.966 billion this ranking is corroborated by 2021 data

### Sample situatedqa_temp_35156c8be377

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5
- **Claim**: By GDP per capita measured in current US dollars, Seychelles leads at an estimated $42,110 as of 2025 , making it the richest country in Africa in terms of purchasing power parity (PPP)

### Sample situatedqa_temp_35156c8be377

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Additionally, South Africa is identified as the top economy by GDP when using 2024 IMF data , underscoring how the answer varies significantly depending on whether total economic output or per-capita wealth is the preferred measure

### Sample situatedqa_temp_3521e5dc831c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1
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
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Mort is a mouse lemur (family Cheirogaleidae), specifically a Goodman's mouse lemur, making him a small primate native to Madagascar

### Sample situatedqa_temp_40e6764f611f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: While he is consistently identified as a mouse lemur across sources, some fan theories and spin-off content add complementary details: Mort is portrayed as a hybrid character who is 40% mouse lemur and 60% other animals (including bear, starfish spider components) one user even theorizes that Mort is an Eldritch God

### Sample situatedqa_temp_40e6764f611f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: However, these additional elements are not universally accepted facts and primarily appear in fan discussions and spin-off materials rather than the main film franchise

### Sample situatedqa_temp_43c0aaae9828

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Hillsong Worship

### Sample situatedqa_temp_44d30ce08bf5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: UCLA

### Sample situatedqa_temp_45cbc4dc3e28

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: The current Chief Justice of the Sindh High Court is **Justice Zafar Ahmed Rajput**, who was appointed as Acting Chief Justice on December 6, 2025, succeeding Justice Muhammad Junaid Ghaffar

### Sample situatedqa_temp_45cbc4dc3e28

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: This is confirmed by the official court list and a subsequent newspaper report noting his formal appointment

### Sample situatedqa_temp_45cbc4dc3e28

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: Earlier sources citing Saadat Khan or Muhammad Junaid Ghaffar as Chief Justice are outdated, as the most recent data places Justice Rajput at the helm

### Sample situatedqa_temp_4ead2f4cd2d5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: Chrishell Stause played Bethany Bryant on The Young and the Restless

### Sample situatedqa_temp_4fbe76f974ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: 1939

### Sample situatedqa_temp_50748f92be3a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5
- **Claim**: The last FIFA World Cup was held in 2026 Argentina are the defending champions after winning the 2022 edition in Qatar

### Sample situatedqa_temp_50748f92be3a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5
- **Claim**: This fact is corroborated by the updated list of World Cup winners that includes the 2022 result

### Sample situatedqa_temp_50748f92be3a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Older sources, such as a Forbes ranking published during the 2014 tournament a document listing only winners up to 2022, are superseded by the newer 2026 result

### Sample situatedqa_temp_557d04a28d7f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: LeBron James

### Sample situatedqa_temp_587e89bbcbe1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: 108

### Sample situatedqa_temp_5a59faf24972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5
- **Claim**: Android 16

### Sample situatedqa_temp_5e6b388e88aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: 2022

### Sample situatedqa_temp_60095fdf39a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: The next Avatar comic is Avatar: The Last Airbender—Kyoshi Warriors, which began releasing in spring 2026

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
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: 1980

### Sample situatedqa_temp_6a683d0d04f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Five sharps in a key signature signify the key of B major

### Sample situatedqa_temp_6a683d0d04f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5
- **Claim**: Using the standard order of sharps—F♯, C♯, G♯, D♯, A♯, E♯, B♯—the first five sharps appear in B major, making it the key with the most sharps before the signature wraps around to include double sharps

### Sample situatedqa_temp_6a683d0d04f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: To confirm this manually, apply the rule that the key is a half-step above the last sharp in the signature; a half-step above A♯ (the fifth sharp) brings you to B, corroborating that five sharps indicate B major

### Sample situatedqa_temp_6a8aed3e3d5f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: 245

### Sample situatedqa_temp_6c424fc78a69

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The Pakistan Tehreek-e-Insaf (PTI) party, led by Imran Khan, won the 2018 general election in Pakistan

### Sample situatedqa_temp_6c424fc78a69

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: PTI became the single largest party in the National Assembly with 157 seats, surpassing the Pakistan Muslim League-Nawaz (PML-N), which won 84 seats the Pakistan People's Party Parliamentarians (PPPP), which won 54 seats

### Sample situatedqa_temp_6c424fc78a69

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: This is corroborated by opinion poll data showing PTI leading the race and by reporting that Khan became Pakistan's prime minister-elect following the election

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5
- **Claim**: Todd Monken

### Sample situatedqa_temp_78c89bf81e8b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: On naval ships, SS most commonly stands for steamship, denoting a vessel powered by steam engines

### Sample situatedqa_temp_7a7a7f43e575

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Washington

### Sample situatedqa_temp_7bb7fe6c9287

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Additionally, Grendel is variously termed the "captain of evil," "corpse-maker," "shadow-stalker," and "terror-monger," all of which serve to emphasize his malevolent and destructive nature

### Sample situatedqa_temp_7cd18101326e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Indiana QB Fernando Mendoza was named the Offensive MVP and DL Mikail Kamara was named the Defensive MVP of the Jan 19, 2026 national championship game

### Sample situatedqa_temp_7d5f559b313d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5
- **Claim**: The most recent GDP figure for the United States is **$31.82 trillion** as of Q1 2026, reported by YCharts and confirmed by Moody's Analytics at **31,819,464 million** for the same quarter

### Sample situatedqa_temp_7d5f559b313d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: This represents approximately 1.26% growth from the prior quarter and 5.92% growth over the past year

### Sample situatedqa_temp_7d5f559b313d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: Earlier data from Wikipedia reported a lower figure of $30.762 trillion for full-year 2025 USAFacts cited $24.2 trillion for Q1 2026 , though the latter appears to reflect a methodological difference in data scaling rather than a temporal update

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Australia's coastline measures approximately 59,681 kilometers (km) in total, combining both mainland and island shorelines

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
- **Supporting Docs Found**: d2
- **Claim**: Tay-Sachs disease is an autosomal recessive genetic disorder

### Sample situatedqa_temp_8808c106a115

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: It is caused by a deficiency of the hexosaminidase A (HEX A) enzyme, which is necessary to break down GM2-ganglioside within cells of the body, particularly in the brain and nerve cells

### Sample situatedqa_temp_8808c106a115

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1
- **Claim**: The disorder is inherited when an individual receives two variant copies of the HEXA gene — one from each parent — and the severity of symptoms varies depending on whether the disease manifests in infancy, adolescence adulthood

### Sample situatedqa_temp_88ef1cfab62e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Hunter Emery plays CO Rick Hopper in Orange is the New Black

### Sample situatedqa_temp_8b66ca78c3d1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: 11,937

### Sample situatedqa_temp_8f57822ed7d2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1
- **Claim**: 2020

### Sample situatedqa_temp_8fbdda192a13

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: September 1967

### Sample situatedqa_temp_901be1437bc7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Maryland

### Sample situatedqa_temp_9049c197a579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: California's total gas tax stands at approximately 70 cents per gallon, making it the highest in the country

### Sample situatedqa_temp_9049c197a579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: This figure has grown from approximately $0.612 per gallon in the 2025–2026 period , reflecting ongoing increases in state excise taxes and related fees that have pushed California's rate above 90 cents per gallon in recent years

### Sample situatedqa_temp_956e1f6e518e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: Apollo 17, December 1972

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
- **Supporting Docs Found**: d3, d4, d2, d5
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
- **Supporting Docs Found**: d4, d5, d3, d2, d1
- **Claim**: The Battle of Badr took place on March 13, 624 CE , corresponding to the 17th of Ramadan in the Islamic calendar

### Sample situatedqa_temp_a8c74bced99a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: All available sources consistently confirm this date without contradiction

### Sample situatedqa_temp_ac0e6a4a7e32

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: Sun Yat-sen

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Shay Mitchell (Emily Fields) — 39

### Sample situatedqa_temp_ae0882e48812

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The two largest deserts in China are the Gobi Desert and the Taklamakan Desert

### Sample situatedqa_temp_b3ad3248b7d4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: The retrieved evidence supports 1438 as the start date and 1533 as the end date

### Sample situatedqa_temp_b6c75b99a2f4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: 670–700 nm

### Sample situatedqa_temp_b797de4c6610

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Cardiac biomarkers are substances that appear in the blood when the heart is stressed or damaged they are used to diagnose and monitor heart disease

### Sample situatedqa_temp_b797de4c6610

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: The most widely used cardiac biomarker is cardiac troponin (troponin T or I), which enters the bloodstream shortly after a heart attack and remains elevated for days, making it the gold standard for detecting myocardial infarction

### Sample situatedqa_temp_b797de4c6610

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Other commonly referenced biomarkers include creatinine kinase (CK), its heart-specific subtype CK-MB myoglobin, though these are less specific and have largely been superseded by troponin in clinical practice

### Sample situatedqa_temp_b797de4c6610

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5
- **Claim**: Additional biomarkers used in specific contexts include natriuretic peptides (such as BNP or NT-proBNP) for assessing heart failure severity, C-reactive protein (CRP) for inflammation associated with cardiovascular disease lactate dehydrogenase (LD) despite its poor cardiac specificity

### Sample situatedqa_temp_bb85e25a8159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5
- **Claim**: Florida Panthers

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: HMS Queen Elizabeth was commissioned on December 7, 2017 formally declared operational in 2020

### Sample situatedqa_temp_bdc0853f9a23

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: The last name Gerard is of French and Norman origin, derived from the Old French personal name Gérard, which itself traces to the ancient Germanic elements gēr ('spear') and hard ('hardy' or 'brave')

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: It is also found in Haiti and has cognates across Germanic and Romance-speaking regions

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: WikiTree and HouseofNames similarly note that the surname was first recorded in the Domesday Book of 1086 and is linked to the Anglo-Saxon Gerard family of Lancashire, while the name's broader Germanic roots trace back to the Proto-Germanic reconstruction gari-hard

### Sample situatedqa_temp_c92bc2a37a57

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Shai Gilgeous-Alexander

### Sample situatedqa_temp_ce10e6f883fb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: India and Pakistan

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: 166

### Sample situatedqa_temp_d0579ca3907c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The Battle of Kadesh reportedly began on Year 5 III Shemu day 9 during the reign of Ramesses II, generally dated to May 1274 BCE , though one source earlier cited 1275 BCE ; it concluded on the same day without a clear end timestamp

### Sample situatedqa_temp_d15849658c20

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: Oleksandr Usyk

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_d59edcfab87f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: Queen Charlotte

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
- **Supporting Docs Found**: d3, d2, d5
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
- **Claim**: Golden State Warriors, 73

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
- **Claim**: Hello, Love, Again

### Sample situatedqa_temp_efa79b1e7fb1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Stephen Curry

### Sample situatedqa_temp_f099d9d1452b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: The current US Director of the CIA is John L. Ratcliffe, who was officially sworn in on January 23, 2025

### Sample situatedqa_temp_f099d9d1452b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This is confirmed by the CIA's own official announcement, which states that Vice President JD Vance administered the oath of office at a White House ceremony

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
- **Supporting Docs Found**: d3, d2
- **Claim**: McDonald's Monopoly game pieces typically come on the packaging of certain menu items, such as Big Macs or large fries, though the specific items vary by year and region

### Sample situatedqa_temp_f3b3face1b6b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5
- **Claim**: In the 2025 edition, over 30 menu items were eligible to receive game pieces — some physical and some digital — and pieces could also be obtained through the McDonald's app

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: 2026

### Sample situatedqa_temp_f971e49123a1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: 13

### Sample trust_align_124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d1
- **Claim**: Cemeteries typically establish endowment care funds by setting aside a portion of each plot sale, as required by laws in many U.S. states, to ensure perpetual care and maintenance even after all plots are sold

### Sample wikirevision_0001

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: Twitter is currently known as X. In April 2023, Twitter merged with X Holdings and ceased to be an independent company, becoming a part of X Corp. This rebranding is confirmed across multiple sources, including the newer Wikipedia revision of Twitter and the article on X itself

### Sample wikirevision_0002

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: Twitter is now known as X. In April 2023, Twitter merged with X Holdings and ceased to be an independent company, becoming part of X Corp. This rebranding is confirmed across multiple sources, including the newer Wikipedia revision of Twitter and the article on X itself

### Sample wikirevision_0003

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: Twitter is now known as X. In October 2023, Twitter rebranded itself as X, officially changing its corporate identity

### Sample wikirevision_0003

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: This rebranding is confirmed across multiple sources, including the newer Wikipedia revision of Twitter and the separate article on X. The change means that Twitter's former name is no longer used for the company or platform

### Sample wikirevision_0004

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The current name of Facebook's parent company is Meta Platforms, Inc. This was confirmed when Facebook rebranded itself as Meta Platforms, Inc. in October 2022, officially changing its corporate identity

### Sample wikirevision_0004

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The rebranding is further corroborated by additional context showing that Meta Platforms, Inc. is the parent company behind Facebook's products and initiatives

### Sample wikirevision_0007

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Google is owned by Alphabet Inc. This is confirmed by Wikipedia's article on Alphabet Inc., which identifies it as the parent company of Google and notes that Google was reorganized as a wholly owned subsidiary of Alphabet in 2015

### Sample wikirevision_0007

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Additionally, the older Wikipedia revision of Alphabet Inc. and the article on Google itself both corroborate this structure, with the latter explicitly stating that Google is Alphabet's largest subsidiary

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
- **Claim**: The latest President of India is Droupadi Murmu, who has held office since July 2022

### Sample wikirevision_0025

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by the official Wikipedia revision that superseded the older version in January 2026, which explicitly names her as the current President with a 2025 official portrait

### Sample wikirevision_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: She is the 15th President of India, succeeding Ram Nath Kovind her tenure is consistent across multiple high-credibility sources

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
- **Claim**: He is the 53rd Chancellor of Germany and leads the Christian Democratic Union (CDU)

### Sample wikirevision_0033

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: This is consistent across multiple sources, including the current Wikipedia revision of the Chancellor of Germany article, which also lists him as the incumbent

### Sample wikirevision_0035

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: The latest Prime Minister of Japan is Sanae Takaichi, who assumed office on 21 October 2025

### Sample wikirevision_0035

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: She is the first female Prime Minister in Japan's history

### Sample wikirevision_0035

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of the Prime Minister of Japan page, as well as the list of prime ministers of Japan

### Sample wikirevision_0040

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The latest President of Argentina is Javier Milei, who took office on 10 December 2023

### Sample wikirevision_0040

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by the newer Wikipedia revision of the President of Argentina page, which supersedes the older revision from December 2024

### Sample wikirevision_0040

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Milei is the current President, serving the Executive branch of the Argentine Nation

### Sample wikirevision_0041

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The current President of Argentina is Javier Milei, who took office on 10 December 2023

### Sample wikirevision_0041

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: He is the 42nd President of Argentina and belongs to the political party Unión por el Cambio

### Sample wikirevision_0041

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple up-to-date sources, including the Wikipedia article on the President of Argentina, which also notes that the position is the highest political office in the Argentine Nation, held by a direct popular vote

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
- **Claim**: Argentina (for the 2022 FIFA World Cup, the most recent completed tournament)

### Sample wikirevision_0047

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Argentina (defending 2022 champion, 3rd title) — the 2026 FIFA World Cup has not been played yet

### Sample wikirevision_0049

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The current Indian Premier League champion is Royal Challengers Bengaluru (RCB), who won the 2026 IPL title — their first championship in the league's history

### Sample wikirevision_0049

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: This supersedes older information referencing the 2023 season, in which Chennai Super Kings had previously claimed their fifth title by defeating Gujarat Titans , as the most recent data confirms RCB's 2026 victory

### Sample wikirevision_0057

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Google is owned by Alphabet Inc. This is confirmed by Wikipedia's newer revision, which identifies Alphabet Inc. as the parent company of Google and notes that Google was reorganized as a wholly owned subsidiary of Alphabet in 2015

### Sample wikirevision_0057

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Older sources, including an earlier Wikipedia revision from May 2026, also reference Alphabet Inc. as the owner but are superseded by the more recent information

### Sample wikirevision_0061

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: The current President of Mexico is Claudia Sheinbaum Pardo, who took office on 1 October 2024, making her the 66th President of Mexico

### Sample wikirevision_0061

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: She is the first woman and the first Jewish person to hold the office, serving as the Head of State and Government at the National Palace in Mexico City

### Sample wikirevision_0061

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple sources, including the older and newer Wikipedia revisions of the President of Mexico article, as well as her own Wikipedia biography

### Sample wikirevision_0064

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The current President of Turkey is Recep Tayyip Erdoğan, having served in office since 28 August 2014

### Sample wikirevision_0064

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: He is the 12th president in Turkey's republican history and holds the dual role of both head of state and head of government

### Sample wikirevision_0064

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: This is consistent across multiple sources, including the older and newer Wikipedia revisions of the President of Turkey page, which both confirm his incumbency since 2014

### Sample wikirevision_0065

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Facebook's parent company is Meta Platforms, Inc. This was confirmed when Facebook rebranded itself as Meta Platforms in 2021, officially changing its corporate identity

### Sample wikirevision_0065

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: As a result, Facebook is no longer its own parent company, but rather a subsidiary owned by Meta Platforms

### Sample wikirevision_0066

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: Facebook's parent company is Meta Platforms, Inc. This was confirmed when Facebook rebranded itself as Meta Platforms in 2021, officially changing its corporate identity

### Sample wikirevision_0066

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The rebranding was announced on October 28, 2021 the company subsequently filed a certificate with the State of Delaware to change its legal name

### Sample wikirevision_0066

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: As a result, Facebook is no longer its own parent company, but rather a subsidiary operated by Meta Platforms

### Sample wikirevision_0067

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The 2025 Ballon d'Or was won by Ousmane Dembélé, making it his first time receiving the award

### Sample wikirevision_0067

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The ceremony took place on 22 September 2025, recognizing the best footballers of the 2024–25 season

### Sample wikirevision_0067

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Dembélé remains the most recent holder

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
- **Claim**: Twitter is currently known as X. In April 2023, Twitter merged with X Holdings and ceased to be an independent company, becoming part of X Corp. This rebranding is confirmed across multiple sources, including the newer Wikipedia revision of Twitter and the article on X itself

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
- **Claim**: The latest Prime Minister of Pakistan is Shehbaz Sharif, who became incumbent on 4 March 2024

### Sample wikirevision_0076

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by the official Wikipedia revision that superseded the older version in April 2025, which explicitly names Shehbaz Sharif as Prime Minister with an incumbency date of 4 March 2024

### Sample wikirevision_0076

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: Additional context: Pakistan has had 20 prime ministers since 1947 Shehbaz Sharif's appointment is further corroborated by the fact that he also appointed Ishaq Dar as Deputy Prime Minister in 2024

### Sample wikirevision_0082

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: The current Ballon d'Or holder is Ousmane Dembélé, who won the 2025 award at the 69th ceremony, representing France

### Sample wikirevision_0082

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: This supersedes older information naming him as the 2025 holder, as the document timestamp is May 2026 and also references the upcoming 2026 ceremony

### Sample wikirevision_0082

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Earlier records show the 2024 Ballon d'Or went to Rodri, but that result has since been updated by the 2025 ceremony

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
- **Claim**: This is consistent across multiple sources, including the high-credibility Wikipedia article on the Prime Minister of France, which also notes that the role has been known as 'Prime Minister' since 1959 when Michel Debré became the first officeholder under the Fifth Republic

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
- **Claim**: The latest Leader of the Labour Party in the UK is Keir Starmer, who was elected to the position on 4 April 2020, following his victory in that year's Labour Party leadership election

### Sample wikirevision_0088

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: This is confirmed by the official Wikipedia entries on both the Leader of the Labour Party and the 2020 Labour leadership election, as well as by the Labour Party's own records

### Sample wikirevision_0088

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Starmer continues to serve as the party's leader and as the UK's Prime Minister after Labour won the 2024 general election

### Sample wikirevision_0089

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Calcutta is officially called Kolkata

### Sample wikirevision_0089

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: The city officially changed its name from Calcutta to Kolkata in 2001 this is confirmed across multiple sources including the newer Wikipedia revision of Calcutta and the main Kolkata article

### Sample wikirevision_0089

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: While Calcutta is still commonly used, especially outside of India, the official legal name of the city is now Kolkata

### Sample wikirevision_0090

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: The latest President of Indonesia is Prabowo Subianto, who took office on 20 October 2024

### Sample wikirevision_0090

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: He is the eighth president of Indonesia and served as minister of defence under President Joko Widodo from 2019 to 2024

### Sample wikirevision_0090

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of the President of Indonesia page, as well as his own Wikipedia biography

### Sample wikirevision_0093

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Jannik Sinner

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
- **Claim**: Australia is the latest Cricket World Cup champion, having defeated India in the 2023 final to win their sixth title

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This result is corroborated across multiple sources, with the 2023 edition being the most recent completed tournament

### Sample wikirevision_0099

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The 2027 Cricket World Cup is next scheduled for 2027, meaning Australia's 2023 title currently stands as the most recent

### Sample wikirevision_0100

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The current Prime Minister of Pakistan is Shehbaz Sharif, who took office on 4 March 2024

### Sample wikirevision_0100

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: He is the 22nd Prime Minister of Pakistan and serves as the leader of the House and head of the country's executive branch

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
- **Supporting Docs Found**: d1
- **Claim**: He has also served as Prime Minister of the United Kingdom since the 2024 general election

### Sample wikirevision_0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: This is consistent across multiple sources, including the older and newer Wikipedia revisions of the Labour Party leadership page, both of which confirm his election on the same date

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This name change was confirmed by the Honourable Court of Haryana in 2016 the city officially changed its name from Gurgaon to Gurugram

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The newer Wikipedia revision and the disambiguation page both consistently use Gurugram as the official name

### Sample wikirevision_0105

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Bangalore was officially called Bangalore until 1 November 2014, when the city officially changed its name to Bengaluru

### Sample wikirevision_0105

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: This change was confirmed by the newer Wikipedia revision of Bangalore, which redirects to Bengaluru the main article on Bengaluru explicitly states the name change occurred on that date

### Sample wikirevision_0105

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: As a result, Bangalore is now officially called Bengaluru

### Sample wikirevision_0109

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: The current Prime Minister of Canada is Mark Carney, who assumed office on 14 March 2025

### Sample wikirevision_0109

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: He is the 24th person to serve in the role and is affiliated with the Conservative Party of Canada

### Sample wikirevision_0109

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
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
- **Claim**: The rebranding was announced during the company's October 2021 earnings call and took effect on January 1, 2022

### Sample wikirevision_0111

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: As a result, Facebook is no longer its own parent company, but rather a subsidiary operated by Meta Platforms

### Sample wikirevision_0112

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: The current President of Indonesia is Prabowo Subianto, who took office on 20 October 2024

### Sample wikirevision_0112

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: He is the eighth president of Indonesia and serves as both head of state and head of government

### Sample wikirevision_0112

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The presidency was established in 1945 with Sukarno as the first president

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
- **Supporting Docs Found**: d4
- **Claim**: The current 2026 French Open men's singles champion is Carlos Alcaraz, who defeated Jannik Sinner in the final to win his second French Open title and fifth major overall

### Sample wikirevision_0120

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: This result is corroborated across multiple sources, with the 2026 tournament edition confirming him as the current singles champion

### Sample wikirevision_0120

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: While two-time defending champion Carlos Alcaraz withdrew from the 2026 event due to a wrist injury, the 2025 edition remains the most recent completed tournament where he claimed the title

### Sample wikirevision_0121

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: The current President of Argentina is Javier Milei, who took office on 10 December 2023

### Sample wikirevision_0121

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: He is the 44th President of Argentina and belongs to the political party Unión por el Cambio

### Sample wikirevision_0121

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: This is consistent across multiple up-to-date sources, including the Wikipedia article on the President of Argentina, which also notes that the office carries the formal title 'President of the Argentine Nation'

### Sample wikirevision_0123

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Jannik Sinner

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
- **Claim**: He is the 31st person to hold the position since the office was created in 1901

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
- **Claim**: She is the first female prime minister in the country's history

### Sample wikirevision_0132

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of the Prime Minister of Japan page, as well as the list of prime ministers of Japan

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
- **Supporting Docs Found**: d3
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
- **Claim**: Jannik Sinner

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
- **Claim**: He is the 17th President of the Philippines and serves as both head of state and head of government

### Sample wikirevision_0149

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: This is consistent across multiple sources, including the older and newer Wikipedia revisions of the President of the Philippines page, as well as the list of presidents of the Philippines

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
- **Claim**: This result is consistent across multiple sources, including the Wikipedia pages for the 2025 US Open and the main US Open article, which identify 2025 as the most recent championship

### Sample wikirevision_0151

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Australia

### Sample wikirevision_0152

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: The latest Ballon d'Or winner is Ousmane Dembélé, who claimed his first award at the 2025 ceremony

### Sample wikirevision_0152

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: This is confirmed by the Wikipedia pages on the Ballon d'Or, which list the 2025 edition as the 69th ceremony and name Dembélé as the holder

### Sample wikirevision_0152

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Older sources, including the 2024 Ballon d'Or page, are superseded by this more recent result

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
- **Claim**: Claudia Sheinbaum is the latest President of Mexico, having assumed office on 1 October 2024

### Sample wikirevision_0154

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: This is confirmed across multiple sources, including the high-credibility Wikipedia article on the President of Mexico, which also notes she is the first woman and first Jewish person to hold the office

### Sample wikirevision_0155

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Facebook's parent company is Meta Platforms

### Sample wikirevision_0156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: The current President of the Philippines is Bongbong Marcos (Ferdinand R. Marcos Jr.), who assumed office on June 30, 2022

### Sample wikirevision_0156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: He is the 17th President and serves as both head of state and head of government

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

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by the official Wikipedia revision that superseded the older version in your query

### Sample wikirevision_0157

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: She is the 15th President of India and holds the highest constitutional office of the country

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
- **Claim**: This is consistent across multiple up-to-date sources, including the Wikipedia article on the President of Indonesia, which also notes that the presidency was established in 1945 as part of the country's founding

### Sample wikirevision_0161

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Gurgaon is officially called Gurugram

### Sample wikirevision_0161

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This name change was confirmed by the Government of Haryana in 2016 the city is now known officially as Gurugram

### Sample wikirevision_0161

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: As a result, Gurgaon is no longer the official name of the city

### Sample wikirevision_0162

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Argentina

### Sample wikirevision_0165

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: The current President of the United States is Donald Trump, who assumed office on January 20, 2025

### Sample wikirevision_0165

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is confirmed across multiple authoritative sources, including the newer Wikipedia revision of the President of the United States article, which supersedes the older revision from July 2025

### Sample wikirevision_0165

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
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
- **Claim**: Claudia Sheinbaum is the current President of Mexico, having assumed office on 1 October 2024

### Sample wikirevision_0167

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: She is the 66th president of Mexico and the first woman to hold the office

### Sample wikirevision_0167

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple up-to-date sources, including the newer Wikipedia revision of the President of Mexico article and her own Wikipedia biography

### Sample wikirevision_0170

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: The current French Open men's singles champion is Carlos Alcaraz, who defeated Jannik Sinner in the 2025 final to win his second French Open title and fifth major overall

### Sample wikirevision_0170

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: This result is corroborated across multiple sources, including the Wikipedia pages for both the 2025 French Open and the current state of the tournament

### Sample wikirevision_0170

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: While the 2026 French Open is mentioned in some sources, it has not yet produced a champion, as the tournament article notes that Alcaraz himself withdrew before the start of the 2026 edition due to a wrist injury

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
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: The latest French Open men's singles champion is Carlos Alcaraz, who defeated Jannik Sinner in the 2025 final to win his second French Open title and fifth major overall

### Sample wikirevision_0172

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: This result is corroborated across multiple sources, with the 2026 edition page also listing Alcaraz as the current singles champion the 2025 edition page providing full details of his victory


================================================================================

*Report generated by CATS v2.0*
