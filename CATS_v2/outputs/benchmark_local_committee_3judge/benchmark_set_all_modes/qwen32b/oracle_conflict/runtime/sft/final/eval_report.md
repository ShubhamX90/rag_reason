# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 128 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.966 (over 736 samples)

**GR F1** *(used in CATS)*: 0.979

**Behavior Adherence**: 0.822 (over 608 applicable samples)

**Factual Grounding**: 0.005 (over 608 applicable samples)

**Single-Truth Recall**: 0.752 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.639

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.979
- **Precision**: 1.000
- **Recall**: 0.959
- **Accuracy**: 0.966
- TP=583, FP=0, FN=25, TN=128

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.837
- **Abstain Recall**: 1.000
- **Abstain F1**: 0.911
- **Specificity**: 0.959
- Abstain TP=128, FP=25, FN=0, TN=583


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (57 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.991
- **GR F1** *(used in CATS)*: 0.993
- **Behavior**: 0.968 (n=154)
- **Grounding**: 0.006 (n=154)
- **Recall**: 0.851 (n=154)
- **CATS**: 0.705

### Type 2: Complementary Info

- **Samples**: 221 (45 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.959
- **GR F1** *(used in CATS)*: 0.974
- **Behavior**: 0.943 (n=176)
- **Grounding**: 0.008 (n=176)
- **Recall**: 0.673 (n=156)
- **CATS**: 0.649

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.982
- **GR F1** *(used in CATS)*: 0.989
- **Behavior**: 0.510 (n=96)
- **Grounding**: 0.005 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.502

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.975
- **GR F1** *(used in CATS)*: 0.986
- **Behavior**: 0.717 (n=145)
- **Grounding**: 0.000 (n=145)
- **Recall**: 0.746 (n=140)
- **CATS**: 0.612

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.784
- **GR F1** *(used in CATS)*: 0.879
- **Behavior**: 0.865 (n=37)
- **Grounding**: 0.000 (n=37)
- **Recall**: 0.689 (n=37)
- **CATS**: 0.608


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

### Sample conflictingqa_04e1627e9fc5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d3
- **Claim**: Nematodes generally do not increase soil fertility directly; instead, they mediate nutrient mineralization and cycling, serving as key indicators of soil health

### Sample conflictingqa_05b33f4ca156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Yes — many salamanders are poisonous to touch, as nearly all species possess toxins in their skin that can cause serious illness if ingested; however, a few specific species such as tiger salamanders and yellow-spotted salamanders are generally considered safe to handle gently the main risk for most salamanders is bacterial contamination rather than lethal poisoning

### Sample conflictingqa_060e5f26c453

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d5
- **Claim**: Yes — the Great Pacific Garbage Patch is larger than Texas

### Sample conflictingqa_0717d0e62f3b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: Fashion designs can receive copyright protection, but only for specific elements—such as graphic patterns, surface designs logos—that qualify as pictorial, graphic sculptural works under U.S. copyright law

### Sample conflictingqa_0717d0e62f3b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: In contrast, the overall configuration or shape of clothing is generally not protected in most countries including the United States, apparel is classified as a functional item subject to lesser protection than standalone creative works

### Sample conflictingqa_0717d0e62f3b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The U.S. Copyright Office has historically treated fashion designs as eligible for a separate 'sui generis' protection similar to that afforded to vessel hulls, rather than standard copyright, though this distinction is a matter of ongoing legislative debate

### Sample conflictingqa_07e290a954ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The evidence on St. John's wort is mixed and depends on the severity of depression

### Sample conflictingqa_07e290a954ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d3
- **Claim**: Multiple clinical studies, including a systematic review cited by the Psychopharmacology Institute, suggest that St. John's wort is more effective than placebo and approximately equivalent to low-dose tricyclic antidepressants or standard SSRIs, particularly for mild to moderate depression

### Sample conflictingqa_07e290a954ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: Scientific research published on MDPI and ScienceDirect further supports its comparable efficacy and safety relative to SSRIs for mild-to-moderate depressive symptoms the drug interaction database Stockley's notes it is licensed for depression in multiple European countries

### Sample conflictingqa_07e290a954ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: However, WebMD and the Cochrane Collaboration highlight that larger studies, including one sponsored by the National Center for Complementary and Alternative Medicine (NCCAM), found that St. John's wort was not more effective than a placebo for moderately severe major depression the Royal Australian College of General Practitioners notes that while some clinical trials show positive results, this evidence is yet to be fully confirmed

### Sample conflictingqa_07e290a954ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Overall, while St. John's wort may be used as an adjunct or alternative for mild depression, it cannot be considered a proven treatment for moderate or severe depression without further confirmation

### Sample conflictingqa_08757808c000

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5
- **Claim**: Weight lifting does not cause high blood pressure, but it can cause temporary spikes during individual lifts long-term training may help reduce blood pressure

### Sample conflictingqa_0875b5f3262a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: No, Allen Ginsberg's *Howl* was famously judged as **not** obscene during the 1957 trial this legal ruling stands as a landmark for literary freedom

### Sample conflictingqa_0a05aabca56a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4, d3
- **Claim**: Yes — anime is a form of cartoon originating in Japan, distinguished primarily by its country of origin, visual style broader subject matter

### Sample conflictingqa_0ad05303220b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: Judaism is not a race; it is a religion and ethnicity

### Sample conflictingqa_0c3c7b487766

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d3
- **Claim**: Iodine supplementation does not universally cause thyroid problems; risks depend heavily on dose, individual susceptibility context

### Sample conflictingqa_0c3c7b487766

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d3
- **Claim**: High-quality evidence indicates that excess iodine intakes may precipitate hyperthyroidism, hypothyroidism, goiter thyroid autoimmunity in some people—particularly those with preexisting thyroid disease or a history of iodine deficiency—while most healthy individuals tolerate moderate increases without harm

### Sample conflictingqa_0c3c7b487766

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: Research also shows that excessive supplementation during pregnancy may elevate cord blood TSH and is associated with increased risk of hypothyroidism and autoimmune thyroiditis in susceptible populations that once adequate intake is achieved, additional iodine does not enhance hormone production and may increase dysfunction risk

### Sample conflictingqa_0dba017da71c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Yes — the world's largest known organism is a fungus, specifically Armillaria solidipes (Honey Fungus), which scientists discovered in the Pacific Northwest spanning approximately 5.5 kilometres

### Sample conflictingqa_0dba017da71c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: This, identifying the record-holder as Armillaria ostoyae (also known as the Humongous Fungus) in Oregon's Malheur National Forest, which extends across some 2,385 acres

### Sample conflictingqa_111bef268aa6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: The evidence presents conflicting findings on this question

### Sample conflictingqa_111bef268aa6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Some sources report that peeling an apple reduces dietary fiber by approximately 50% and vitamin C by around 30%, while others claim that peeling does not significantly reduce vitamin C content and that much of the nutrient value remains in the flesh

### Sample conflictingqa_111bef268aa6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: Apple peels are known to contain higher concentrations of certain nutrients—such as vitamin E, vitamin K, iron, folate antioxidants—meaning that peeling does remove at least some of the apple's total nutritional value, but the overall loss depends on which nutrients are prioritized

### Sample conflictingqa_114c06976f62

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: The Church of the Flying Spaghetti Monster is generally considered a parody religion or social movement rather than a traditional faith

### Sample conflictingqa_11c5ef7c4545

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: The retrieved evidence presents competing views

### Sample conflictingqa_11c5ef7c4545

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Some sources argue that anyone can become an entrepreneur if they are willing to learn, adapt face risks , while others argue that entrepreneurship is not for everyone because it requires rare skills, mindsets risk tolerances that not all individuals possess or can develop

### Sample conflictingqa_11c5ef7c4545

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: There is no single universal consensus — the answer depends on philosophical and methodological assumptions about whether entrepreneurship is a teachable practice or a innate calling

### Sample conflictingqa_151865dc414b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d3
- **Claim**: The retrieved evidence indicates that pulsatile tinnitus can often be cured or resolved when its underlying cause is identified and treated

### Sample conflictingqa_151865dc414b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: Northwell Health notes that conditions such as venous sinus stenosis, tumors, arteriovenous malformations high blood pressure — all common causes of pulsatile tinnitus — can be addressed with procedures (such as venous sinus stenting) or medications that eliminate or significantly reduce symptoms

### Sample conflictingqa_151865dc414b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3
- **Claim**: However, Chicago Hearing Services adds a note of nuance: when pulsatile tinnitus does not have a treatable cause when treatment is inappropriate or incomplete, the focus shifts to managing its effects through sound therapy, masking other strategies

### Sample conflictingqa_151865dc414b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3
- **Claim**: In summary, while a universal cure does not exist, the evidence strongly suggests that targeted treatment of the root cause is the most effective path to resolution for the majority of patients

### Sample conflictingqa_1d741779a31e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3
- **Claim**: he evidence is mixed

### Sample conflictingqa_1d741779a31e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Some sources indicate that artificial sweeteners are safe for diabetics and can help manage blood sugar, while others suggest that certain artificial sweeteners may worsen glycemic control, alter gut microbiota potentially increase the risk of type 2 diabetes and cardiovascular disease

### Sample conflictingqa_1e9c6cd8e30b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4, d3
- **Claim**: Yes — palm oil production causes serious environmental harm through deforestation, biodiversity loss pollution; however, sustainably certified palm oil may mitigate these risks

### Sample conflictingqa_21f33954c8af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Dog breeding is not universally unethical, but some practices are

### Sample conflictingqa_220ec09fbb2c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: Cows technically have one stomach that is divided into four distinct compartments — the rumen, reticulum, omasum abomasum — rather than four separate stomachs

### Sample conflictingqa_237adb87065f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d3
- **Claim**: The retrieved evidence indicates that the Silurian period marks the first appearance of simple vascular plants on land, making it a critical period for the birth of land plants, though some researchers argue that plant origins trace back to the preceding Ordovician period

### Sample conflictingqa_2395695f1604

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d3
- **Claim**: The majority of scientific research, including a peer-reviewed 2005 JACCN review and a 2012 BC Children's Hospital study cited by ENT specialists, concludes that milk and dairy products do not cause increased mucus production — the sensation is caused by mouth mucins forming aggregates with fat emulsions

### Sample conflictingqa_2395695f1604

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: However, a 2009 clinical study on asthma patients found a long-standing association between excessive milk consumption and increased respiratory tract mucus production, suggesting context-dependent effects , while a 2021 critical review notes dairy may affect sensory perception or mucus viscosity without actually initiating production

### Sample conflictingqa_24c25ef3a801

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5
- **Claim**: The retrieved evidence is mixed

### Sample conflictingqa_24c25ef3a801

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5
- **Claim**: Some sources argue that money can buy happiness, but usually only up to a point — for example, one study found that emotional well-being rises logarithmically with income, meaning that each additional dollar adds less happiness than the last levels off entirely beyond about $75,000–$100,000 per year ; other evidence suggests that strategic spending on experiences, prosocial goods small indulgences can deliver meaningful happiness gains even at higher income levels

### Sample conflictingqa_24c25ef3a801

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5
- **Claim**: On the other hand, a majority of experts and studies conclude that money is not a reliable pathway to happiness except indirectly — for instance, by facilitating social connections, experiences personal growth by reducing anxiety caused by financial insecurity — and that non-monetary factors such as relationships, health values matter far more

### Sample conflictingqa_24fa0020a521

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d3
- **Claim**: Most healthy children do not need a daily multivitamin if they eat a well-balanced diet, according to the American Academy of Pediatrics (AAP) and Mayo Clinic

### Sample conflictingqa_24fa0020a521

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Vitamin D and iron are the notable exceptions: the AAP specifically recommends that all breastfed infants receive 400 IU of vitamin D daily starting shortly after birth that children over 1 year old receive 600 IU daily , with vitamin D also beneficial for children with limited sun exposure or dark skin

### Sample conflictingqa_24fa0020a521

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d3
- **Claim**: For most children, a varied diet that includes fruits, vegetables, whole grains, dairy or fortified alternatives protein sources is sufficient to meet nutritional needs the AAP cautions that megadoses of certain vitamins can be toxic ; supplements are generally only recommended when a child's diet is restricted due to picky eating, food allergies dietary restrictions like veganism

### Sample conflictingqa_2500bfd10608

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The safety of fluoride in drinking water is genuinely contested

### Sample conflictingqa_2500bfd10608

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: High-credibility sources note that fluoride is considered safe at drinking water concentrations of 0.7 mg/L or lower and that the CDC promotes community water fluoridation as beneficial for dental health , while the NIH concludes that higher fluoride levels are linked to lowered IQ in children and Harvard researchers highlight potential neurotoxic effects at excessive doses

### Sample conflictingqa_2500bfd10608

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: On the other hand, emerging research and advocacy groups argue that even typical fluoridated water may carry neurological risks, particularly for infants and children, leading some experts to call for a reassessment of the practice

### Sample conflictingqa_2500bfd10608

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Overall, the evidence presents a nuanced picture in which fluoride's safety depends heavily on concentration and population context, rather than being universally benign or dangerous

### Sample conflictingqa_264e97231363

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: Not really — chlorine doesn't turn hair green, but copper (often from algaecides) can bind to hair and turn it green when oxidized chlorine may accelerate that process

### Sample conflictingqa_287debe5bb90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The retrieved evidence presents competing views

### Sample conflictingqa_287debe5bb90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Some sources argue that meaningful knowledge requires moving beyond verbal and conceptual thought, while others argue that the mind is all we can truly know

### Sample conflictingqa_288cd1b45aab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: Research and expert opinion are divided on whether wrist rests minimize wrist pain during typing

### Sample conflictingqa_288cd1b45aab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3
- **Claim**: Some studies and sources report moderate benefits, such as reducing wrist extension angles by 10–15 degrees and decreasing reported discomfort by up to 30% when used correctly ; others note that wrist rests can reduce muscle fatigue in the upper arm by up to 32% and encourage a more neutral wrist position

### Sample conflictingqa_288cd1b45aab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: However, some experts argue that wrist rests are not universally beneficial and that placing wrists firmly on the rest while typing is counterproductive, potentially compressing nerves and tendons in the carpal tunnel that wrist rests are not necessary for good ergonomics and can even create pressure marks and tingling when misused

### Sample conflictingqa_288cd1b45aab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3
- **Claim**: Overall, the evidence suggests that wrist rests may help when used properly—allowing the heels of the palms to rest only during pauses and keeping wrists hovering while typing—but may actually increase harm if used incorrectly

### Sample conflictingqa_29f69e16a0c3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d5
- **Claim**: Yes — flowers communicate with bees using both electrical fields and acoustic vibrations, as well as through fragrances, colors patterns

### Sample conflictingqa_2c0ea18839df

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: Epigenetic changes can be hereditary in some organisms and under certain conditions, a phenomenon known as transgenerational epigenetic inheritance

### Sample conflictingqa_2c0ea18839df

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: Research has shown that modifications such as DNA methylation and histone marks can be transmitted from parents to offspring and even to grandoffspring the once-radical idea of epigenetic inheritance now has a growing body of evidence behind it

### Sample conflictingqa_2c0ea18839df

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: However, the extent of heritability is debated: some researchers argue that the widely accepted mechanism of two rounds of demethylation during mammalian reproduction wipes out most epigenetic information, making true transgenerational inheritance nearly impossible except at rare escape sites , while others point to documented cases where specific marks—such as maternal DNA methylation and histone modifications—can evade reprogramming and be reliably inherited

### Sample conflictingqa_311fca0928d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The retrieved evidence presents conflicting views

### Sample conflictingqa_311fca0928d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: Some sources argue that IPv6 is fundamentally more secure because it mandates IPsec support and offers improved address space and header design, while others argue that IPv6 is not inherently more or less secure than IPv4 and that most security incidents result from design and implementation flaws rather than protocol weaknesses

### Sample conflictingqa_34610226ee3f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: No — dinosaurs died out 65 million years ago, far beyond the ~1 million year threshold after which DNA degrades entirely, making de-extinction via cloning impossible

### Sample conflictingqa_34fef928d452

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: Yes — Archaeopteryx was capable of powered flight, according to a 2025 study that confirmed the animal had asymmetric feathers and tertial feathers, which are both critical adaptations for generating lift

### Sample conflictingqa_35491baf4f4b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
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
- **Claim**: Additionally, research indicates that the moon once had a thicker, transient atmosphere approximately 3 to 4 billion years ago, formed by intense volcanic eruptions that spewed gases faster than they could escape to space this ancient atmosphere persisted for about 70 million years before being lost

### Sample conflictingqa_3601f7480501

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Research and opinion on unlimited vacation time are mixed

### Sample conflictingqa_3601f7480501

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: Some studies, such as one cited by the Chamber of Commerce, indicate that taking vacations can increase productivity, improve job satisfaction reduce stress , while limited empirical evidence from Cornell University suggests that employees with unlimited PTO take roughly the same amount of time off as those with traditional accrual-based plans

### Sample conflictingqa_3601f7480501

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: However, other research presents a paradoxical finding: a Namely study found that employees with unlimited PTO took an average of only 13 days off per year compared to 15 days under traditional plans some early adopters reported higher burnout rates due to employees feeling pressure to limit their time away

### Sample conflictingqa_3601f7480501

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3
- **Claim**: Additional research from the Society for Human Resource Management (SHRM) warns that the lack of a defined minimum can lead to employees taking less time off than needed , while a2b Testing's survey found that only 22% of respondents with unlimited PTO took more than 20 days off in a year

### Sample conflictingqa_3601f7480501

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Employee perspectives further diverge—some appreciate the flexibility and trust , while others report anxiety about how much time is appropriate to take —suggesting that unlimited PTO's value is highly conditional on company culture, communication practices individual mindset

### Sample conflictingqa_37ab7146eb9b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: Robots can be programmed to detect and respond to stimuli analogous to pain, but they cannot genuinely feel pain as humans do

### Sample conflictingqa_37ebad668bb7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3
- **Claim**: The retrieved evidence consistently indicates that data is nearly always required for machine learning, with d3 explicitly stating that ML requires training on historical data and d5 emphasizing that quality data is necessary for models to operate efficiently

### Sample conflictingqa_37ebad668bb7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: However, the extent of data needed varies significantly depending on the algorithm, problem complexity model capacity — d1 recommends a '10 times rule' for small models, while d2 suggests that better algorithms can sometimes overcome data limitations d4 warns of diminishing returns as more data is added

### Sample conflictingqa_37ebad668bb7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Overall, while data is not optional in practice, the optimal amount is highly context-dependent rather than universally fixed

### Sample conflictingqa_39fe5c441657

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The retrieved evidence presents competing views

### Sample conflictingqa_39fe5c441657

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5
- **Claim**: Some sources argue that astral projection is real as an experience but lack physical evidence to support it as a literal physical event; others characterize popular claims as hallucination or suggest the phenomenon may be explained by brain activity during REM sleep

### Sample conflictingqa_3afd7f725cb4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: The answer depends on whom you ask; experts and advocates affirm that audiobooks are genuine reading—used by libraries, schools book-tracking platforms alike—while a notable minority of readers and even some professionals persist in doubting their legitimacy

### Sample conflictingqa_3bd13d25098b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The Moon is generally considered geologically inactive compared to Earth, as most volcanic activity ceased around 3 billion years ago and the core dynamo field disappeared between 2.5 and 1 billion years ago

### Sample conflictingqa_3bd13d25098b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: However, recent research presents conflicting but credible evidence suggesting ongoing activity: NASA scientists identified small mare ridges on the far side that formed within the last 200 million years and may still be forming today , while Indian researchers detected lobate scarps and debris avalanches at the lunar south pole, consistent with ongoing tectonic deformation

### Sample conflictingqa_3bd13d25098b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3
- **Claim**: Additional studies note that the Moon is not entirely without active geology, as meteorite impacts and chemical interactions with the solar wind do induce localized activity a 2025 study hints that the lunar subsurface may be more dynamically active than previously believed

### Sample conflictingqa_3c835387fe6d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Yes, though indirectly — the Komodo dragon evolved in Australia and lived there for approximately four million years before dispersing to Indonesia; research indicates it became locally extinct in Australia around 300,000 years ago

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: It depends on whether the artificial tree is reused for enough years; experts differ on the break-even point

### Sample conflictingqa_3f3c3399259a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d3
- **Claim**: The relationship between fish oil and heart disease risk is genuinely contested in the evidence

### Sample conflictingqa_3f3c3399259a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Some clinical trials have found that high doses of EPA (from fish oil) may lower cardiovascular event risk, though with tradeoffs such as increased atrial fibrillation risk the broader consensus from large reviews and medical institutions is that fish oil supplements do not clearly prevent heart attack or stroke compared to existing standard treatments

### Sample conflictingqa_3f3c3399259a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: Research is similarly mixed across different patient groups and outcome measures, though omega-3 from fish themselves (as opposed to supplements) is generally regarded more favorably

### Sample conflictingqa_411445406724

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Yes — cycads were once considered the dominant Mesozoic plants, but recent research suggests other plant groups were more abundant

### Sample conflictingqa_42d60ecaee9f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: Emoji are commonly described as a new form of language, but the scholarly consensus is more nuanced

### Sample conflictingqa_42d60ecaee9f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: On one hand, some researchers argue that emoji are not a distinct new language but rather an evolution of older visual communication systems—such as hieroglyphs or cuneiform—adapting these ancient forms to the digital age ; others note that emoji function as a multi-modal addition to existing languages, supplementing textual content with visual tone and nuance in much the same way that gestures accompany spoken language

### Sample conflictingqa_42d60ecaee9f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4, d3
- **Claim**: On the other hand, linguists generally emphasize that emoji currently lack the key hallmarks of a fully formed language—such as a standardized grammar, consistent mutual intelligibility a complete expressive vocabulary—leading to the conclusion that they do not yet qualify as a new language in the strictest sense ; some experts go further, classifying emoji not as a language at all, but as elements of writing systems or even as regressive linguistic phenomena that replace more complex forms of expression

### Sample conflictingqa_4317242e485c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: Yes — the IUCN states that well-managed trophy hunting can provide revenue and incentives for people to conserve and restore wild populations, maintain land for conservation protect wildlife from poaching

### Sample conflictingqa_4786f87b62be

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The retrieved evidence presents competing views

### Sample conflictingqa_4786f87b62be

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Some sources argue that the gender wage gap is real and measurable, while others argue that it is a myth or exaggerated because it reflects factors like occupation and parenting choices rather than direct discrimination

### Sample conflictingqa_517b918aa677

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The constitutional status of prayer in U.S. public schools is nuanced and contested

### Sample conflictingqa_517b918aa677

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The U.S. Supreme Court has repeatedly held that officially organized or government-endorsed prayer in schools is coercive and therefore violates the Establishment Clause of the First Amendment , while the Court has also affirmed that students retain individual First Amendment rights to pray privately and engage in religious expression ; under the Trump administration's 2026 Department of Education guidance, schools are required to maintain a stance of neutrality toward faith and permit students and employees to pray on the same terms as other expressive activities, as long as no coercion or endorsement is involved the Equal Access Act further requires schools to allow student-led religious clubs on the same terms as other student groups

### Sample conflictingqa_52181cd092aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: Yes — the Great Pacific Garbage Patch is roughly twice the size of Texas, though the ratio varies by depth and sampling methodology

### Sample conflictingqa_5233eab573e5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Yes — there are more tigers in captivity than in the wild

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: The retrieved evidence presents competing views

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: Some sources argue that software patents can protect valuable innovations and provide legal exclusivity, while others argue that software is too abstract or that patents stifle innovation

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4
- **Claim**: The evidence on whether bicarbonate supplementation prevents progression in chronic kidney disease (CKD) is mixed and depends on disease stage and dose

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: Some studies suggest a benefit: a prospective study found that sodium bicarbonate slowed the rate of creatinine clearance decline in stage 4 CKD a peer-reviewed study noted that oral bicarbonate supplementation may slow progression in early CKD stages by reducing urinary profibrotic biomarkers ; additionally, a research gate analysis reported that bicarbonate supplementation can slow CKD progression to ESRD and improve nutritional status

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: However, other studies present conflicting findings — a randomized controlled trial found no effect of bicarbonate administration on kidney failure progression the KDIGO clinical practice guidelines note that the use of sodium bicarbonate in CKD patients with normal serum bicarbonate levels remains uncertain , underscoring the need for further investigation

### Sample conflictingqa_56fd6bf22253

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Adenoids can potentially grow back after removal, but it is generally considered rare, especially in older children or when surgery is thorough

### Sample conflictingqa_57190bca6f7a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: The 1815 Mount Tambora eruption was among the deadliest in recorded history, though it is not unambiguously the single deadliest eruption

### Sample conflictingqa_57190bca6f7a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: It killed at least 10,000 people directly via pyroclastic flows and tsunamis approximately 80,000 more from disease and famine, yielding a combined figure of roughly 90,000 deaths on Sumbawa, Lombok Bali alone

### Sample conflictingqa_57190bca6f7a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: However, some sources characterize it as the most powerful or destructive eruption in recorded history rather than explicitly the deadliest certain high-credibility sources (e.g., Wikipedia's main volcano page) list it as one of the top ten deadliest eruptions alongside events like Krakatoa and Mount Pelee

### Sample conflictingqa_5e7a6a2debfd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Male bees do not work in the sense of performing labor tasks within the hive; d1 and d2 state that worker bees (females) do all the work, while d3 adds that males may still incidentally transfer pollen

### Sample conflictingqa_613a0093714b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The phrase is popularly associated with 17th century England, but scholars note this connection is a theory rather than a confirmed fact

### Sample conflictingqa_62b1aff6586d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: The ozone hole is healing, but has not fully healed

### Sample conflictingqa_63fde268aa8c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Religious and philosophical traditions such as Sanatana Dharma and Cartesian dualism assert that the mind is separate from the body, while some scientists and philosophers argue that the mind-body distinction is a fiction and that thoughts, sensations movements arise from the same psychobiology

### Sample conflictingqa_676188bf8139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5
- **Claim**: Yes, the Chinese Lantern Festival includes honoring deceased ancestors, though this is not its sole purpose

### Sample conflictingqa_6988dd820a61

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The retrieved evidence is conflicting

### Sample conflictingqa_6988dd820a61

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Some researchers argue that major earthquakes are more likely during full and new moons because tidal stresses are highest at those times, with a 2016 Nature Geoscience study finding that high tidal stress was often followed by major quakes

### Sample conflictingqa_6988dd820a61

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: However, a 2016 study published in Seismological Research Letters, which analyzed 204 magnitude 8 or higher earthquakes dating back to the 1600s, concluded that earthquake incidence had no relationship to the moon's position and described the data as 'completely random'

### Sample conflictingqa_6988dd820a61

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Most seismologists, including the USGS, consider the correlation between earthquake occurrence and lunar phase to be coincidental rather than causal

### Sample conflictingqa_6d7279b0a8ed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: No, the Gutenberg Bible was not the first book printed with movable type

### Sample conflictingqa_6d7279b0a8ed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: While it is widely recognized as the earliest major European book produced using mass-produced metal movable type, significant textual precedents exist

### Sample conflictingqa_6d7279b0a8ed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: A Korean Buddhist text called the Jikji, printed in 1377 using wooden movable type, predates the Gutenberg Bible by approximately 78 years and represents the oldest known surviving book printed with movable type

### Sample conflictingqa_6d7279b0a8ed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Additionally, Chinese and Korean inventors were producing printed books using movable type centuries before Gutenberg was born, though these earlier experiments did not achieve the same commercial or cultural scale as Gutenberg's groundbreaking work in the West

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: The evidence is divided on whether split ends can be repaired

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: Some sources argue that split ends cannot be permanently repaired because hair is dead tissue that cannot regenerate that most product claims of repair are temporary fixes that mask rather than restore the damage

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, other sources note that certain bond-building products may help repair some of the chemical bonds in damaged hair, potentially reducing the appearance and incidence of split ends over time, though they do not fully restore the hair shaft to its original state

### Sample conflictingqa_6fe31cd2ef65

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: It depends on the context — rolling is required for double RR and R at the start of words, but not for single R in the middle of words

### Sample conflictingqa_734859937e46

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: yes

### Sample conflictingqa_73560dfab1ae

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d5
- **Claim**: The evidence on whether high doses of vitamin C alleviate common cold symptoms is mixed and contested

### Sample conflictingqa_73560dfab1ae

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: High-quality clinical research, including a Cochrane review cited by the National Institutes of Health, found that vitamin C does not significantly reduce cold duration or severity in the general population, though it may modestly reduce symptom duration in specific groups such as marathon runners or skiers exposed to cold environments

### Sample conflictingqa_73560dfab1ae

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: A separate peer-reviewed meta-analysis published in BMC Public Health did find that vitamin C significantly decreased the severity of common colds by 15% compared to placebo had a greater effect on severe symptoms than mild ones , suggesting a potential benefit for those with more intense cold experiences

### Sample conflictingqa_73560dfab1ae

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: However, other sources caution that most people already obtain sufficient vitamin C from their diets and that high-dose supplements may carry risks such as increased kidney stone formation in certain individuals , while some anecdotal reports and older studies continue to advocate for vitamin C as a valuable cold-fighting tool

### Sample conflictingqa_747727772a30

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d3
- **Claim**: Bees can fly in the rain, but only reluctantly and usually only in light rain or under pressing circumstances such as defending the hive

### Sample conflictingqa_76956c2fba7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The relationship between saturated fats and heart disease risk is genuinely contested in the scientific literature

### Sample conflictingqa_76956c2fba7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: Some studies, including research presented at the European Society of Cardiology Congress 2024, suggest that a diet high in saturated fat may increase cardiovascular disease risk factors such as liver fat and cholesterol levels, potentially raising the risk of heart disease regardless of weight gain ; additionally, the American Heart Association notes that saturated fats increase LDL cholesterol, which is associated with plaque buildup in arteries

### Sample conflictingqa_76956c2fba7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: However, other analyses—such as a systematic review published in the British Journal of Sports Medicine—found no significant association between saturated fat consumption and all-cause mortality Examine.com's examination of meta-analyses noted that RCTs and observational studies do not consistently support the notion that saturated fat strongly increases the risk of heart disease

### Sample conflictingqa_76956c2fba7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: Overall, while saturated fats are widely advised against due to their cholesterol-raising effects, the evidence remains mixed and contested

### Sample conflictingqa_7998e59d9b97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: Conventionally grown food is higher in certain pesticide residues, but organic food is generally more expensive and has lower yields — by about 20–25% on average

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4, d3
- **Claim**: The retrieved evidence presents competing views

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: Some sources argue that the Catholic Church is the one true church on the basis of biblical foundations and apostolic succession, while others argue that Scripture alone determines which church is true and that the Catholic Church's claims lack biblical support

### Sample conflictingqa_7cf85109a70d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: No, brass is not more durable than bronze

### Sample conflictingqa_7cf85109a70d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Bronze is generally harder and more wear-resistant than brass, making it more durable in demanding applications

### Sample conflictingqa_80857a692531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3
- **Claim**: Farmed and wild salmon are broadly similar in nutritional value, with both serving as excellent sources of protein, omega-3 fatty acids vitamins

### Sample conflictingqa_80857a692531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: However, some studies note that wild salmon tends to contain higher amounts of certain vitamins — such as vitamin D and vitamin A — and lower levels of contaminants like PCBs, while farmed salmon generally has a higher fat content that can accommodate more omega-3s

### Sample conflictingqa_80857a692531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3
- **Claim**: Overall, the consensus across high-credibility sources is that the differences are moderate and that farmed salmon remains a valuable, nutrient-rich food choice

### Sample conflictingqa_80baf25496cd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Multiculturalism may present both opportunities and challenges for social unity, depending on perspective and context

### Sample conflictingqa_8848765fc18a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5
- **Claim**: Spelunking is generally considered synonymous with caving, but some sources draw distinctions based on experience level or equipment standards

### Sample conflictingqa_894f4a4b9552

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d5
- **Claim**: The majority of scientific evidence strongly supports the existence of dark matter: observations of galaxy rotation curves, cluster dynamics, gravitational lensing the cosmic microwave background all indicate the presence of unseen mass influencing gravity

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence indicates that bird calls are generally unique to each species rather than to individual birds

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Calls serve specific functions like territorial defense, mating predator alarms birds can often recognize conspecific calls while responding to interspecies alarm calls as well

### Sample conflictingqa_8cf8ebb94554

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: The evidence on whether knee braces prevent injuries is mixed

### Sample conflictingqa_8cf8ebb94554

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: Some research, particularly in the context of contact sports like football, suggests that certain types of braces (such as prophylactic or functional braces) can reduce the risk of specific knee injuries or reinjuries the American Academy of Orthopaedic Surgeons notes that knee braces are used as prophylactic devices for contact sports

### Sample conflictingqa_8cf8ebb94554

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: However, major clinical guidelines and review articles caution that there is currently no conclusive evidence demonstrating that knee braces broadly prevent knee injuries in the general population their use is considered debatable without a strong rehabilitative or surgical indication

### Sample conflictingqa_8cf8ebb94554

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: Additionally, experts note that while knee braces may offer stability and pain relief during recovery, they cannot prevent all types of injuries and should not replace professional medical advice or a comprehensive rehabilitation program

### Sample conflictingqa_8efa53ba7c60

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d3
- **Claim**: The retrieved evidence consistently confirms that birds are descended from dinosaurs but clarifies that they are not direct descendants of T-Rex specifically

### Sample conflictingqa_8efa53ba7c60

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: Birds belong to the theropod group of dinosaurs, which includes T-Rex as a distant relative evolved from a more bird-like subgroup called maniraptorans

### Sample conflictingqa_9251c7a33ec5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: he evidence is mixed

### Sample conflictingqa_9251c7a33ec5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5
- **Claim**: Some research suggests that spaying or neutering can negatively affect long-term health by disrupting hormonal balances and potentially increasing the risk of conditions such as certain cancers, urinary incontinence hypothyroidism

### Sample conflictingqa_9251c7a33ec5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Other sources argue that these procedures offer substantial health benefits by eliminating the risk of reproductive cancers and hormone-driven diseases that potential drawbacks such as weight gain or behavioral changes are generally manageable

### Sample conflictingqa_9251c7a33ec5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The consensus leans toward the benefits outweighing the risks for most pets, but individual circumstances may vary some experts, particularly those focusing on male dogs, suggest that the scale of adverse effects compared to benefits is a legitimate concern

### Sample conflictingqa_9261438d6ee2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Uncertainty — some researchers assert fish feel pain because they respond to noxious stimuli and possess nociceptors, while others argue that fish behavior and neuroanatomy are sufficiently different from humans that these responses do not constitute genuine pain

### Sample conflictingqa_9275911a2961

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Yes, certain antacids can cause kidney stones — specifically, antacids containing calcium or magnesium can lead to kidney stone formation when used in excess

### Sample conflictingqa_9275911a2961

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: Calcium-containing antacids pose the primary risk, as high calcium intake can cause hypercalcemia and subsequently kidney stones, particularly in susceptible individuals or when combined with calcium supplements

### Sample conflictingqa_9275911a2961

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Magnesium-containing antacids similarly carry a risk, as a case report documented a patient who developed magnesium-ammonium-phosphate kidney stones after ingesting 25–30 Gelusil tablets daily for years ; additionally, magnesium builds up in the kidneys when kidney function is reduced, further increasing stone risk

### Sample conflictingqa_9275911a2961

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d3
- **Claim**: It is worth noting that the risk is generally considered manageable at normal recommended doses, but prolonged or excessive use increases the hazard some research has also associated proton pump inhibitors (PPIs) and H2 blockers with elevated kidney stone risk , underscoring that the link between antacids and kidney stones varies by drug type and dosage

### Sample conflictingqa_962d8f5d5574

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: Based on the available evidence, all snakes appear to be capable of swimming, though the strength of support varies across sources

### Sample conflictingqa_962d8f5d5574

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: High-credibility sources note that swimming ability is well-documented for 525 snake species but remains unconfirmed for approximately 89% of all species, leaving the universal claim partially supported

### Sample conflictingqa_962d8f5d5574

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: Notably, even venomous and largely terrestrial snakes such as copperheads and diamondbacks have been observed swimming readily, suggesting that swimming is a nearly ubiquitous, though not formally verified, capability across snake species

### Sample conflictingqa_9b11b8e571aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d3
- **Claim**: No, gonorrhea is not only transmitted sexually

### Sample conflictingqa_9b11b8e571aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5
- **Claim**: While it is primarily spread through sexual contact—vaginal, anal oral sex—it can also be transmitted from mother to baby during childbirth, making sexual transmission 'extremely rare' but 'not entirely impossible'

### Sample conflictingqa_9b11b8e571aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Additionally, gonorrhea can spread through non-penetrative skin-to-skin genital contact and the exchange of bodily fluids, further expanding its transmission routes beyond intercourse alone

### Sample conflictingqa_9b73cb6cce52

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Giant African Land Snails can make acceptable pets for experienced handlers and certain situations — they are popular, low-maintenance fulfill educational or novelty value — but pose serious legal barriers in some jurisdictions and require meticulous, costly care

### Sample conflictingqa_9ceca2645833

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: The retrieved evidence presents competing views

### Sample conflictingqa_9ceca2645833

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Some argue that affirmative action is not reverse discrimination because it aims to redress historical injustices rather than punish individuals without cause

### Sample conflictingqa_9ceca2645833

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: Others argue that affirmative action can constitute reverse discrimination when it grants preferences to individuals from protected groups at the expense of qualified candidates from non-protected groups

### Sample conflictingqa_9ceca2645833

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Scholars and jurists continue to debate the legitimacy and scope of affirmative action policies

### Sample conflictingqa_a1e36a8db854

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: he evidence is divided

### Sample conflictingqa_a1e36a8db854

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Some studies and regulatory bodies (e.g., EPA, Health Canada, European agencies) conclude that glyphosate is not likely to be carcinogenic to humans when used according to directions, while others (e.g., IARC, Washington University Seattle Statement) consider the evidence sufficiently strong to classify it as a probable human carcinogen or justify urgent regulatory action

### Sample conflictingqa_a1e36a8db854

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: Additional research links glyphosate to neurodegenerative disorders, liver and kidney damage endocrine disruption, though these findings are similarly contested the degree of harm at typical exposure levels remains an active area of scientific debate

### Sample conflictingqa_a25014a5c5b5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d5
- **Claim**: Not all plants can survive without any light: most plants require light for photosynthesis to produce energy and food extended darkness typically kills them

### Sample conflictingqa_a25014a5c5b5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: However, some plants possess adaptations that allow them to endure low-light conditions or artificial light certain species like snake plants or ZZ plants are renowned for thriving in windowless environments when given adequate indirect or artificial illumination

### Sample conflictingqa_a2f06d54b240

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: yes

### Sample conflictingqa_a3264277980a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The question of whether Orson Welles's 1938 War of the Worlds radio broadcast caused mass panic is genuinely contested among scholars and historians

### Sample conflictingqa_a3264277980a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: While the broadcast is legendary for allegedly triggering widespread hysteria, including suicides and hospitalizations, empirical evidence does not support these claims: no verified suicides were reported by newspapers Princeton researchers found that rumors of people treated for shock at St. Michael's Hospital were inaccurate

### Sample conflictingqa_a3264277980a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Surveys conducted immediately after the broadcast suggested that fewer than 2% of the national audience was tuned in most who did listen recognized it as fiction from the opening announcements

### Sample conflictingqa_a3264277980a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: However, some sources note that a meaningful subset of listeners—perhaps 10–15%—did experience genuine fear the broader narrative of national panic was amplified by newspapers seeking to discredit the emerging medium of radio

### Sample conflictingqa_a3980a2921cf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Yes — hair oil is beneficial for all hair types

### Sample conflictingqa_a507c5b61631

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d3
- **Claim**: Volcanic activity is among the leading proposed triggers for the Paleocene-Eocene Thermal Maximum (PETM), but the evidence is not conclusive

### Sample conflictingqa_a7ff288bc615

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Yes

### Sample conflictingqa_a864ff85e648

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Yes

### Sample conflictingqa_a8ec1c0d2b92

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The retrieved evidence presents conflicting opinions

### Sample conflictingqa_a8ec1c0d2b92

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Some sources argue that green tea does not directly cause kidney stones and may even reduce the risk due to its antioxidant content and diuretic effects , while others note that green tea contains oxalates and recommend moderation for those at high risk of calcium oxalate stones

### Sample conflictingqa_a8ec1c0d2b92

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: A 2019 study cited by one source found that daily green tea consumption did not increase the risk of kidney stone formation , but a urologist quoted by the New York Times warned that iced tea (which includes green tea) is "one of the worst things to drink" for those prone to the most common type of kidney stones a 2014 study in the Journal of Urology reported that green tea increased oxalate excretion in rats

### Sample conflictingqa_a994724a28e7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: False

### Sample conflictingqa_a9bed39d234d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: The negative-calorie food concept is controversial, with d1 claiming certain foods can burn more calories than they provide, while d5 states there is no evidence supporting the idea and even low-calorie foods contain more calories than it takes to break them down

### Sample conflictingqa_b2524e4883ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The retrieved evidence presents complementary aspects of the answer

### Sample conflictingqa_b2524e4883ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4
- **Claim**: Most sources agree that small meteor shower debris poses little direct threat to life on Earth, burning up harmlessly in the atmosphere and even providing beneficial atmospheric sodium layers used in astronomy

### Sample conflictingqa_b2524e4883ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: Larger chunks—such as those potentially present in the Taurid stream—could theoretically cause significant damage upon impact that even small particles can pose serious risks to spacecraft and satellites in orbit

### Sample conflictingqa_b2524e4883ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: In summary, meteor showers are generally considered a low threat to Earth's surface populations but remain a legitimate concern for space assets and warrant ongoing monitoring

### Sample conflictingqa_b323dd4b5820

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Current CO2 levels are not unprecedented in Earth's history if we look at concentrations alone, as levels were similarly high during the Pliocene epoch 3.3–4.3 million years ago and may have reached 1,000 kPa in Earth's earliest days; however, the rate of increase since the Industrial Revolution—100–200 times faster than any natural increase in the historical record—is itself unprecedented if current trends continue, CO2 could soon reach 800 ppm, a level not seen in 50 million years

### Sample conflictingqa_b7fd50f9f980

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d3
- **Claim**: Yes, 'alright' is generally considered an acceptable alternative spelling of 'all right,' used widely in casual and informal contexts, while 'all right' is the more traditional and preferred form in formal, academic professional writing

### Sample conflictingqa_b9854bd5a19e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5
- **Claim**: Yes, the human brain has decreased in size over time

### Sample conflictingqa_b9854bd5a19e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: Fossil evidence indicates that modern human skulls are on average 12.7% smaller than those of Homo sapiens from the last ice age, with this trend beginning around 100,000 years ago and continuing through the present day

### Sample conflictingqa_b9854bd5a19e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: Research published in PubMed further confirms that since the Late Pleistocene (approximately 30,000 years ago), human brain size decreased by approximately 10%, a reduction paralleled by a corresponding decrease in body size

### Sample conflictingqa_b9854bd5a19e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: These findings are corroborated by skeletal evidence from every inhabited continent, which shows that human brains have become smaller in the past 10,000 to 20,000 years , a trend attributed to factors including reduced physical activity, changes in diet the offloading of cognitive tasks to external technologies

### Sample conflictingqa_bac0f4d62f96

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The retrieved evidence is conflicting

### Sample conflictingqa_bac0f4d62f96

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: Some sources argue that meteorites can come from comets, while others argue that few if any large meteorites come from comets

### Sample conflictingqa_bd2e652cd64d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Yes, electric toothbrushes are generally considered better for your teeth than manual ones

### Sample conflictingqa_bd2e652cd64d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: Research cited by Cleveland Clinic shows that electric toothbrushes remove plaque more effectively than manual brushes, which can help prevent cavities and gum disease

### Sample conflictingqa_bd2e652cd64d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: An 11-year study of over 2,000 patients found that those using electric toothbrushes had 22% less gum recession and 18% less tooth decay compared to manual users a 2019 study of nearly 3,000 people confirmed that sonic toothbrushes reduced signs of periodontal disease and tooth loss more than manual brushing did

### Sample conflictingqa_bd2e652cd64d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Additional benefits include built-in timers that encourage adequate brushing duration, pressure sensors to prevent gum damage improved outcomes for orthodontic patients and those with limited hand mobility

### Sample conflictingqa_bdee100fa8e5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The question of whether Orson Welles's 1938 War of the Worlds broadcast caused a real-life panic is genuinely contested among scholars and historians

### Sample conflictingqa_bdee100fa8e5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d3
- **Claim**: While the broadcast is iconic and widely remembered as triggering mass hysteria, multiple studies and sources argue that the panic was largely exaggerated—fueled by newspapers seeking to discredit radio as a competitor to print, rather than reflecting the true listener reaction

### Sample conflictingqa_bdee100fa8e5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d3
- **Claim**: Research based on contemporary surveys and letter collections suggests that few listeners actually tuned in to the broadcast even fewer took it as genuine fiction , with one scholar noting that the program "didn't actually cause mass panic at all" ; d4 similarly states that "historians have argued the supposed panic was always exaggerated." On the other side, some sources maintain that the panic was real if limited—that thousands fled in genuine terror —and at least some witnesses reported immediate, visceral reactions to the simulated invasion , leaving the historical consensus divided over the breadth and nature of the public response

### Sample conflictingqa_be17259fe5c0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: Not according to a 2020 genetic study, which concluded that penguins first evolved in Australia and New Zealand rather than in Antarctica

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The environmental comparison between paper and plastic straws depends on the metric considered and the context of use

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d3
- **Claim**: On the one hand, paper straws are biodegradable and avoid the persistent pollution associated with plastic, making them a better choice when disposed of properly or composted ; additionally, some studies suggest that plastic straws may cause 44 times fewer greenhouse gas emissions during manufacture than paper alternatives

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: On the other hand, a UK government assessment found that paper straws actually emit more greenhouse gases when they decompose in landfills compared to plastic straws some sources argue that the energy required to produce paper straws and their tendency to degrade quickly in use present their own environmental drawbacks

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Overall, the evidence is mixed experts generally agree that refusing straws altogether is the most environmentally friendly option if possible

### Sample conflictingqa_bf6e9339f2c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3
- **Claim**: Yes, nutritional yeast is a complete protein source for vegans

### Sample conflictingqa_bf6e9339f2c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3
- **Claim**: Yeast protein contains all essential amino acids in the required quantities recommended by the FAO, making it a complete protein

### Sample conflictingqa_bf6e9339f2c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Nutritional yeast is an 'excellent source of highly digestible complete protein' and that the protein content of S. cerevisiae offers a valuable alternative to traditional animal-based protein sources

### Sample conflictingqa_bfbbc2c7a1af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: Yes

### Sample conflictingqa_c1119b945459

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d3
- **Claim**: The retrieved evidence indicates that Hindu beliefs are complex and multifaceted

### Sample conflictingqa_c1119b945459

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d3
- **Claim**: Ultimately, the diversity of Hindu beliefs means that not all Hindus conceive of divinity in the same way

### Sample conflictingqa_c34991d9897e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: Yes, copyright can protect logos — but only if the logo contains artistic or creative elements; plain text or generic designs typically do not qualify

### Sample conflictingqa_c34991d9897e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: In the UK, a logo will almost always qualify as an "artistic work" and receive automatic copyright protection from the moment it is created, though this protection is limited to direct copying and does not prevent independent creation of similar designs

### Sample conflictingqa_c34991d9897e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: In Australia and other jurisdictions, the same rule applies: copyright protects the design elements of a logo, while trademark law handles broader brand protection

### Sample conflictingqa_c3e66630928c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d3
- **Claim**: Research and user opinions are divided on whether coffee grounds deter slugs and snails; laboratory tests found that caffeine solutions above 0.1% concentration reliably deter snails, but dry grounds have a weaker, less consistent effect

### Sample conflictingqa_c418fecfc1e2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: Not really — plants can survive in low light or without direct sunlight, but no plant can grow without any light at all

### Sample conflictingqa_c574530da7a3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: Religious and theological views differ; science offers no settled answer

### Sample conflictingqa_cc71318e5853

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The retrieved evidence presents competing views

### Sample conflictingqa_cc71318e5853

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4, d3
- **Claim**: Some sources argue that death remains a taboo topic in modern society, especially in American culture, where open discussion is seen as uncomfortable and avoided before the pandemic it was considered one of the most taboo subjects ; others note that the pandemic has brought death into broader public consciousness

### Sample conflictingqa_cc71318e5853

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: On the other hand, academic analysis argues that death is not truly taboo but rather occupies a complex, shifting social space influenced by factors like bereavement organizations and changing attitudes over time that modernity has made many once-sensitive topics discussible while death has remained less so

### Sample conflictingqa_cd661c2c20b6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Gwen Stacy's death is frequently cited as a transformative moment in Spider-Man's history and the comics industry, but its precise role in ending the Silver Age is debated

### Sample conflictingqa_d295f9ea94b3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: No, Botox is not a type of plastic surgery

### Sample conflictingqa_d295f9ea94b3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: Plastic surgery typically involves surgical interventions that reshape or reconstruct different parts of the body, whereas Botox is a minimally invasive non-surgical cosmetic procedure that uses botulinum toxin injections to relax facial muscles and reduce the appearance of wrinkles

### Sample conflictingqa_d295f9ea94b3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1, d5
- **Claim**: While Botox is frequently performed in plastic surgery offices and some sources, refer to it as a 'cosmetic surgery' procedure, the consensus across high-quality sources such as d2 and d4 is that it remains distinct from traditional plastic surgery due to its lack of incisions, shorter recovery time temporary results

### Sample conflictingqa_d9a36fe4c135

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Religious authorities and scholars hold differing views on the Bible's infallibility; no single scriptural or universal consensus exists among all traditions

### Sample conflictingqa_dc6f972e8447

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: The retrieved evidence indicates that cryptocurrency markets are vulnerable to manipulation through a variety of mechanisms — including bots, arbitrage exploitation, leverage amplification, sell walls, pump-and-dump schemes wash trading — suggesting that manipulation is both possible and ongoing

### Sample conflictingqa_dd426f7706e8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: In folklore and pop culture, the full moon is frequently associated with werewolf transformations, but it does not typically create werewolves

### Sample conflictingqa_dd426f7706e8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: Rather, it serves as a trigger for shape-shifting in already-cursed or already-bitten individuals

### Sample conflictingqa_dd426f7706e8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: Some traditions and modern interpretations introduce the idea that a full moon bite causes lycanthropy, blurring the line between trigger and creator

### Sample conflictingqa_e2e2361dc28b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: Yes

### Sample conflictingqa_e60bdcf75ba7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d3
- **Claim**: Yes, multiple sources report that organic yields are generally lower than conventional yields; however, the magnitude varies considerably by crop type and farming conditions

### Sample conflictingqa_e93e708d49a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Over their lifetime, typical rooftop solar panels in the United States produce enough energy to more than compensate for the energy required to manufacture, install recycle them — and in some cases, they achieve energy payback within just a few years

### Sample conflictingqa_e93e708d49a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: However, the degree of net energy benefit varies significantly depending on geographic location, grid access whether a home battery is installed — for instance, Stanford researchers found that the energy return on investment drops by about 21% when a battery is added to a solar system , while Australian data shows that even in less-sunny regions like Melbourne, a 1-kW solar array still generates an average of 3.6 kWh per day over a year

### Sample conflictingqa_ece626a6cba9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Yes

### Sample conflictingqa_f1932b75ace7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d3
- **Claim**: There is significant scientific debate and conflicting evidence regarding whether bee stings treat arthritis

### Sample conflictingqa_f1932b75ace7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: Proponents, including early 20th-century physicians and some current users, report that bee venom alleviates arthritis pain and inflammation, with one study noting that bee venom contains multiple anti-inflammatory components

### Sample conflictingqa_f1932b75ace7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3
- **Claim**: However, peer-reviewed research has documented cases of 'beekeeper's arthropathy'—a non-infectious arthritis affecting small hand joints in beekeepers following bee stings a review by Healthline emphasizes that while some patients anecdotally report dramatic improvements, more rigorous research is needed to confirm safety and efficacy

### Sample conflictingqa_f1932b75ace7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: Overall, the scientific community has not established bee stings as a proven medical treatment for arthritis experts caution that the practice carries serious risks of allergic reaction

### Sample conflictingqa_f22b389be1d6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Heads differ on whether barefoot running is healthier; the predominant clinical view is that modern running shoes reduce impact forces and related injuries

### Sample conflictingqa_f39c966c2ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The retrieved evidence presents competing views

### Sample conflictingqa_f39c966c2ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: The dominant folkloric tradition, reported by the RSC and other sources, holds that Macbeth was cursed from its first performance: the actor playing Lady Macbeth reportedly died suddenly, forcing Shakespeare to replace him a real dagger allegedly used in the production led to the death of the actor playing King Duncan

### Sample conflictingqa_f39c966c2ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: However, a statistical analysis cited by Scribd challenges this narrative, suggesting that Macbeth does not experience disproportionately more accidents or mishaps than other Shakespearean plays, implying that the curse is a persistent legend rather than a demonstrable statistical reality

### Sample conflictingqa_f3b163170581

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3
- **Claim**: Yes — but not from modern apes; rather, humans evolved from ancient ape ancestors that lived millions of years ago, diverging from the same family that gave rise to modern apes like chimpanzees and gorillas

### Sample conflictingqa_f3ba8599e370

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The retrieved evidence presents competing views

### Sample conflictingqa_f3ba8599e370

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Some sources argue that yoga is not a religion because it emphasizes direct experience rather than faith, does not require belief in a higher power is compatible with many different religious beliefs

### Sample conflictingqa_f3ba8599e370

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Other sources argue that yoga's essence is identical to that of religion—both aiming to join the individual to divinity—and that yoga contains significant religious elements

### Sample conflictingqa_f3ba8599e370

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: Ultimately, whether yoga is a religion depends on how one defines religion and what aspects of yoga they emphasize

### Sample conflictingqa_f43b2c51deea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Yes

### Sample conflictingqa_f4693bea2c31

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: Emoji are widely used as written signs, but whether they constitute a form of written language is subject to ongoing debate

### Sample conflictingqa_f4693bea2c31

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: Most linguists currently view emoji as a complex system of pictographs that augment and add nuance to text—comparable to intonation and gesture in spoken language—rather than as a distinct, freestanding language; early written languages like cuneiform and hieroglyphics were logographic emoji share some features with these scripts

### Sample conflictingqa_f4693bea2c31

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: However, some researchers push back on this view, noting that emoji function as a rich form of paralinguistic communication—softening written statements and conveying emotive nuances—that may be developing into something more linguistically significant over time, particularly as users combine emojis in inventive and context-dependent ways

### Sample conflictingqa_f4811561af0c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: The retrieved evidence indicates that the Dutch were among the earliest European explorers of Australia, with Willem Janszoon's 1606 voyage to Cape York Peninsula being the first recorded European landing on the continent

### Sample conflictingqa_f4811561af0c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the evidence is incomplete regarding the overall question of whether Australia was 'discovered' by the Dutch, as it does not address prior indigenous occupation or subsequent European claims

### Sample conflictingqa_f777a43ba278

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The link between yerba mate and cancer is nuanced and subject to ongoing scientific debate

### Sample conflictingqa_f777a43ba278

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: The NIH cites epidemiological studies showing increased risks of esophageal, head and neck bladder cancers associated with yerba mate consumption, particularly when drunk at very high temperatures

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Some sources suggest that the elevated cancer risk may stem largely from the extreme heat of the beverage rather than the yerba mate itself, as animal studies have also identified anti-cancer properties in the herb that led the National Cancer Institute to launch a clinical trial

### Sample conflictingqa_f777a43ba278

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: Overall, while population studies indicate higher cancer incidence among heavy, long-term consumers — especially when combined with other risk factors like smoking or alcohol — the evidence does not conclusively establish yerba mate as a direct cause of cancer the picture remains largely unresolved

### Sample conflictingqa_f7fec8c0688b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: The official explanation from the U.S. Air Force attributes the Phoenix Lights to military flares dropped during a training exercise, specifically illumination flares released by A-10 aircraft from Luke Air Force Base

### Sample conflictingqa_f7fec8c0688b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d3
- **Claim**: However, this explanation is widely contested: many witnesses reported seeing a silent, boomerang-shaped craft with five lights that blocked out stars, characteristics that skeptics argue flares cannot replicate , while former Arizona governor Fife Symington stated that what he saw was 'not man-made' and 'certainly not high-altitude flares because flares don't fly in formation'

### Sample conflictingqa_f7fec8c0688b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d3
- **Claim**: The result is a genuine disagreement between the official military account and the recollections of many witnesses and observers, making the true cause of the Phoenix Lights a subject of持续争议。

### Sample conflictingqa_f8da23d84ecc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: Once considered the same dinosaur, Apatosaurus and Brontosaurus were reclassified as distinct genera in a 2015 study

### Sample conflictingqa_f970957c5e52

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The retrieved evidence presents conflicting opinions

### Sample conflictingqa_f970957c5e52

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Some sources argue that the Oxford comma is optional and its omission is not a grammatical error, while others argue that it is necessary for clarity, especially in complex lists

### Sample conflictingqa_f97fef94decc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: There is no definitive scientific consensus that VR headsets are harmful to eyesight; most experts note that modern headsets do not cause permanent damage

### Sample conflictingqa_f97fef94decc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: However, conflicting opinions exist: one user reported vision problems that a doctor attributed to prolonged use some sources argue that low-resolution displays or prolonged exposure can lead to eye fatigue and convergence issues

### Sample conflictingqa_f97fef94decc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: Moderation is widely recommended, as excessive use has been linked to temporary symptoms like dryness and blurred vision, similar to other digital screen use

### Sample conflictingqa_fa98c00bd697

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: Black holes can be detected and studied with telescopes, but seeing them directly is impossible because their gravity is so strong that not even light can escape

### Sample conflictingqa_fa98c00bd697

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: High-credibility sources confirm that black holes are not actually visible, but scientists can observe their effects — such as warped light, accretion disks jet streams — using a variety of telescopes

### Sample conflictingqa_fa98c00bd697

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: Some sources claim that simple telescopes can capture indirect evidence of nearby black holes, while others note that advanced instruments like the Event Horizon Telescope are required for the best possible images

### Sample conflictingqa_fad0d30903d2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Yes — Woodstock 1969 is widely regarded as a defining symbol of peace, love unity, with hundreds of thousands of attendees demonstrating a spirit of harmony despite chaotic conditions

### Sample conflictingqa_fbedb688b1d9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: Religious scholars and commentators hold differing views on whether Mormons are Christians, with the primary conflict centering on the definition of 'Christian.' Some argue that because Mormons believe in Jesus Christ and seek to follow Him, they should be considered Christians by definition the official LDS Church website states that members 'unequivocally affirm themselves to be Christians' ; others counter that Mormon theology repudiates many core Christian doctrines—including the nature of God, salvation scriptural authority—making the label inappropriate except under a purely self-identified sense

### Sample conflictingqa_fbedb688b1d9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: This debate is further reflected in the broader religious community, where some Christian denominations accept Mormons as fellow Christians while others do not

### Sample conflictingqa_fcdb9e210683

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: Yes — viral genomes are placed into phylogenetic trees, though virions themselves are inert and not considered alive

### Sample freshqa_0293a11bd364

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Hindi

### Sample freshqa_02b3ba89ebd0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5
- **Claim**: Kevin McCarthy was not elected Speaker of the House on the ninth ballot in January 2023; he won 200 votes on that ballot, falling 18 votes short of the 218 needed for a majority

### Sample freshqa_02b3ba89ebd0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: The query contains a factual error: McCarthy did not secure the speakership until the 15th ballot, after all six remaining Republican detractors voted 'present' to lower the threshold the House subsequently elected him as the Republican Speaker

### Sample freshqa_0436c0b3a9d7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3
- **Claim**: Aryna Sabalenka and Amanda Anisimova

### Sample freshqa_047057d22309

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: King Charles III has not stripped Prince Harry's title as the Duke of Sussex

### Sample freshqa_047057d22309

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Prince Harry and Meghan Markle agreed to stop using the title 'His Royal Highness' in early 2020 as part of their departure from the Royal Family Buckingham Palace subsequently updated the official website to reflect this change

### Sample freshqa_047057d22309

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Harry retains the title of Duke of Sussex, with one legal expert noting that the title is hereditary and cannot be revoked without a statute another report stating that King Charles III is reportedly under pressure from Prince William to strip Harry and Meghan of their titles, suggesting the action has not yet occurred

### Sample freshqa_049cc3f14d5e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: St. Petersburg State University

### Sample freshqa_1009f5c49e12

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Paris

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d3
- **Claim**: August 16, 1977

### Sample freshqa_122352ad92e3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Apr 1

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1ef881d26e2e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: Maryam Mirzakhani

### Sample freshqa_1fe3ba1b1cba

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Lando Norris

### Sample freshqa_2130eea851fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: 1,035,072+

### Sample freshqa_25b286cb2af1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: Venus has no moons

### Sample freshqa_263eca8e024e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d5
- **Claim**: Dangal

### Sample freshqa_2877cf4bd00f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3
- **Claim**: 78–79

### Sample freshqa_28e155139ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: The latest stable version of Android is Android 16, released in June 2025

### Sample freshqa_2b9ba7e192e2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Dina Boluarte (Dina Margarita Boluarte Zanatta), who became Peru's first female president on December 7, 2022, after being sworn in during a turbulent political crisis that followed her predecessor's impeachment

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
- **Claim**: This supersedes earlier records showing Samara Joy as the 2025 winner for the same category

### Sample freshqa_3227ea6c6056

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The latest major version of .NET is 10.0, released in May 2026

### Sample freshqa_3227ea6c6056

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3
- **Claim**: This supersedes earlier sources that identified .NET 7.0 or .NET 6.0 as the newest versions, as the .NET versioning schema has since evolved past those releases

### Sample freshqa_354e7097602d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The first atomic bomb test in the United States took place at the Trinity Site, located on the barren plains of the Jornada del Muerto desert within the current-day White Sands Missile Range in New Mexico

### Sample freshqa_354e7097602d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: Specifically, the test occurred on July 16, 1945, at a site situated approximately 210 miles south of Los Alamos, New Mexico, on the then-U.S. Army's Alamogordo Bombing and Gunnery Range

### Sample freshqa_354e7097602d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d3
- **Claim**: This historic event, code-named "Trinity" by physicist J. Robert Oppenheimer, marked the successful detonation of a plutonium implosion device and served as a pivotal milestone in the Manhattan Project

### Sample freshqa_35bf342002aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: 7

### Sample freshqa_3847b5cb9b42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d3
- **Claim**: Russia's invasion of Ukraine (2022–present)

### Sample freshqa_39dcd7b38c39

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d3
- **Claim**: Maya Angelou was the first African American woman to appear on a U.S. quarter

### Sample freshqa_3ad16f379533

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Russia has been invading Ukraine, beginning with its full-scale military offensive on February 24, 2022

### Sample freshqa_3ad16f379533

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: This act of aggression has been consistently characterized by the Australian government, the BBC academic sources as an unprovoked violation of international law

### Sample freshqa_3ad16f379533

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: NPR and the University of Florida further confirm that the conflict has persisted for over four years, with Russia continuing its occupation of Ukrainian territories

### Sample freshqa_3cd0b514193b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d3
- **Claim**: The minimum hourly wage in Tokyo is ¥1,226 per hour

### Sample freshqa_3dc3cf00bce6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d3
- **Claim**: Pembroke Welsh Corgi

### Sample freshqa_42796b35e143

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: 3

### Sample freshqa_4590bdd9e269

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_4a98eba95e97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Joe Biden did not visit Russia as president of the United States

### Sample freshqa_4a98eba95e97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: His only meeting with Russian President Vladimir Putin took place on June 16, 2021, at Villa La Grange in Geneva, Switzerland, not within Russia's territory

### Sample freshqa_4a98eba95e97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5
- **Claim**: This is confirmed across multiple authoritative sources, which note that Biden's foreign travel to Russia was ruled out due to the ongoing war in Ukraine, making the Geneva summit the sole bilateral encounter between the two leaders during Biden's presidency

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: 0

### Sample freshqa_4e635a2542a8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: Red Garland

### Sample freshqa_50f8f03fd30f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: Two months old

### Sample freshqa_5574b1447bdb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Wuhan

### Sample freshqa_5d6e5db69928

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: Greenland

### Sample freshqa_5eb89aae15f3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: KGF: Chapter 1

### Sample freshqa_5ecee1c55713

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Portugal

### Sample freshqa_64c12116affc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5
- **Claim**: Donald J. Trump is the President of the United States, having served two non-consecutive terms: first from January 20, 2017 to January 20, 2021 currently from January 20, 2025 to the present

### Sample freshqa_64c12116affc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5
- **Claim**: He is the 47th President, succeeding Joseph R. Biden Jr. who served from January 20, 2021 to January 20, 2025

### Sample freshqa_6927f007d7cc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5
- **Claim**: Alexia Jayy

### Sample freshqa_6a45fadeb16b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Based on recent data, Costco's Executive membership costs $130 annually , providing a 2% cashback reward on eligible purchases

### Sample freshqa_6a45fadeb16b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Older sources citing $120 or $65 incremental costs are superseded by current pricing, which stands at $130 per year for the Executive level

### Sample freshqa_6f42c128eb6c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Harry Maguire has never won the Ballon d'Or

### Sample freshqa_7bc7bb2dde20

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: The latest Academy Award for Best Picture was won by **One Battle After Another** at the 98th Academy Awards in 2026

### Sample freshqa_7bc7bb2dde20

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: The film, directed by Paul Thomas Anderson, earned six Oscars including Best Director and Best Adapted Screenplay, making it the most recent recipient of the Academy's top honor

### Sample freshqa_7bc92b47dc43

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: 2

### Sample freshqa_7dce5d575302

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: Kaka

### Sample freshqa_7e63fcff2dea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: The first animal to land on the moon was a tortoise, specifically one of two Russian tortoises that traveled on the Zond 5 mission in September 1968

### Sample freshqa_7e63fcff2dea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: This mission marked the first time any living beings circled the Moon, though it did not technically land — the craft made a water landing upon return to Earth

### Sample freshqa_7e63fcff2dea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d4
- **Claim**: Earlier documents had incorrectly identified the first animal in orbit or speculated about lunar landers , but the most accurate evidence confirms tortoises as the first animals to travel beyond Earth's orbit and return safely

### Sample freshqa_7f1c3aae61a5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Luke Littler

### Sample freshqa_80642f637dc7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: Lionel Messi

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: Bayonne, New Jersey

### Sample freshqa_8eca5bd62ae0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d3
- **Claim**: Beijing became the first city in Olympic history to host both the Summer and Winter Games

### Sample freshqa_8eca5bd62ae0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d3
- **Claim**: It hosted the 2008 Summer Olympics and was then selected to host the 2022 Winter Olympics, making it the only city at the time to have held both events

### Sample freshqa_8eca5bd62ae0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: This distinction is confirmed by multiple sources, including the Wikipedia list of Olympic host cities, which notes that Beijing achieved this milestone in 2022

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
- **Claim**: The Britannica record shows the Raptors finished 25–57 in the 2023–24 season, missing the playoffs entirely

### Sample freshqa_a47283064972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This losing record is further contextualized by earlier evidence showing the team's performance declined steadily from 2019–20 (53–19) to 2022–23 (41–41), with the 2023–24 season representing the low point

### Sample freshqa_a47283064972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: While the Raptors won the 2019 NBA Championship and had strong seasons in 2016–17 and 2017–18 , those results are no longer current the most recent data confirms a losing record in the 2023–24 campaign

### Sample freshqa_a50d0f1f3cdf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: 9 September 2022

### Sample freshqa_a5492f36ca23

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: January 10, 2016

### Sample freshqa_a8b908895e11

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d3
- **Claim**: San José

### Sample freshqa_ab11b5dce00e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: USA

### Sample freshqa_b0ffe73c1789

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Colleen Hoover has written and published 26 books, according to Forbes

### Sample freshqa_b0ffe73c1789

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This figure is consistent with her author page on Goodreads, which lists approximately 20-25 titles depending on edition counts Britannica, which confirms her prolific output without specifying the exact total

### Sample freshqa_b0ffe73c1789

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Some sources citing older data report a lower count of 20 books , but these are superseded by more recent tallies showing she has continued publishing new works

### Sample freshqa_b3264b37f54b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Yes, Arsenal is at the top of the Premier League standings

### Sample freshqa_b99c189f2222

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_c3f10dc1632d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: Jiangsu Province

### Sample freshqa_c479e83e408f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: 15

### Sample freshqa_c4976f8629cb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The heaviest reptile in the world is the saltwater crocodile (Crocodylus porosus), according to a scientific consensus cited by the Crocodile Specialist Group

### Sample freshqa_c7315f8b3029

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: GPT-5.5 was released on May 5, 2026

### Sample freshqa_c7ac9d61059a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: $51,380

### Sample freshqa_cbfca321cce4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4, d3
- **Claim**: Vincent van Gogh

### Sample freshqa_cf331ed7d09f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3
- **Claim**: macOS 26 Tahoe

### Sample freshqa_d4d59d75b206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: 2015, 2016 2018

### Sample freshqa_d510972df578

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: By nominal production budget, Star Wars: The Rise of Skywalker ($490 million) and Avatar ($425 million) lead, though adjusted-for-inflation rankings place The Force Awakens at $552 million

### Sample freshqa_dd85dcbc2262

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: Aryna Sabalenka is the number 1 ranked female tennis player in the world as of May 4, 2026

### Sample freshqa_dd85dcbc2262

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: This is confirmed by the WTA official rankings and multiple supporting sources that track her career trajectory from her initial ascent in 2023 through continuous updates

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: 12

### Sample freshqa_ddd643091cbe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3
- **Claim**: A permanent cure for cancer has not been developed; the concept of a universal permanent cure remains elusive because cancer encompasses numerous diseases with varying biology

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_e502143179d6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: October 2022

### Sample freshqa_edf4ae4f32e5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d5
- **Claim**: 1941

### Sample freshqa_ef3ad40c6540

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: Lakers

### Sample freshqa_f11e8d758fc6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Slugs do not have lungs in the same sense as mammals; rather, they possess a single lung-like structure called a mantle cavity, which communicates with the outside through a breathing pore called the pneumostome

### Sample freshqa_f11e8d758fc6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This cavity is functional for gas exchange but is not a discrete organ — it is a hollow space within the mantle lined with blood-vessel-rich tissue

### Sample freshqa_f11e8d758fc6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Therefore, the concept of counting 'lungs' as distinct organs does not directly apply, though some sources describe the arrangement as analogous to a single lung

### Sample freshqa_f5d8e53958c6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Hawaii

### Sample freshqa_f5eef5c415e8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: 28

### Sample freshqa_f6ac249bdf53

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4, d3
- **Claim**: Ta-Nehisi Coates

### Sample freshqa_f6cc6071caa5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: A total of 893 Nazca figurative geoglyphs have been discovered, according to research published in Proceedings of the National Academy of Sciences

### Sample freshqa_f6cc6071caa5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: This figure grew rapidly through successive AI-assisted survey campaigns: in November 2024, approximately 730–1,000 geoglyphs were known depending on methodology earlier counts cited only 358 or 430 before the advent of modern machine learning tools

### Sample freshqa_f6cc6071caa5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The consistent application of artificial intelligence has nearly tripled the confirmed total since the 1940s, underscoring ongoing scholarly revision of the ancient site's full extent

### Sample freshqa_fc07d4525dbf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The youngest age eligible for COVID-19 vaccination in the United States varies by vaccine type

### Sample freshqa_fc07d4525dbf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Moderna's Spikevax vaccine is authorized for children as young as 6 months, while the Pfizer vaccine is approved for those 5 years and older Novavax is cleared for ages 12 and up

### Sample freshqa_fc07d4525dbf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: Some sources, citing broader federal communication, state that the vaccine is available for all children and adults ages 6 months and older California public health guidance confirms eligibility for infants as young as 6 months ; however, other authoritative sources note that Pfizer's updated version is no longer available to children under 5 without underlying health conditions, creating a nuanced picture depending on which vaccine is accessed

### Sample freshqa_fd00b29e848c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: February 18–March 19

### Sample freshqa_ff04793833cc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d3
- **Claim**: Andrew Johnson was not elected as President of the United States in any year; he became President in 1865 after Abraham Lincoln's assassination

### Sample healthcontradict_0060

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: No, a tepid sponge bath does not reduce fever in children and is not recommended by NHS guidelines

### Sample healthcontradict_0145

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: Yes, yoga may improve asthma management, but the evidence is mixed

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
- **Supporting Docs Found**: d7, d2, d5, d6, d10
- **Claim**: Boston College

### Sample hotpotqa_0064

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d10
- **Claim**: Victor Mature

### Sample hotpotqa_0070

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: 15

### Sample hotpotqa_0071

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d10, d3
- **Claim**: Keyshia Cole

### Sample hotpotqa_0073

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0079

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
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
- **Supporting Docs Found**: d1, d9, d5
- **Claim**: "My Own Worst Enemy"

### Sample hotpotqa_0134

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d10
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
- **Claim**: More than 1,600 German scientists, engineers technicians were recruited in post-Nazi Germany as part of Operation Paperclip, though the subset who became involved in the U.S. space program is not explicitly quantified

### Sample hotpotqa_0196

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d3
- **Claim**: St James Street appears as a segment of Whitecross Street on the 1610 map of Monmouth by cartographer John Speed , who is widely recognized as the best-known English mapmaker of the Stuart period

### Sample misinformation_0023

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: No, drinking bleach does not cure infections; it is toxic and can cause severe injury or death

### Sample qacc_0013

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d7, d1, d2, d5, d6
- **Claim**: The Fourteenth Amendment is the primary vehicle through which the Bill of Rights is applied to the states, via the incorporation doctrine established by Supreme Court rulings

### Sample qacc_0013

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d6, d4
- **Claim**: The U.S. Courts further clarify that under this doctrine, many Bill of Rights guarantees have been incorporated into the Fourteenth Amendment, extending them to the states , while the Constitutional Accountability Center notes that the Fifth Amendment's protections specifically apply only to the federal government, making incorporation against the states a distinct question

### Sample qacc_0013

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7
- **Claim**: It is worth noting that some provisions, such as the right to bear arms, were historically understood as applicable only to the federal government , though the Supreme Court has since ruled that the right to keep and bear arms does apply to the states through the Fourteenth Amendment's due-process clause

### Sample qacc_0023

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d8, d5, d3
- **Claim**: Pentheus was torn apart by the maenads at the end of The Bacchae

### Sample qacc_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Justin Timberlake

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d7, d2, d8, d5, d6
- **Claim**: The most reliable sources report 506 instances of the f-word in The Wolf of Wall Street, though some sources incorrectly cite 569

### Sample qacc_0156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d6, d4
- **Claim**: Sheldon Collins

### Sample qacc_08cf866bcb9b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0a580da7f2cd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: 1987

### Sample qacc_0ac549afb037

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: The last name Hansen is of Scandinavian and Northern European origin, derived from the personal name Hans and used as a patronymic suffix in Danish, Norwegian, Dutch, Flemish North German languages

### Sample qacc_0ac549afb037

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: It is the most common surname in Norway and is widely distributed across Northern Europe, with the largest concentration in Denmark

### Sample qacc_0ac549afb037

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d5
- **Claim**: Genealogical data further confirms that the surname was historically formed by adding -son or -sen to a father's name, making it hereditary in nature modern ancestry analysis corroborates these roots, showing that people with the surname Hansen carry the highest concentrations of British & Irish, French & German Scandinavian heritage

### Sample qacc_0b3b6452e224

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: The Statue of Liberty was designed by French sculptor Frédéric Auguste Bartholdi, who modeled the statue's face after his own mother and drew inspiration from the Roman goddess of liberty, Libertas

### Sample qacc_0b75ed799d46

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The Screen Actors Guild Awards are held at the Shrine Auditorium & Expo Hall in Los Angeles, California

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: Following the Allied victory in North Africa, the document indicates the next major campaign was in Italy, implying the Allies moved on to the Italian Campaign after North Africa

### Sample qacc_0d85f1089c4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The brand ambassador of the 'Beti Bachao Beti Padhao' campaign varies by state

### Sample qacc_0d85f1089c4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: At the national level, Parineeti Chopra was the prominent brand ambassador for the Haryana-specific initiative , while Madhuri Dixit also became a brand ambassador for the campaign

### Sample qacc_0d85f1089c4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: Simultaneously, Sakshi Malik was appointed as the brand ambassador for the Haryana government's version of the initiative Bhawna Dehariya Mishra along with her daughter Siddhi Mishra were selected for the Madhya Pradesh iteration

### Sample qacc_1025b0681710

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Cassie Scerbo plays Lauren Tanner in Make It or Break It

### Sample qacc_132167e66120

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d5
- **Claim**: India won the Cricket World Cup on multiple occasions

### Sample qacc_132167e66120

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: The country's first ODI World Cup title came in 1983, when India defeated the West Indies by 43 runs at Lord's in England, led by captain Kapil Dev

### Sample qacc_132167e66120

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: In addition to that historic 1983 victory, India also claimed the T20 Cricket World Cup in 2007 (under MS Dhoni's captaincy) , 2024 2026 , making them the only team to win the T20 World Cup three times and the first to successfully defend the title

### Sample qacc_15ffab2466f7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5
- **Claim**: The retrieved evidence indicates that The Phantom of the Opera has played in Toronto at multiple venues

### Sample qacc_15ffab2466f7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The production opened at the Pantages Theatre in 1989 and ran there through 1999 later played there again from April 16, 2000 through September 26, 2000

### Sample qacc_15ffab2466f7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: In 2018, a new production toured to Toronto and played at the Princess of Wales Theatre from June 7 to June 30, 2018

### Sample qacc_160a528ae07e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: 3

### Sample qacc_17dc360eea55

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: 13

### Sample qacc_19ca08790764

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4, d3
- **Claim**: Oliver Stark

### Sample qacc_1a764b8b6cf5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d3
- **Claim**: The rule of the three rightly guided caliphs was called the Rashidun Caliphate (Arabic: الخلافة الراشدة, al-Khilafah al-Rashidah)

### Sample qacc_1a764b8b6cf5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: This term is used interchangeably with 'the Rightly Guided Caliphate,' and the first four caliphs—Abu Bakr, Umar, Uthman Ali—are referred to as the Rashidun (الراشدون, al-Rāshidūn) or Al-Khulafa-ur-Rashidun

### Sample qacc_1b95727cc286

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d3
- **Claim**: The real characters of Paid in Full are Azie Faison, Rich Porter Alpo Martinez, who were New York drug dealers in the 1980s and 1990s

### Sample qacc_1b95727cc286

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: The film is loosely based on their lives, with Ace (Wood Harris) corresponding to Azie Faison, Mitch (Mekhi Phifer) to Rich Porter Rico (Cam'ron) to Alpo Martinez

### Sample qacc_1b95727cc286

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: IMDB confirms the same trio as the inspiration, listing the film as "Based on the true story of Azie Faison Jr., Alberto Martinez and Richard Porter."

### Sample qacc_213701765f94

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: US Airways Flight 1549 landed on the Hudson River on January 15, 2009, with the airplane ditching in the river at approximately 1531

### Sample qacc_2243f17ccc38

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: 1972

### Sample qacc_252987b8054c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: Tori Spelling played Violet Anne Bickerstaff in Saved by the Bell

### Sample qacc_287da9f37864

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d3
- **Claim**: Lionel Messi first played for Barcelona's first team on November 16, 2003, when he made a substitute appearance in a friendly match against Porto during the inauguration of the Estádio do Dragão

### Sample qacc_287da9f37864

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: At just 16 years and four months old, the young Argentine talent replaced Fernando Navarro in the 75th minute under manager Frank Rijkaard

### Sample qacc_287da9f37864

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: His official competitive debut followed on October 16, 2004, in a La Liga match against Espanyol, where he came on as a substitute for Deco at the age of 17 years and three months

### Sample qacc_290c939ed6e7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: February 9, 2018

### Sample qacc_292033e4b039

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: Muhammad

### Sample qacc_2a7f7e06e365

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: The first vertebrates to exist on earth were fish — specifically jawless fishes — which are recognized as the earliest group to possess a vertebral column

### Sample qacc_2a7f7e06e365

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: These early vertebrates emerged around 500–520 million years ago during the Early Cambrian period, making them the most ancient group within the subphylum Vertebrata

### Sample qacc_2a7f7e06e365

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Among the earliest known jawless fish are species belonging to the order Myxini, commonly known as hagfish, which retain many ancestral characteristics and are considered living fossils

### Sample qacc_2cbc9a53426f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: Adrienne Barbeau played Oswald's mom (Kim Harvey) on The Drew Carey Show

### Sample qacc_2e1b5edb5e0d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The stratum lucidum

### Sample qacc_2ed872eb1114

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: Louisiana bayou country, specifically the Isle de Jean Charles and surrounding swamps and rural areas of southern Louisiana

### Sample qacc_2f6d2647a424

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: Pete Rose

### Sample qacc_34cba3c71e06

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Missi Hale

### Sample qacc_367b09e4ed80

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5
- **Claim**: Jenny Slate

### Sample qacc_37fdedfe4478

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Susan Tedeschi

### Sample qacc_3c1297608017

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: The practice of crossing fingers for good luck is generally traced to pre-Christian European traditions in which the cross was a potent magical symbol associated with binding wishes and protecting against evil; one person would cross index fingers with another to form a cross while making a wish the practice later evolved to a solo gesture

### Sample qacc_3c1297608017

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Some historians also propose that crossing fingers was adopted by early Christians as a secret recognition symbol—formed by touching thumbs and crossing index fingers to create the ichthys or fish symbol—used to identify fellow believers during periods of persecution that this Christian association eventually came to imply invoking divine protection rather than merely wishing

### Sample qacc_3d4ebfa8b6dd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: As a player, Bill Russell holds the record with 11 NBA championships; as a coach, Phil Jackson leads with 11 titles — making them tied for the all-time record when their respective roles are considered

### Sample qacc_403a59870dc2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The Rams have won the Super Bowl twice

### Sample qacc_403a59870dc2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Their first victory was on January 30, 2000, when the St. Louis Rams defeated the Tennessee Titans 23-16 in Super Bowl XXXIV at the Georgia Dome in Atlanta

### Sample qacc_403a59870dc2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Their second win came on February 13, 2022, when the Rams defeated the Cincinnati Bengals 23-20 in Super Bowl LVI at SoFi Stadium in Inglewood, California

### Sample qacc_41c44ecfd0f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4, d3
- **Claim**: The lymphatic vessels located in the small intestine are called lacteals

### Sample qacc_41c44ecfd0f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4, d3
- **Claim**: Lacteals are specialized lymphatic capillaries found in the intestinal villi, responsible for absorbing dietary fats and fat-soluble vitamins, as well as playing a role in the gut immune response

### Sample qacc_4387048ed24f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d3
- **Claim**: Anne Bancroft

### Sample qacc_44b315f6f4bb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: The retrieved evidence indicates that the Queen's crown jewels are kept in the Tower of London, specifically in a large vault there , though the jewels also previously spent time at Westminster Abbey and have been displayed there during royal ceremonies

### Sample qacc_4fb90d57c274

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: December 27, 1991

### Sample qacc_51b23ea15977

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The Soviet Union was leading the space race in April 1961, as evidenced by Yuri Gagarin's historic flight aboard Vostok 1 on April 12, 1961, which made him the first human to travel into space

### Sample qacc_51b23ea15977

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d3
- **Claim**: This achievement placed the USSR firmly ahead of the United States, whose first orbital attempt would not occur until nearly a year later

### Sample qacc_51c89636151e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The retrieval supports both Manwë and the Valar as senders of the eagles, depending on the specific context

### Sample qacc_51c89636151e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3
- **Claim**: The most direct answer is Manwë, the King of the Valar, who sent the eagles to Middle-earth , while the eagles also follow the orders of the Valar more broadly and are not bound to any mortal master

### Sample qacc_51c89636151e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: In the film sequence, it is specifically stated that 'The Eagles are coming

### Sample qacc_51c89636151e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: The Lord of the Rings Trilogy'

### Sample qacc_531aff489b71

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: Kelly Reilly

### Sample qacc_54be882d5b58

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: The Italy episodes were filmed in the town of Anguillara Sabazia, located on Lake Bracciano outside of Rome

### Sample qacc_5a9576fc5d8c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Jodie Sweetin

### Sample qacc_5eeb46719843

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Canada did not gain independence from Great Britain on a single date, as the transition was an evolutionary process rather than a momentous declaration

### Sample qacc_5eeb46719843

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The dominant view among historians is that the Statute of Westminster in 1931 marks the point at which Canada obtained full legal autonomy within the British Commonwealth — gaining the right to sign treaties and participate in international organizations as a distinct, equal member

### Sample qacc_5eeb46719843

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Some sources, noting Canada's earlier assertion of diplomatic authority in 1919, describe the period between 1919 and 1931 as the window in which sovereignty was fully acquired , while others highlight the 1867 formation of the Dominion of Canada as a precursor to full independence

### Sample qacc_5fb5c311d373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d5
- **Claim**: Lin-Manuel Miranda wrote "How Far I'll Go" for Moana

### Sample qacc_6485f021b694

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: Carroll O'Conner and Jean Stapleton performed the theme song for All in the Family

### Sample qacc_66ba2af9c3b9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Soman Chainani

### Sample qacc_67b35f41ba84

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Alice Kremelberg

### Sample qacc_6837d86d03ac

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d5
- **Claim**: Prince William, the Prince of Wales, is first in line to the British throne

### Sample qacc_6969589d80c1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5
- **Claim**: Matt Monro

### Sample qacc_6af6e8cb8f34

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: Queen Charlotte, the German-born wife of George III, is credited with introducing the first Christmas tree to Britain, setting one up at Queen's Lodge, Windsor in December 1800

### Sample qacc_6b3b372cf27d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d3
- **Claim**: Zooey Deschanel

### Sample qacc_6edf1477bd7e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: Steve McEwan

### Sample qacc_6f8df54650a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: A U.S. passport provides visa-free or visa-on-arrival access to 180 countries and territories, making it among the most powerful passports in the world for travel freedom

### Sample qacc_6f8df54650a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d3
- **Claim**: This figure is consistent across multiple sources, with one travel guide noting approximately 179 destinations benefit U.S. passport holders under similar visa-free terms the U.S. Customs and Border Protection agency further corroborating that U.S. citizens enjoy reciprocal visa-free treatment in 42 countries through the Visa Waiver Program

### Sample qacc_6f8df54650a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Older data citing lower counts, such as 160 visa-free destinations , are considered outdated and superseded by more recent travel indices and official updates

### Sample qacc_720d8b210357

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d3
- **Claim**: Multiple — The number of DNA replication origins in eukaryotes varies by organism; in humans, approximately 30,000 to 50,000 origins are activated at each cell division, while some 20 origins have been identified in complex eukaryotes more broadly

### Sample qacc_7916ffefdb99

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: John B. Watson

### Sample qacc_798b6853d20f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d3
- **Claim**: glucose

### Sample qacc_7bf02a7deb69

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Charlie Day

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d3
- **Claim**: October 1968

### Sample qacc_7f5e5a4a4391

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The letter J was gradually introduced to the alphabet across languages during the Late Medieval and Renaissance periods, with England formally distinguishing it from I around 1600

### Sample qacc_883303a2d535

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Nana in Snow Dogs is identified as an Australian Shepherd

### Sample qacc_8882ab46be5d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: 38

### Sample qacc_899648874637

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: Kate Walsh plays Dr. Addison Shepherd (also referred to as Addison Montgomery or Addison Forbes Montgomery) in Grey's Anatomy

### Sample qacc_8d7c14ed548f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Factor X

### Sample qacc_8daf80e943fa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: 5.88 trillion miles

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_8ef7b3cf5c3f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: The dominant ethnic group in southern South America, including Argentina and Uruguay, are those of European descent

### Sample qacc_8ef7b3cf5c3f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: In Uruguay specifically, about 88 percent of the population is of European descent, with the remaining roughly one-quarter being of Italian origin ; similarly, Argentines also overwhelmingly self-identify as having European ancestry

### Sample qacc_8ef7b3cf5c3f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: Regional sources further corroborate that the Southern Cone nations—Chile, Argentina, Uruguay Paraguay—share similar European ethnic patterns as a result of historical colonization

### Sample qacc_9404250d756f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: The End of the F***ing World was primarily filmed in the United Kingdom, with production taking place across multiple locations including Camberley (Surrey) and the Isle of Sheppey (Kent) for the first season, as well as in Wales for the second season

### Sample qacc_9404250d756f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: The main character's Yorkshire accent is consistent with the show's setting in Southern England, though the specific filming towns in Wales are not named in the available evidence

### Sample qacc_940e6d9275f4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d5
- **Claim**: Billy Idol

### Sample qacc_946ecfb478b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4
- **Claim**: Justin Timberlake, Max Martin Shellback wrote "Can't Stop the Feeling!"

### Sample qacc_950881e7c998

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: Boston Red Sox

### Sample qacc_9a9a28d7e159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The Fairy Tail anime's final season has already been released — the third and final season aired from October 7, 2018 to September 29, 2019, confirming the series has long since concluded

### Sample qacc_9a9a28d7e159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: This supersedes earlier reports that had indicated a 2018 release aligns with the fact that the original manga also concluded in 2017, making way for the current sequel series, Fairy Tail: 100 Years Quest, which continues to publish new chapters bi-weekly

### Sample qacc_9b16fd6882f3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: Russ Ballard (Argent) wrote and originally sang “God Gave Rock and Roll to You”; Kiss later covered the song in 1991

### Sample qacc_9c2f95b14a78

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The Duluth Model is an intervention program that emphasizes a coordinated community response to domestic violence, placing accountability on offenders rather than victims addressing battering as a pattern of power and control

### Sample qacc_9c2f95b14a78

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: It engages multiple agencies including law enforcement, courts social services to protect victims, monitor offender compliance offer court-ordered educational programs for batterers

### Sample qacc_9c2f95b14a78

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Research has found that participants in Duluth Model interventions were less likely to recidivate compared to non-participants, though the model is not universally regarded as superior to alternative approaches

### Sample qacc_9fbf28f5786f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3
- **Claim**: The International Space Station (ISS) was conceived in 1993 and its construction and assembly began in earnest in 1998 , with the first crewed expedition arriving in October 2000

### Sample qacc_9fbf28f5786f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The station's continuous human presence has been sustained ever since, marking 25 years of occupation as of November 2025

### Sample qacc_a1c73439eca9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_a3c882e062c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The scheduled completion of the Sagrada Familia has been updated to 2026, though some sources note this date applies specifically to the main spire, with the broader goal of finishing the last towers (Glory Façade) more realistically expected in the early 2030s

### Sample qacc_a44267c115d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: Most of the water in the body is located within the intracellular space, representing approximately two-thirds of total body water, while the remaining one-third is found in the extracellular space (interstitial fluid and blood plasma)

### Sample qacc_a4dde83c35da

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d3
- **Claim**: The Ming Dynasty had an autocratic government in which the emperor ruled personally after abolishing the prime minister position, assisted by the Grand Secretariat

### Sample qacc_a4dde83c35da

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: This system was described as absolute and highly centralized, with the founding emperor Zhu Yuanzhang also abolishing the Censorate and relying on trusted eunuchs to maintain control

### Sample qacc_a4dde83c35da

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: The Ming governmental structure persisted continuously from 1368 to 1911, making it one of the most stable in Chinese history

### Sample qacc_a635c2fd4869

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5
- **Claim**: Roberta Flack

### Sample qacc_a6a2f8b1f0b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d3
- **Claim**: 233

### Sample qacc_a6b48b7accc4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: The first official T20 match was played between Sussex and Surrey in England in 2003 , while the first ever T20 international was contested by New Zealand and Australia

### Sample qacc_a6df0af8c2ba

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Hosanna is a Hebrew expression meaning 'save us' or 'save us now,' derived from the words yasha (save) and na (please)

### Sample qacc_a6df0af8c2ba

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4, d3
- **Claim**: It is used as a cry for salvation or deliverance, particularly in Jewish liturgy and during the Feast of Tabernacles is also recorded in the Christian Gospels as a shout of praise greeted upon Jesus's entry into Jerusalem

### Sample qacc_a78a32b7b9a1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Atlanta Falcons

### Sample qacc_a91ae87c969d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Linda Davis

### Sample qacc_a927c4cccc6a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5
- **Claim**: 1977

### Sample qacc_aa94588b9477

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: 14 January 1960

### Sample qacc_aaf0f638e99b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5
- **Claim**: A yellow 35 mph sign is an advisory speed sign, not a regulatory speed limit sign

### Sample qacc_aaf0f638e99b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5
- **Claim**: These 'Horizontal Alignment Signs' with speed advisories are used solely to advise motorists of a safe speed for upcoming curves or roadway changes do not carry legal enforcement power — only black-on-white signs are considered regulatory and enforceable under general statutes

### Sample qacc_aaf0f638e99b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d3
- **Claim**: Similar principles apply in other jurisdictions: a yellow 35 mph sign typically indicates a suggested speed to navigate a curve safely in ideal conditions, but drivers must also consider current traffic, weather road conditions

### Sample qacc_ac5341df9a83

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: Troops for UN Security Council military actions (peacekeeping missions) come from Member States; the Council authorizes deployments via resolution UN Headquarters then liaises with countries to identify and send personnel — drawn from formed units, individual staff officers military observers — who remain under national authority while serving the UN

### Sample qacc_ac5341df9a83

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: This is corroborated by the official UN website, which states that both troops and police for peacekeeping operations are contributed by Member States, consistent with the Security Council's role as the authorizing body

### Sample qacc_b0346f60b6ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: Celebrity Big Brother is broadcast on CBS in the USA

### Sample qacc_b0ee06f2950d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d5
- **Claim**: Season 6 of American Horror Story is titled "Roanoke"

### Sample qacc_b198a514fff8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d3
- **Claim**: New Mexico was admitted to the Union as the 47th state

### Sample qacc_b1bd9518429b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Gibraltar

### Sample qacc_b281f09f0959

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Joseph McCarthy

### Sample qacc_ba7aaa9b36c8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5
- **Claim**: A fire broke out in the West Wing of the White House on Christmas Eve 1929, during a party for the children of presidential aides, destroying much of the wing

### Sample qacc_bc34664caee4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: California

### Sample qacc_bc7e9a7b4a83

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Usain Bolt

### Sample qacc_bf0bff050f03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: New Zealand

### Sample qacc_bfbb5f55a63f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_c264cb69676e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: A synovial joint, specifically a saddle joint (also called incudomalleolar joint), connects the incus and malleus

### Sample qacc_c27400199055

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5
- **Claim**: Ghana

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3
- **Claim**: Seth MacFarlane

### Sample qacc_c675e6cd8ad6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: George Bruns composed the score for Disney's 1973 animated film Robin Hood

### Sample qacc_c69855566c76

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d3
- **Claim**: Paul Reubens plays Pee-wee Herman in *Pee-wee's Big Holiday*

### Sample qacc_c731579bb51c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: 565

### Sample qacc_c88807a22775

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: .22 Long Rifle

### Sample qacc_c9b95dd57e73

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Peter Sarstedt

### Sample qacc_cb5bcdb1ef9b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d3
- **Claim**: Elliott Gould played Trapper John in the 1970 movie MASH, while Wayne Rogers played the character in the subsequent TV series

### Sample qacc_cbddef47777e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: Mishael Morgan

### Sample qacc_cbf25273f973

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: The last name Tavarez is of Spanish and Portuguese origin, derived from the habitational name Tavares found in Portugal and Tavarez in the Azores

### Sample qacc_cbf25273f973

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: It is also used in the Dominican Republic, Cuba Mexico shares roots with the surnames Tavares and Tabárez

### Sample qacc_cbf25273f973

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Research on people with the Tavarez surname suggests recent ancestry links to Cuba and Mexico, as well as broader Spanish and Portuguese genetic origins

### Sample qacc_ce4983c8a9c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Most effigy mounds were built between approximately 700 and 1200 CE, with the majority constructed during the Late Woodland period (roughly 750–1050 CE)

### Sample qacc_d00b0063e747

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: yes

### Sample qacc_d03e85bdc95a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Aristotle

### Sample qacc_d39801b5de65

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4, d3
- **Claim**: The Continental Congress voted to adopt the Declaration of Independence on July 4, 1776

### Sample qacc_d3b85d857358

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The plane that dropped the atomic bomb on Hiroshima was the Enola Gay, a Boeing B-29 Superfortress bomber

### Sample qacc_d3b85d857358

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d5
- **Claim**: On August 6, 1945, the Enola Gay dropped the atomic bomb code-named 'Little Boy' over Hiroshima, killing approximately 70,000 people instantly and destroying roughly three-quarters of the city

### Sample qacc_d3b85d857358

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The aircraft was named after Enola Gay Tibbets, the mother of the mission's pilot, Colonel Paul Tibbets is currently preserved at the Smithsonian's National Air and Space Museum

### Sample qacc_d44802dc3c96

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: November 1936

### Sample qacc_d60bf850c4ff

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Cadbury sells its products in over 50 countries

### Sample qacc_d78d45c0e30f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Colombia and Japan advanced to the round of 16 from Group H in the 2018 FIFA World Cup

### Sample qacc_d7c6682b5335

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_d7df0a1856b7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3
- **Claim**: The Milky Way is classified as a barred spiral galaxy (Hubble type SBc or Sc), making it a member of the spiral galaxy group within the Hubble sequence

### Sample qacc_d8b24beb2f90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: The balance sheet

### Sample qacc_d96b47272030

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d5
- **Claim**: September 23, 1889

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d3
- **Claim**: XXXTENTACION

### Sample qacc_e06ada156e0e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d5
- **Claim**: Nicole Gale Anderson

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: In Mexico, toll roads are called autopistas (or cuota highways) the federal agency that operates many of them is called CAPUFE (Caminos y Puentes Federales)

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: The toll fee is specifically called a 'cuota,' and by law, every toll autopista has a parallel free route with the suffix 'libre.'

### Sample qacc_e6d89fce1b8e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Owen Hunt

### Sample qacc_e7318f6f3bbe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4, d3
- **Claim**: strengths

### Sample qacc_e87ffc07efd1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: Franklin D. Roosevelt

### Sample qacc_ea469c846404

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: 2025–26

### Sample qacc_eb6f14795c45

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d5
- **Claim**: Joan Cusack

### Sample qacc_eb7c676e133e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: The last time an astronaut went to the moon was December 14, 1972, during the Apollo 17 mission

### Sample qacc_eb7c676e133e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: Apollo 17 Commander Eugene Cernan was the last to step off the lunar surface, closing out the final extravehicular activity after three days of exploration at the Taurus-Littrow site

### Sample qacc_eb7c676e133e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: This mission remains the most recent human landing on the moon, making Cernan the last person to walk on the lunar surface

### Sample qacc_ec5b0067c29a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: Number One Observatory Circle

### Sample qacc_ecbc6adf8a48

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The retrieved evidence places the writing of the First Epistle of John within the first century, with one source specifically suggesting 70–90 AD , while scholarly consensus generally places it later, around 95–110 AD

### Sample qacc_ecbc6adf8a48

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Older sources, such as d3, present a broader range including dates before 70 AD, though most scholars consider these less likely given the epistle's engagement with developed theological controversies characteristic of a later period

### Sample qacc_ecbc6adf8a48

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: In summary, while there is no single universally accepted date, the weight of recent scholarly evidence points to the 90s or early 100s AD as the most probable period of composition

### Sample qacc_ecd3d9c0ca11

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: Guy Norris

### Sample qacc_f10c7ad4bb81

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Acronyms and initialisms

### Sample qacc_f1776add7672

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4
- **Claim**: ICD-10 codes have a flexible length depending on the version and extension level

### Sample qacc_f1776add7672

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: For standard ICD-10-CM codes, the core structure is three to seven characters — starting with a letter, followed by numbers and letters — with a decimal point after the third character if the code exceeds three digits

### Sample qacc_f1776add7672

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: For specific inpatient procedures (ICD-10-PCS), each code is fixed at seven alphanumeric characters

### Sample qacc_f2218f8c979e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d3
- **Claim**: The rib primal

### Sample qacc_f69c37496013

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: 1987

### Sample qacc_fbdae168fc6f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Indira Gandhi

### Sample qacc_fbe562911999

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d3
- **Claim**: In the Indian Warrant of Precedence, the Speaker of the Lok Sabha is placed at Sl

### Sample qacc_fbe562911999

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d3
- **Claim**: No. 6 , ranking above the Chief Justice of India and below only the President, Vice-President, Prime Minister, Governor former Presidents

### Sample qacc_fbe562911999

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is confirmed across multiple sources, including the official Parliament of India documents and independent references

### Sample qacc_ff2cb00f4c03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: 7

### Sample situatedqa_geo_00f6ee9a6705

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: The Villages are located in Florida, specifically spanning the three Florida counties of Lake, Sumter Marion

### Sample situatedqa_geo_082db791e263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d3
- **Claim**: It depends on jurisdiction; federal US law generally requires 18, though state rules and permits may raise the minimum age

### Sample situatedqa_geo_09a6a048cfbf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: It depends on where you are; in the United States (federal minimum legal drinking age is 21 years old)

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: In Ontario, red licence plates indicate that a vehicle either belongs to a motor vehicle dealer or to a diplomat; blue-and-white standard plates are the most common personal plates

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: Globally, red plates carry different meanings depending on the jurisdiction — for example, in Spain they identify vehicles undergoing registration or temporarily out of service in Turkey they denote senior executive vehicles

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: For the United States specifically, World War II casualties amounted to approximately 416,800 military deaths and 1,700 civilian deaths , making a combined total of roughly 418,500

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: On a broader scale, the Allies and Axis powers collectively suffered nearly 70 million deaths, with the Soviet Union bearing the heaviest losses

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: For a more detailed breakdown, the National Archives also maintains separate casualty lists for the Navy, Marine Corps Coast Guard, available on their website

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The minimum age to drive a transport vehicle varies by jurisdiction and vehicle type, but federal employment regulations provide a useful reference point

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Employees under 16 years of age may not drive motor vehicles on public roads as part of their jobs, while 17-year-olds may drive in limited circumstances — such as during daylight hours, with a valid state license in vehicles not exceeding 6,000 pounds gross vehicle weight — provided such driving is occasional and does not exceed 20% of their worktime

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: For commercial transport specifically, some companies like Classic Transport require drivers to be at least 23 years old many states set their own minimums for different classes of vehicles

### Sample situatedqa_geo_362420ec2416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Sikkim

### Sample situatedqa_geo_3815eef77a3c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The welfare state was introduced at different times across nations

### Sample situatedqa_geo_3815eef77a3c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: In the United States, President Roosevelt established the American welfare state in the 1930s with the New Deal legislation, while in Europe it traces back to the late 19th century, with Germany under Bismarck serving as an early pioneer

### Sample situatedqa_geo_3815eef77a3c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: Specific milestones include Germany's social insurance legislation of the 1880s, Britain's Liberal reforms of 1906–1914 the formal founding of Britain's modern welfare state in 1948 following the Beveridge Report

### Sample situatedqa_geo_48e19f6b37bc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: California is the 3rd largest U.S. state by area, with approximately 163,695 square miles

### Sample situatedqa_geo_48e19f6b37bc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: This is confirmed across multiple sources, including the U.S. Census Bureau data cited by Britannica, which lists California directly after Alaska and Texas in the national ranking

### Sample situatedqa_geo_48e19f6b37bc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: While California is not the 3rd largest state in the world, which would be a different geographic scope, it precisely matches the query regarding U.S. state rankings

### Sample situatedqa_geo_4cb699778b59

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: 6 years

### Sample situatedqa_geo_5f5c20228969

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_66684169f016

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: The Dandi March was led by Mahatma Gandhi and involved thousands of participants, including notable figures such as Mithuben Petit and Pyare Lal Nayar

### Sample situatedqa_geo_66684169f016

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The march began with Gandhi and seventy-nine Ashramites the ranks swelled to approximately 60,000 people who were eventually arrested for civil disobedience

### Sample situatedqa_geo_66684169f016

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: A partial list of companions who walked alongside Gandhi from Sabarmati to Dandi included 31 Gujarati members, 13 Maharashtrians 8 individuals from Uttar Pradesh, among others

### Sample situatedqa_geo_6cca7fdb9b41

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The global point furthest from any ocean is the Eurasian pole of inaccessibility, located in northwestern China near the Kazakhstan border

### Sample situatedqa_geo_6cca7fdb9b41

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: For Britain specifically, the village of Coton in the Elms in Derbyshire is often cited as the furthest point from the sea, with Church Flatts Farm situated approximately 113km (70 miles) from the nearest coastline ; alternatively, some sources claim this title belongs to Ashby de la Zouch, Cross-in-Hand Tring depending on the definition used

### Sample situatedqa_geo_7222d6123c27

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: Calcutta became the capital of British India in 1772, when Warren Hastings transferred all important offices there

### Sample situatedqa_geo_7222d6123c27

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4
- **Claim**: It remained the capital for nearly 140 years until 1911, when the British decided to move the seat of government to Delhi during the Delhi Durbar

### Sample situatedqa_geo_7222d6123c27

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This shift was formally completed in 1931 with the inauguration of New Delhi as the new capital

### Sample situatedqa_geo_74e7e677cae5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: 1935

### Sample situatedqa_geo_779fd84224fa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: Sydney Cove (also referred to as Sydney Harbour); the First Fleet arrived there in January 1788

### Sample situatedqa_geo_7f4d90dffb38

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_80c72c2797bf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: The United States operates under a constitutional republic framework, as confirmed by the White House's official description of the three equal branches — legislative, executive judicial — and the requirement that all states maintain a republican form of government per the U.S. Constitution

### Sample situatedqa_geo_80c72c2797bf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This is further corroborated by USA.gov, which details how the federal government is structured into these three branches to ensure checks and balances by Wikipedia's broader classification of democracies as a major form of government alongside others like monarchies and dictatorships

### Sample situatedqa_geo_85af31651715

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: England (1 July 2007, Health Act 2006); Scotland (26 March 2006); Wales (2 April 2007)

### Sample situatedqa_geo_864fc7e36714

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3
- **Claim**: Historically, the bulk of U.S. immigrants came from Europe, but this shifted to Latin America and Asia since 1965

### Sample situatedqa_geo_897e47478bbc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: 649,481

### Sample situatedqa_geo_8c889f8ce07a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d5
- **Claim**: Under U.S. law, treaty ratification is a joint process involving both the executive and legislative branches: the President submits treaties to the Senate, the Senate Foreign Relations Committee considers and reports them the full Senate must approve by a two-thirds majority before the President can proclaim entry into force

### Sample situatedqa_geo_8c889f8ce07a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The Senate does not actually ratify treaties itself; rather, it provides advice and consent the final act of ratification—signing and depositing the instrument of ratification—is performed by the President

### Sample situatedqa_geo_932ca9fa3259

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Levee maintenance responsibilities vary by ownership and jurisdiction

### Sample situatedqa_geo_932ca9fa3259

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: The U.S. Army Corps of Engineers (USACE) is responsible for building, maintaining inspecting USACE-owned levees , while levee owners and operators are responsible for the everyday care, maintenance emergency response for levees under their control

### Sample situatedqa_geo_932ca9fa3259

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3
- **Claim**: Historically, the Mississippi River Commission (aided by the Army Corps of Engineers) was established in 1879 to maintain levees along the Mississippi local entities such as levee boards also hold responsibility for specific levee systems ; d5> For any specific levee, the National Levee Database provides the responsible entity

### Sample situatedqa_geo_9f6c5f849b40

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: The most populous cities in the world are Tokyo (Japan), Shanghai (China) Jakarta (Indonesia), with respective 2025 population estimates of 33.4 million, 29.6 million 41.9 million

### Sample situatedqa_geo_9f6c5f849b40

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: These figures come from global rankings that distinguish clearly between city proper populations and larger urban agglomerations

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
- **Supporting Docs Found**: d2, d3
- **Claim**: This 1970 version superseded earlier federal air pollution laws passed in 1955 and 1963, making it the most current and comprehensive Clean Air Act legislation

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: Eisenhower

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: The California state flag features a California grizzly bear (Ursus arctos californicus), which is the official state animal of California

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: The grizzly bear was placed on the flag during the 1846 Bear Flag Revolt, when American settlers captured Sonoma and raised a flag bearing a bear as a symbol of strength and resistance

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: It is worth noting that the California grizzly bear is an extinct subspecies of the brown bear — fully extirpated by the 1920s — making California the only state to carry the image of an extinct animal on its flag

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
- **Claim**: Scotland are the current holders of the Calcutta Cup, having won the Six Nations fixture in 2026

### Sample situatedqa_geo_eec0ebc0dd88

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: This supersedes older records showing England winning in 1997 or Scotland claiming the trophy in 2018 , as the 2026 result is the most recent available

### Sample situatedqa_geo_f2031e426cee

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Arjun Ram Meghwal is the present Law Minister of India (Ministry of Law and Justice)

### Sample situatedqa_geo_f2031e426cee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: He is also serving as Cabinet Minister for the Ministry of Parliamentary Affairs

### Sample situatedqa_geo_f2031e426cee

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This supersedes earlier information identifying Kiren Rijiju as Law Minister, as Mr. Meghwal's appointment is confirmed by the official Wikipedia profile of the Ministry of Law and Justice

### Sample situatedqa_geo_f26078ec6467

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Spain

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: The first form of national government after the Revolutionary War was the Articles of Confederation, adopted by the Second Continental Congress on November 15, 1777 ratified by the states in 1781

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This document established a loose confederation of states with a weak central government, creating a 'league of friendship' in which state power was largely preserved

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: After the Treaty of Paris in 1783 formally ended the war, Americans continued to experiment with this system, but influential groups found it inadequate, eventually leading to the Constitutional Convention and the drafting of the current U.S. Constitution

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The shift from tea to coffee in the U.S. began during the American Revolution when the Boston Tea Party made imported tea politically unfashionable coffee—grown in the New World—became the patriotic alternative

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: This cultural shift persisted even after the Revolution coffee completely eclipsed hot tea by 1865 following the Civil War, when returning veterans continued drinking it after receiving it as standard rations

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: In modern times, the trend has reversed in some contexts, with individuals and groups switching back to tea for health reasons or personal preference , illustrating that the relationship between the two beverages has evolved over time rather than changing definitively at a single point

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: The Federal Open Market Committee (FOMC) is the primary body that sets U.S. monetary policy

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: It is a Federal Reserve System entity consisting of the seven members of the Board of Governors, the president of the Federal Reserve Bank of New York four other Reserve Bank presidents serving on a rotating basis

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The FOMC meets regularly — typically every six weeks — to discuss the economic outlook and make decisions on key monetary tools such as interest rates and the money supply

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: While the FOMC is the chief policymaking body, the Federal Reserve Banks and the Board of Governors also play roles in implementing monetary policy the FOMC's decisions are informed by their participation

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: In the United States, environmental policy is set at both the federal and state levels

### Sample situatedqa_temp_051502801f9c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_05d714be23fd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: Ludacris

### Sample situatedqa_temp_0ad0871fcc33

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d5
- **Claim**: Wilt Chamberlain

### Sample situatedqa_temp_0c2289f57504

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: Hamid Ansari

### Sample situatedqa_temp_14a587def215

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: 2026

### Sample situatedqa_temp_14f29fe7ff15

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3
- **Claim**: The British under General Howe defeated the Continental Army at the Battle of Brandywine on September 11, 1777, opening the way for the British conquest of Philadelphia

### Sample situatedqa_temp_14f29fe7ff15

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: This engagement was the largest single-day battle of the American Revolutionary War in terms of manpower while Washington's army was outmaneuvered and forced to retreat to Valley Forge, the battle left the Revolutionary army intact, making it a strategic turning point that convinced France to join the war effort

### Sample situatedqa_temp_14f70522567e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: Lionel Messi

### Sample situatedqa_temp_180f238d8296

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Australia (5), West Indies (2), India (2), England (1), Pakistan (1), Sri Lanka (1) — for the ODI Cricket World Cup; T20 data is partially covered in the retrieved evidence

### Sample situatedqa_temp_1987d35f994b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: October 27, 1986

### Sample situatedqa_temp_19badef7553b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: The Philadelphia Eagles won the Super Bowl twice in the retrieved evidence

### Sample situatedqa_temp_19badef7553b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: Their first championship came on February 4, 2018, when they defeated the New England Patriots 41-33 in Super Bowl LII, marking their first title since 1960

### Sample situatedqa_temp_19badef7553b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Their second title was claimed on February 9, 2025, when they defeated the Kansas City Chiefs 40-22 in Super Bowl LIX

### Sample situatedqa_temp_1baff64de20e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Rumer Willis

### Sample situatedqa_temp_1c56e575f096

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: New South Wales last won the State of Origin series in 2024

### Sample situatedqa_temp_23d05dc2d7dc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: LeBron James

### Sample situatedqa_temp_2cafa66bbe5d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: 23 miles

### Sample situatedqa_temp_2f45ec399f17

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Novak Djokovic

### Sample situatedqa_temp_2f5dc02a5ce1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: Cory A. Booker

### Sample situatedqa_temp_2f73dc6c2df1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d3
- **Claim**: Mariah Carey

### Sample situatedqa_temp_301378915064

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: Merritt Wever

### Sample situatedqa_temp_3026b0491e32

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: John Williams composed the music for the first three Harry Potter films: *Philosopher's Stone*, *Chamber of Secrets* *Prisoner of Azkaban*

### Sample situatedqa_temp_32d33d503f69

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4, d3
- **Claim**: Henry Danger: The Movie is coming out on January 17, 2025

### Sample situatedqa_temp_35156c8be377

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: The answer depends on the metric used to define 'richest.' By total GDP, South Africa is generally considered the wealthiest African country, with an estimated 2024 GDP of $403 billion , surpassing Nigeria's 2021 GDP of $377 billion and consistent with earlier data showing Nigeria as the top GDP earner in 2016

### Sample situatedqa_temp_35156c8be377

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: By GDP per capita, however, Seychelles leads all of Africa at an estimated $42,110 in 2025 , reflecting the island nation's high-value services and tourism sector

### Sample situatedqa_temp_35156c8be377

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: In summary, South Africa ranks first by total economic output, while Seychelles ranks first by per-capita income

### Sample situatedqa_temp_3521e5dc831c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5
- **Claim**: Gagan Narang

### Sample situatedqa_temp_3781ac7b3ead

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Darren Criss

### Sample situatedqa_temp_3df0e6082901

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: LSU

### Sample situatedqa_temp_40e6764f611f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Mort is a mouse lemur (family Cheirogaleidae), specifically identified as a Goodman's mouse lemur in Madagascar and its spin-offs

### Sample situatedqa_temp_43c0aaae9828

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: Hillsong Worship

### Sample situatedqa_temp_44d30ce08bf5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: UCLA

### Sample situatedqa_temp_45cbc4dc3e28

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: The current Chief Justice of the Sindh High Court is **Justice Zafar Ahmed Rajput**, who was appointed to the position on 6th December 2025 and has continued in office since that date

### Sample situatedqa_temp_45cbc4dc3e28

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: This is confirmed by the official Sindh High Court list of Chief Justices, which shows his tenure extending from 6th December 2025 to the present

### Sample situatedqa_temp_45cbc4dc3e28

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Earlier records, including Wikipedia and the High Court's own pages, had listed him as Acting Chief Justice from 15th September 2025, but these designations have since been superseded by his formal appointment as Chief Justice

### Sample situatedqa_temp_4ead2f4cd2d5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3
- **Claim**: Bethany Bryant

### Sample situatedqa_temp_4fbe76f974ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The song became widely known through Judy Garland's 1939 film The Wizard of Oz, but the specific release date of the song itself is not explicitly stated in the retrieved evidence

### Sample situatedqa_temp_50748f92be3a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: The last FIFA World Cup was held in 2022 Argentina won the title after defeating France in the final on December 18, 2022

### Sample situatedqa_temp_50748f92be3a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: This result is further corroborated by the 2022 edition listed in the official FIFA champions table, which names Argentina as the winner with Lionel Scaloni as head coach

### Sample situatedqa_temp_50748f92be3a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: While the snippet also mentions the 2026 tournament, it does not identify the winner of that edition

### Sample situatedqa_temp_557d04a28d7f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: LeBron James

### Sample situatedqa_temp_587e89bbcbe1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d3
- **Claim**: 108

### Sample situatedqa_temp_5a59faf24972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: The latest stable version of Android is Android 16, which was released on June 10, 2025

### Sample situatedqa_temp_5a59faf24972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: This supersedes earlier information identifying Android 15 as the latest version, as d4 and d5 confirm that Android 16 has since been released for Pixel phones and other manufacturers

### Sample situatedqa_temp_5e6b388e88aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The Colorado Avalanche last won the Stanley Cup in 2022, defeating the Tampa Bay Lightning 2-1 in Game 6 on June 26, 2022

### Sample situatedqa_temp_5e6b388e88aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: This was the team's third overall championship and first in 21 years, marking a historic comeback after the Avalanche had previously won in 1996 and 2001

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
- **Supporting Docs Found**: d4, d3
- **Claim**: Düsseldorf, Germany

### Sample situatedqa_temp_61d9095f5b70

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: July 23, 1986

### Sample situatedqa_temp_657c130afab6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4, d3
- **Claim**: 1980

### Sample situatedqa_temp_6a683d0d04f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: Five sharps in a key signature signify the key of B major

### Sample situatedqa_temp_6a683d0d04f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Using the standard order of sharps—F♯, C♯, G♯, D♯, A♯—the key can be identified by remembering that the tonic is a half-step above the last sharp written ; alternatively, the mnemonic 'Fast Cars Go Dangerously Around Every Bend' helps encode this sequence the circle of fifths confirms B major as the destination after ascending five steps from C

### Sample situatedqa_temp_6a8aed3e3d5f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: episode 245

### Sample situatedqa_temp_6c424fc78a69

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The Pakistan Tehreek-e-Insaf (PTI) party won the 2018 general election in Pakistan, becoming the first political force in the National Assembly with 157 seats out of 342

### Sample situatedqa_temp_6c424fc78a69

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Led by Imran Khan, PTI defeated the Pakistan Muslim League-Nawaz (PML-N), which came second with 84 seats, securing a coalition government on August 17, 2018

### Sample situatedqa_temp_6c424fc78a69

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: This is further corroborated by opinion poll data showing PTI leading the race with a double-digit margin over PML-N expert analysis confirming Imran Khan's victory in the July 2018 election

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: Todd Monken

### Sample situatedqa_temp_78c89bf81e8b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: On naval ships, "SS" generally stands for "submersible ship," used in hull classification symbols such as SSN (nuclear-powered attack submarine), SSBN (nuclear-powered ballistic missile submarine) SSGN (nuclear-powered guided missile submarine)

### Sample situatedqa_temp_78c89bf81e8b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: In earlier merchant marine contexts, SS traditionally stood for "steamship," but this usage is largely historical

### Sample situatedqa_temp_7a7a7f43e575

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: Washington

### Sample situatedqa_temp_7cd18101326e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Indiana QB Fernando Mendoza was named the Offensive MVP of the January 2026 national championship game

### Sample situatedqa_temp_7d5f559b313d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Most recent available data from Moody's Analytics shows the United States Nominal GDP reached 31,819,464 million USD in Q1 2026

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Australia's coastline length varies depending on the measurement methodology and data source

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3
- **Claim**: The official government survey data from Geoscience Australia records a total coastline length of 59,681 km, comprising 35,821 km of mainland coastline and 23,860 km of island coastline , while a scientific study published in Nature confirms this figure and explains how different measurement scales yield consistent results

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: Earlier reports citing lower figures, such as 25,760 km or approximately 22,292 miles , reflect outdated datasets or different definitional scopes; the most accurate and up-to-date figure, as confirmed by high-precision mapping techniques, is approximately 59,681 km

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_7f07f7a6c607

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d3
- **Claim**: Mohamed Salah

### Sample situatedqa_temp_8808c106a115

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Tay-Sachs disease is an autosomal recessive genetic disorder caused by a deficiency of the hexosaminidase A (HEXA) enzyme, which leads to the buildup of GM2-ganglioside in nerve cells

### Sample situatedqa_temp_8808c106a115

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5
- **Claim**: It is inherited when a child receives two variant copies of the HEXA gene — one from each parent — and the specific type of mutation determines the severity and onset of symptoms, ranging from classic infantile to juvenile and late-onset forms

### Sample situatedqa_temp_88ef1cfab62e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: Hunter Emery plays CO Rick Hopper in Orange is the New Black

### Sample situatedqa_temp_8b66ca78c3d1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: 11,937

### Sample situatedqa_temp_8b8427cd7b6d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5
- **Claim**: The Cumberland River begins in the mountains of eastern Kentucky, formed by the confluence of its headwater forks — Poor Fork, Clover Fork Martin's Fork — near Harlan County

### Sample situatedqa_temp_8b8427cd7b6d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4
- **Claim**: It flows generally westward through the mountains of Kentucky before turning south into Tennessee, traveling through Nashville then bending northwest back into Kentucky before finally joining the Ohio River at Smithland, Kentucky

### Sample situatedqa_temp_8f57822ed7d2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d3
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
- **Supporting Docs Found**: d2
- **Claim**: California's total gas tax stood at approximately 70 cents per gallon, making it the highest in the country

### Sample situatedqa_temp_9049c197a579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: This figure has grown from roughly 90 cents per gallon in March 2025, reflecting updates in state and federal tax rates

### Sample situatedqa_temp_9049c197a579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The official California Department of Tax and Fee Administration data further corroborates this trend, showing the excise tax rate at $0.612 per gallon for the period beginning July 2025

### Sample situatedqa_temp_956e1f6e518e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d3
- **Claim**: The last time anyone was on the moon was December 1972 during NASA's Apollo 17 mission

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
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Ramesh Kuntal Megh

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4, d3
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_a5e5db28902b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: 23,000,000

### Sample situatedqa_temp_a8024d95c6e1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: Episode 10

### Sample situatedqa_temp_a8c74bced99a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d5
- **Claim**: March 13, 624 CE

### Sample situatedqa_temp_ac0e6a4a7e32

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d3
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

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Mitchell was 23 when the show first aired in 2010 she has since aged alongside her character, who was recast as a 23-year-old adult in Season 6B

### Sample situatedqa_temp_ae0882e48812

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The two largest deserts in China are the Gobi Desert and the Taklamakan Desert

### Sample situatedqa_temp_b3ad3248b7d4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d3
- **Claim**: The Inca Empire began in 1438 during the reign of Pachacuti, who expanded the Kingdom of Cusco into a full-blown empire it ended in 1533 when the last Sapa Inca, Atahualpa, was captured and killed by Spanish conquistadors led by Francisco Pizarro

### Sample situatedqa_temp_b3ad3248b7d4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d3
- **Claim**: This date is consistent across multiple sources, including Britannica, which notes that the Spanish conquest of the Inca empire began in 1532 a detailed Inca timeline that places Atahualpa's capture on July 26, 1533

### Sample situatedqa_temp_b6c75b99a2f4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d3
- **Claim**: 700 nm

### Sample situatedqa_temp_b797de4c6610

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Cardiac biomarkers are substances that appear in the blood when the heart is stressed or damaged they are used to diagnose and monitor heart disease

### Sample situatedqa_temp_b797de4c6610

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d5
- **Claim**: The most commonly used cardiac biomarker is cardiac troponin (troponin T or I), which enters the bloodstream shortly after a heart attack and remains elevated for days, making it the preferred marker per AHA guidelines ; other traditional markers include creatinine kinase (CK), CK-MB (a heart-specific subtype of CK) natriuretic peptides like BNP

### Sample situatedqa_temp_b797de4c6610

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Additional biomarkers that have been used but are less specific or less frequently employed today include aspartate aminotransferase (AST), lactate dehydrogenase (LD) myoglobin

### Sample situatedqa_temp_bb7a678b9ec9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5, d3
- **Claim**: Los Angeles (1932, 1984, 2028 Summer), Lake Placid (1932, 1980 Winter), Atlanta (1996 Summer), Palisades Tahoe/Squaw Valley (1960 Winter) St. Louis (1904 Summer)

### Sample situatedqa_temp_bb85e25a8159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: Florida Panthers

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: HMS Queen Elizabeth was commissioned on December 7, 2017 formally declared operational in 2020

### Sample situatedqa_temp_bdc0853f9a23

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: India's rank in the 2018 Global Peace Index was 136th out of 163 countries

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: The last name Gerard is of French and Norman origin, derived from the Old French personal name Gérard, which itself traces to the ancient Germanic elements gēr meaning 'spear' and hard meaning 'hardy,' 'brave,' or 'strong'

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: It is also found in Haiti and has variant spellings such as Gerrard, Gerhardt Gérard across different regions

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: In England, the surname is further documented as early as the Domesday Book of 1086, listing the Latin forms Gerardus and Girardus is associated with the ancient Lancashire family seat

### Sample situatedqa_temp_c92bc2a37a57

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_ce10e6f883fb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: India and Pakistan

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The WTO has 166 members , making it the most up-to-date figure available

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This count supersedes earlier reports of 164 members, which reflected the organization's size as of July 2016 or undated sources , as the WTO has continued to grow through new accessions over time

### Sample situatedqa_temp_d0579ca3907c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: The Battle of Kadesh reportedly started in late May 1274 BCE, though sources differ on the specific start date; the most cited is Year 5 III Shemu day 9 of Ramesses II the battle is generally considered to have ended on the same day

### Sample situatedqa_temp_d15849658c20

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Oleksandr Usyk is the current IBF, WBA Super WBC heavyweight world champion, while Daniel Dubois holds the WBO title as of May 9, 2026

### Sample situatedqa_temp_d15849658c20

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: This means no single boxer currently holds all four major titles (IBF, WBO, WBA IBO) simultaneously, making the unified champion status unoccupied under the strict definition of holding all four

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4, d3
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_d59edcfab87f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: Queen Charlotte of Mecklenburg-Strelitz, a German princess and queen consort who married King George III of Great Britain in 1761, is the namesake of the city of Charlotte, North Carolina

### Sample situatedqa_temp_d59edcfab87f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d3
- **Claim**: The city was officially incorporated in 1768 as 'Charlotte Town' and has been known as the Queen City ever since, honoring Queen Charlotte's legacy

### Sample situatedqa_temp_d59edcfab87f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: This naming tradition is consistently confirmed across multiple sources, including the city's own official records and encyclopedic references

### Sample situatedqa_temp_d63c04b198fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: 133

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3
- **Claim**: August 20, 1989

### Sample situatedqa_temp_dc4d0c4e24ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_dcdb77c2f3d9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: Paris, France

### Sample situatedqa_temp_df5975a9678a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d3
- **Claim**: Saina Nehwal

### Sample situatedqa_temp_e3caa7edab1f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: 73

### Sample situatedqa_temp_ebd3c33e8be8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: Jonathan Bailey

### Sample situatedqa_temp_eed796e2fbed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: Scottie Scheffler

### Sample situatedqa_temp_eeebc5ea5c4f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3
- **Claim**: The highest grossing movie in the Philippines is the romantic comedy **Hello, Love, Again**, which earned approximately ₱1.6 billion in domestic box office revenue

### Sample situatedqa_temp_eeebc5ea5c4f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3
- **Claim**: This film, a collaboration between ABS-CBN Studios, Star Cinema GMA Pictures, surpassed the previous record holder, *Rewind* (₱924 million), by a wide margin

### Sample situatedqa_temp_eeebc5ea5c4f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: Earlier sources, such as a 2015 report, had identified the 2013 romantic comedy *It Takes a Man and a Woman* (₱405 million) as the all-time leader , but that record has since been superseded by the phenomenal performance of *Hello, Love, Again*

### Sample situatedqa_temp_efa79b1e7fb1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: Stephen Curry (4,248 career 3-pointers as of April 2026); Ray Allen (2,973)

### Sample situatedqa_temp_f099d9d1452b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: John Ratcliffe is the current Director of the CIA, having been officially sworn in on January 23, 2025

### Sample situatedqa_temp_f099d9d1452b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: He is the first person ever to serve as both CIA Director and Director of National Intelligence

### Sample situatedqa_temp_f099d9d1452b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Additional reporting by CNN and other outlets further corroborate his appointment and ongoing role as CIA chief

### Sample situatedqa_temp_f196a847a496

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: 7

### Sample situatedqa_temp_f1e9ce4608b3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5
- **Claim**: Azzi Fudd

### Sample situatedqa_temp_f3b3face1b6b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4, d3
- **Claim**: McDonald's Monopoly pieces typically come on the packaging of various eligible menu items, including breakfast sandwiches, burgers fries

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: 2021

### Sample situatedqa_temp_f971e49123a1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: 13

### Sample wikirevision_0001

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: Twitter is currently known as X. In April 2023, Twitter merged with X Holdings and ceased to be an independent company, becoming part of X Corp. This rebranding is confirmed by the older Wikipedia revision of Twitter, which redirects to the newer article on X, as well as by the dedicated article on X, which explicitly states that X is the current name of the platform formerly known as Twitter

### Sample wikirevision_0002

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3
- **Claim**: Twitter is now known as X. In April 2023, Twitter merged with X Holdings and ceased to be an independent company, becoming part of X Corp. This rebrand was confirmed when Wikipedia's article on Twitter redirected to the article on X, indicating the name change had taken effect

### Sample wikirevision_0002

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: As a result, Twitter's former name has been superseded by X as the current designation for the platform

### Sample wikirevision_0003

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: Twitter is now known as X. In April 2023, Twitter merged with X Holdings and ceased to be an independent company, becoming part of X Corp. This rebranding is confirmed across multiple sources, with Wikipedia redirecting 'Twitter' to the article on X (social network), which notes the name change from 2006–2023

### Sample wikirevision_0004

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The current name of Facebook's parent company is Meta Platforms, Inc. This was confirmed when Facebook rebranded itself as Meta Platforms, Inc. in October 2022, officially changing its corporate identity

### Sample wikirevision_0004

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The rebranding is further corroborated by additional context showing that Meta Platforms, Inc. is the parent company behind Facebook's products and initiatives

### Sample wikirevision_0007

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Google is owned by Alphabet Inc., a parent company formed in 2015 when Google reorganized as a wholly owned subsidiary

### Sample wikirevision_0007

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Alphabet holds Google as its largest subsidiary and is itself a public company traded on the Nasdaq

### Sample wikirevision_0007

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: This structure has remained consistent over time, with Alphabet serving as the parent company that owns Google

### Sample wikirevision_0010

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Microsoft

### Sample wikirevision_0013

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Microsoft acquired LinkedIn in December 2016

### Sample wikirevision_0013

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: This makes Microsoft the owner of LinkedIn

### Sample wikirevision_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The latest President of India is Droupadi Murmu, who has been in office since July 2022

### Sample wikirevision_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: She is the 15th President of India and holds the highest office of the country, serving as the ceremonial head of state and supreme commander of the Indian Armed Forces

### Sample wikirevision_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: This is consistent across multiple sources, including the newer Wikipedia revision dated May 2026, which also confirms her continued tenure

### Sample wikirevision_0028

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The latest Prime Minister of India is Narendra Modi, who has served in office since 26 May 2014

### Sample wikirevision_0028

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: He is the Honourable Mr. Prime Minister and holds the highest office of the Government of India, being appointed by the President and responsible to the Lok Sabha

### Sample wikirevision_0032

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
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
- **Supporting Docs Found**: d1, d2
- **Claim**: The current Chancellor of Germany is Friedrich Merz, who took office on May 6, 2025

### Sample wikirevision_0033

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: This is confirmed by the official Wikipedia revision that superseded the older version in March 2026, which explicitly names him as incumbent with that date

### Sample wikirevision_0033

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: His tenure as the 53rd Chancellor of the Federal Republic of Germany is further corroborated by the newer revision of the same article

### Sample wikirevision_0035

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: The latest Prime Minister of Japan is Sanae Takaichi, who assumed office on 21 October 2025

### Sample wikirevision_0035

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: She is the first female Prime Minister in the country's history and is currently the incumbent, having been appointed by the Emperor following nomination by the National Diet

### Sample wikirevision_0035

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of the Prime Minister of Japan page, as well as the list of Japanese prime ministers

### Sample wikirevision_0040

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The latest President of Argentina is Javier Milei, who took office on 10 December 2023

### Sample wikirevision_0040

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: He is the incumbent President, serving as head of state and government at the Casa Rosada

### Sample wikirevision_0040

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple up-to-date sources, including the Wikipedia article on the President of Argentina, which also notes that earlier low-quality sources incorrectly identified him as a former president

### Sample wikirevision_0041

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The current President of Argentina is Javier Milei, who took office on 10 December 2023

### Sample wikirevision_0041

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: He is the 54th President of Argentina and belongs to the political party Unión por el Cambio

### Sample wikirevision_0041

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple up-to-date sources, including the Wikipedia article on the President of Argentina, which also notes that the position is the highest political office in the Argentine Nation

### Sample wikirevision_0042

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
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
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Argentina (Argentina)

### Sample wikirevision_0047

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: Argentina (defending 2026 champion, 3rd title)

### Sample wikirevision_0049

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The current Indian Premier League champion is Royal Challengers Bengaluru (RCB), who won the 2026 IPL — their first title in the league's history

### Sample wikirevision_0049

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: This supersedes older information from the 2025 season, in which RCB was also listed as the champion earlier records dating back to the 2023 season

### Sample wikirevision_0049

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The 2026 IPL was the 19th edition of the tournament, played from March 28 to May 31 across 13 venues

### Sample wikirevision_0057

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: Google is owned by Alphabet Inc., a publicly traded company listed on the NASDAQ under the symbols GOOGL (Class A share) and GOOG (Class C share)

### Sample wikirevision_0057

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Alphabet was founded by Larry Page and Sergey Brin in 2015 specifically to serve as Google's parent company since then Google has been reorganized as a wholly owned subsidiary of Alphabet

### Sample wikirevision_0057

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Page and Brin together own approximately 14% of Alphabet's publicly listed shares and control 56% of the company's voting power through super-voting stock

### Sample wikirevision_0061

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: The current President of Mexico is Claudia Sheinbaum Pardo, who took office on 1 October 2024, making her the 66th president of Mexico

### Sample wikirevision_0061

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: She is the first woman and the first Jewish person to hold the office, serving as both head of state and head of government

### Sample wikirevision_0061

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple sources, including the older and newer Wikipedia revisions of the President of Mexico article, as well as her own Wikipedia biography

### Sample wikirevision_0064

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The current President of Turkey is Recep Tayyip Erdoğan, having served in office since 28 August 2014

### Sample wikirevision_0064

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: He is the 12th president of the Republic of Turkey and also serves as the country's head of government and commander-in-chief

### Sample wikirevision_0064

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple sources, including the high-credibility Wikipedia articles on both the President of Turkey and the Vice President of Turkey

### Sample wikirevision_0065

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: Facebook's parent company is Meta Platforms, Inc. This was confirmed in April 2025, when Facebook officially changed its corporate identity to Meta Platforms, Inc. The rebranding was announced in October 2021 and completed in December 2021, making Meta Platforms the current parent company of Facebook

### Sample wikirevision_0066

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: Facebook's parent company is Meta Platforms, Inc. This was confirmed when Facebook rebranded itself as Meta Platforms, Inc. in October 2021, officially changing its corporate identity

### Sample wikirevision_0066

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The rebranding was announced during the company's Connect conference and was intended to reflect a strategic shift toward building the metaverse

### Sample wikirevision_0066

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: As a result, Facebook is no longer its own parent company, but rather a subsidiary of Meta Platforms, Inc

### Sample wikirevision_0067

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The 2025 Ballon d'Or was won by Ousmane Dembélé, making it his first award

### Sample wikirevision_0067

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The ceremony took place on 22 September 2025, recognizing the best footballers of the 2024–25 season

### Sample wikirevision_0067

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: Dembélé remains the most recent holder

### Sample wikirevision_0071

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: Benjamin Netanyahu is the current Prime Minister of Israel, having assumed office on 29 December 2022

### Sample wikirevision_0071

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of the Prime Minister of Israel page, as well as the list of Israeli prime ministers

### Sample wikirevision_0072

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3
- **Claim**: Twitter is currently known as X. In April 2023, Twitter merged with X Holdings and ceased to be an independent company, becoming part of X Corp. This rebrand was confirmed when Wikipedia's redirect from 'Twitter' to 'X (social network)' was updated on May 16, 2026

### Sample wikirevision_0072

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: As a result, the platform that was once called Twitter is now officially called X

### Sample wikirevision_0074

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: The current Vice President of the United States is JD Vance, who assumed office on January 20, 2025

### Sample wikirevision_0074

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: This is confirmed across multiple authoritative sources, including the newer Wikipedia revision and the list of vice presidents of the United States

### Sample wikirevision_0074

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The Office of the Vice President further corroborates his tenure, noting that it is headed by the chief of staff to Vice President JD Vance

### Sample wikirevision_0076

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The latest Prime Minister of Pakistan is Shehbaz Sharif, who took office on 4 March 2024

### Sample wikirevision_0076

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: This is confirmed by the official Wikipedia revision of the Prime Minister of Pakistan page, which lists him as the current incumbent

### Sample wikirevision_0076

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Additional context: This is consistent across multiple sources, including the UN protocol list of Heads of State and Government, as well as the Wikipedia list of Prime Ministers of Pakistan

### Sample wikirevision_0082

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The 2025 Ballon d'Or was won by Ousmane Dembélé (FRA), marking his first time receiving the award

### Sample wikirevision_0082

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The ceremony took place on 22 September 2025, recognizing the best footballers of the 2024–25 season

### Sample wikirevision_0082

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: The Ballon d'Or is scheduled for 26 October 2026, making Dembélé the most recent holder

### Sample wikirevision_0085

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The current Prime Minister of France is Sébastien Lecornu, who took office on 9 September 2025

### Sample wikirevision_0085

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: He is the 32nd Prime Minister of the French Republic and serves under President Emmanuel Macron

### Sample wikirevision_0085

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple sources, including the high-credibility Wikipedia article on the Prime Minister of France, which also notes that the office has been known as 'Prime Minister' since 1959 when Michel Debré became the first holder under the Fifth Republic

### Sample wikirevision_0086

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The current Prime Minister of Pakistan is Shehbaz Sharif, who took office on 4 March 2024

### Sample wikirevision_0086

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: He is the 22nd Prime Minister of Pakistan and serves as the leader of the House and head of the executive branch of the Government of Pakistan

### Sample wikirevision_0086

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple up-to-date sources, including the Wikipedia article on the Prime Minister of Pakistan, which also notes that the officeholder is typically the chairman or president of the party with a majority in the National Assembly

### Sample wikirevision_0088

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The latest Leader of the Labour Party in the UK is Keir Starmer, who was elected to the position on 4 April 2020, following his victory in that year's Labour Party leadership election

### Sample wikirevision_0088

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: This is confirmed by the official Wikipedia entries on both the Leader of the Labour Party and the 2020 Labour leadership election, as well as by the Labour Party's own records

### Sample wikirevision_0088

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Keir Starmer also serves as the Prime Minister of the United Kingdom

### Sample wikirevision_0089

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Calcutta is officially called Kolkata

### Sample wikirevision_0089

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The city officially changed its name from Calcutta to Kolkata in 2001 this is confirmed across multiple sources including the newer Wikipedia revision of Calcutta and the main Kolkata article

### Sample wikirevision_0089

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: While Calcutta is still commonly used, especially outside of India, the official legal name of the city is now Kolkata

### Sample wikirevision_0090

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: The latest President of Indonesia is Prabowo Subianto, who took office on 20 October 2024

### Sample wikirevision_0090

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: He is the eighth president of Indonesia and served as the 26th minister of defence under President Joko Widodo from 2019 to 2024

### Sample wikirevision_0090

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of the President of Indonesia page, as well as his own Wikipedia article

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
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Bangalore is officially called Bengaluru

### Sample wikirevision_0097

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The city officially changed its name from Bangalore to Bengaluru on 1 November 2014, as confirmed by the Government of Karnataka

### Sample wikirevision_0097

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: This change is consistently reflected across all sources, including the Royal Challengers Bengaluru cricket team and the newer Wikipedia revision, which supersede the older 2018 revision that still listed Bangalore as the official name

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The latest Cricket World Cup champion is Australia, who defeated India by six wickets in the 2023 final held in Ahmedabad on 19 November 2023

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This was Australia's sixth Cricket World Cup title the tournament was the 13th edition organized by the ICC

### Sample wikirevision_0099

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: While the 2027 Cricket World Cup is scheduled to take place in South Africa, Zimbabwe Namibia, no confirmed champion exists for that future tournament

### Sample wikirevision_0100

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The current Prime Minister of Pakistan is Shehbaz Sharif, who took office on 4 March 2024

### Sample wikirevision_0100

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: He is the 22nd Prime Minister of Pakistan and serves as the leader of the House and head of the country's executive branch

### Sample wikirevision_0100

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This is consistent across multiple up-to-date sources, including the Wikipedia article on the Prime Minister of Pakistan, which also notes that the office has been held by 20 different individuals since 1947

### Sample wikirevision_0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The current Leader of the Labour Party in the UK is Keir Starmer, who was elected to the position on 4 April 2020, following his victory in that year's Labour Party leadership election

### Sample wikirevision_0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: He has also served as Prime Minister of the United Kingdom since the 2024 general election

### Sample wikirevision_0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: This is consistent across multiple sources, including the older and newer Wikipedia revisions of the Leader of the Labour Party page, both of which confirm Keir Starmer as the current leader

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This change was confirmed by the Haryana Government in 2016 the city is now uniformly referred to as Gurugram in official contexts

### Sample wikirevision_0105

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: Bangalore is officially called Bengaluru

### Sample wikirevision_0105

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3
- **Claim**: The city officially changed its name from Bangalore to Bengaluru on 1 November 2014, as confirmed by the newer Wikipedia revision and the Royal Challengers Bengaluru cricket team's name change

### Sample wikirevision_0105

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This change is consistent across multiple sources, including the Government of Karnataka's own documentation

### Sample wikirevision_0109

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: The current Prime Minister of Canada is Mark Carney, who assumed office on 14 March 2025

### Sample wikirevision_0109

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: He is the 24th person to serve in the role and is affiliated with the Conservative Party

### Sample wikirevision_0109

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple up-to-date sources, including the Wikipedia article on the Prime Minister of Canada, which also notes that the incumbent must have the confidence of the House of Commons

### Sample wikirevision_0111

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: Facebook's parent company is currently called Meta Platforms, Inc. This was confirmed when Facebook rebranded itself as Meta Platforms in 2021, officially changing its corporate identity

### Sample wikirevision_0111

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The rebranding is further corroborated by additional context showing that Meta Platforms, Inc. owns and operates Facebook as well as other major platforms including Instagram, WhatsApp Messenger

### Sample wikirevision_0112

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: The current President of Indonesia is Prabowo Subianto, who took office on 20 October 2024

### Sample wikirevision_0112

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3
- **Claim**: He is the eighth president of Indonesia and serves as both head of state and head of government

### Sample wikirevision_0112

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of the President of Indonesia page, as well as his own Wikipedia article

### Sample wikirevision_0115

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The current Leader of the Conservative Party in the UK is Kemi Badenoch, who was elected to the position on 2 November 2024, following her victory against Robert Jenrick in the party's 2024 leadership election

### Sample wikirevision_0115

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: This is confirmed by the official Wikipedia entries on both the Conservative Party leadership and the Conservative Party in the House of Lords, as well as corroborated by the Guardian's coverage of the 2024 leadership result

### Sample wikirevision_0119

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: Jannik Sinner

### Sample wikirevision_0120

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The 2025 French Open men's singles champion is Carlos Alcaraz, who defended his title by defeating world No. 1 Jannik Sinner in the final

### Sample wikirevision_0120

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: This result is corroborated across multiple sources, including the Wikipedia article on the 2025 French Open and the main French Open article listing current champions

### Sample wikirevision_0120

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: While d3 mentions the 2026 French Open and Alcaraz's withdrawal, it does not identify the 2026 champion

### Sample wikirevision_0121

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The current President of Argentina is Javier Milei, who took office on 10 December 2023

### Sample wikirevision_0121

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: He is the incumbent President of the Argentine Nation, serving as both head of state and head of government

### Sample wikirevision_0121

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple up-to-date sources, including the Wikipedia article on the President of Argentina, which also notes that earlier revisions (as of November 2025) reference his incumbency

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
- **Supporting Docs Found**: d4
- **Claim**: This result supersedes earlier information referencing the 2024 edition, in which Novak Djokovic was the defending champion but lost in the third round

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The current President of Germany is Frank-Walter Steinmeier, who has been in office since 19 March 2017

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: He is the Federal President of the Federal Republic of Germany, serving as the country's head of state, with Bellevue Palace in Berlin as his official residence

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: This is consistent across multiple sources, including the current Wikipedia revision of the President of Germany article, which confirms his incumbency from 19 March 2017

### Sample wikirevision_0125

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The latest Prime Minister of Australia is Anthony Albanese, who has served in office since 23 May 2022

### Sample wikirevision_0125

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: He is the 31st person to hold the position since the office was created in 1901

### Sample wikirevision_0129

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
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
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: The current Prime Minister of Japan is Sanae Takaichi, who assumed office on 21 October 2025

### Sample wikirevision_0132

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: She is the first female prime minister in Japanese history and serves as the country's head of government

### Sample wikirevision_0132

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of the Prime Minister of Japan page, as well as the list of prime ministers of Japan

### Sample wikirevision_0134

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The current Prime Minister of Australia is Anthony Albanese, who has served in office since 23 May 2022

### Sample wikirevision_0134

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: He is the Honourable Mr. Anthony Albanese, having served as the 31st Prime Minister of Australia

### Sample wikirevision_0135

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
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
- **Supporting Docs Found**: d2, d3
- **Claim**: As a result, Calcutta is no longer the official name of the city, which is now officially called Kolkata

### Sample wikirevision_0141

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The latest Wimbledon men's singles champion is Jannik Sinner, who won the 2025 Wimbledon Championships

### Sample wikirevision_0141

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by the official Wikipedia revision that superseded the older version in October 2025, which explicitly names Sinner as the current men's singles champion

### Sample wikirevision_0141

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The newer revision further corroborates this, listing Sinner again as the current champion and noting that the 2026 Wimbledon Championships are scheduled for June–July 2026

### Sample wikirevision_0142

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: The latest Vice President of the United States is JD Vance, who assumed office on January 20, 2025

### Sample wikirevision_0142

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: This is confirmed across multiple authoritative sources, including the newer Wikipedia revision and the list of vice presidents of the United States

### Sample wikirevision_0142

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The Office of the Vice President further corroborates his tenure, noting that it is headed by the chief of staff to Vice President JD Vance

### Sample wikirevision_0145

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
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
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: The latest President of the Philippines is Bongbong Marcos, who assumed office on June 30, 2022

### Sample wikirevision_0149

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: He is the 17th and current President, succeeding Rodrigo Duterte

### Sample wikirevision_0149

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: This is consistent across multiple high-credibility sources, including the Wikipedia article on the President of the Philippines, which also notes that the incumbent President is Bongbong Marcos with his assumption of office on June 30, 2022

### Sample wikirevision_0150

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The latest US Open men's singles champion is Carlos Alcaraz, who defeated Jannik Sinner to win the 2025 title

### Sample wikirevision_0150

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The 2025 US Open was the 145th edition of the tournament, held at the USTA Billie Jean King National Tennis Center in New York City this victory marked Alcaraz's first Grand Slam singles title

### Sample wikirevision_0151

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Australia is the current ICC Men's Cricket World Cup champion, having defeated India in the 2023 final to claim their sixth title

### Sample wikirevision_0151

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: This result is corroborated across multiple sources, with the 2023 edition being the most recent completed tournament

### Sample wikirevision_0151

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The next scheduled tournament, the 2027 ICC Men's Cricket World Cup, is set to be held in South Africa, Zimbabwe Namibia

### Sample wikirevision_0152

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The latest Ballon d'Or winner is Ousmane Dembélé, who claimed his first award at the 2025 ceremony

### Sample wikirevision_0152

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The 2025 Ballon d'Or was the 69th annual ceremony, recognizing the best footballers of the 2024–25 season and taking place on 22 September 2025

### Sample wikirevision_0152

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: While d4 references the 2024 Ballon d'Or d3 and d1 describe the 2025 ceremony without naming the winner, the most current information available confirms Dembélé as the 2025 recipient

### Sample wikirevision_0153

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The latest President of Germany is Frank-Walter Steinmeier, who has been in office since 19 March 2017

### Sample wikirevision_0153

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: He is the Federal President of the Federal Republic of Germany, serving as the country's head of state, with Bellevue Palace in Berlin as his official residence

### Sample wikirevision_0153

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: This is consistent across multiple sources, including the current Wikipedia revision of the President of Germany article, which confirms his incumbency from 19 March 2017

### Sample wikirevision_0154

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: The latest President of Mexico is Claudia Sheinbaum Pardo, who took office on 1 October 2024, becoming the 66th President of Mexico

### Sample wikirevision_0154

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: She is the first woman and the first Jewish person to hold the office, serving as President until 2030

### Sample wikirevision_0154

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: This is consistent across multiple sources, including the high-credibility Wikipedia article on the President of Mexico, which also confirms her incumbency from 1 October 2024

### Sample wikirevision_0155

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: Facebook's parent company is Meta Platforms, Inc. This was confirmed when Facebook rebranded itself as Meta Platforms, Inc. in 2021 to reflect a strategic shift toward developing the metaverse

### Sample wikirevision_0156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: The current President of the Philippines is Bongbong Marcos, who assumed office on June 30, 2022

### Sample wikirevision_0156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: He is the 17th President of the Philippines and serves as both head of state and head of government

### Sample wikirevision_0156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of the President of the Philippines page, as well as the list of presidents of the Philippines

### Sample wikirevision_0157

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The current President of India is Droupadi Murmu, who has held the office since 24 July 2022

### Sample wikirevision_0157

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: She is the 15th President of India and succeeded Ram Nath Kovind

### Sample wikirevision_0157

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple sources, including the newer Wikipedia revision of the President of India article

### Sample wikirevision_0160

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: The current President of Indonesia is Prabowo Subianto, who took office on 20 October 2024

### Sample wikirevision_0160

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3
- **Claim**: He is the eighth president of Indonesia and serves a five-year term

### Sample wikirevision_0160

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of the President of Indonesia page, as well as his own Wikipedia article

### Sample wikirevision_0161

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Gurgaon is officially called Gurugram

### Sample wikirevision_0161

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This change was confirmed by the Haryana Government in 2016 the city is now officially known by this new name

### Sample wikirevision_0161

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: As a result, the older name 'Gurgaon' is gradually being phased out in official contexts, though it is still commonly used

### Sample wikirevision_0162

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d3
- **Claim**: Argentina (defending champion; 2022 FIFA World Cup winner, 3rd title) — the snippet identifies Argentina as the current or defending champion per the 2026 Wikipedia revision

### Sample wikirevision_0165

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: The current President of the United States is Donald Trump, who assumed office on January 20, 2025

### Sample wikirevision_0165

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is confirmed across multiple authoritative sources, including the newer Wikipedia revision of the President of the United States article, which supersedes the older revision from July 2025

### Sample wikirevision_0165

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: The list of presidents of the United States also corroborates that Donald Trump is the incumbent, noting this is his second non-consecutive presidency

### Sample wikirevision_0166

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The current Prime Minister of India is Narendra Modi, who has served in office since 26 May 2014

### Sample wikirevision_0166

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: He is the Honourable Mr. Prime Minister and holds the highest office of the Government of India, being appointed by the President and responsible to the Lok Sabha

### Sample wikirevision_0167

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: The current President of Mexico is Claudia Sheinbaum Pardo, who took office on 1 October 2024, making her the 66th President of Mexico

### Sample wikirevision_0167

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: She is the first woman and the first Jewish person to hold the office, serving as both head of state and head of government

### Sample wikirevision_0167

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple sources, including the older and newer Wikipedia revisions of the President of Mexico article, as well as her own Wikipedia biography

### Sample wikirevision_0170

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The current French Open men's singles champion is Carlos Alcaraz, who defeated Jannik Sinner in the 2025 final to claim his second title

### Sample wikirevision_0170

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: This result is corroborated across multiple sources, with the 2026 Wikipedia revision also listing Alcaraz as the current singles champion

### Sample wikirevision_0170

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: However, it is worth noting that the 2026 tournament featured a different outcome, as Alcaraz withdrew before the start of the event due to a wrist injury

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
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: The latest French Open men's singles champion is Carlos Alcaraz, who defeated Jannik Sinner in the 2025 final to win his second title

### Sample wikirevision_0172

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: This result is corroborated across multiple sources, including the Wikipedia pages for both the 2025 French Open and the general French Open article, which list Alcaraz as the current men's singles champion

### Sample wikirevision_0172

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: While the 2026 French Open was subsequently held, Alcaraz withdrew due to a wrist injury and no new champion is explicitly named in the available evidence


================================================================================

*Report generated by CATS v2.0*
