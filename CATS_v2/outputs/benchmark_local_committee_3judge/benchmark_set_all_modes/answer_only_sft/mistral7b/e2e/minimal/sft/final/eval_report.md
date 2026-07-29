# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 128 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.999 (over 736 samples)

**GR F1** *(used in CATS)*: 0.999

**Behavior Adherence**: 0.735 (over 608 applicable samples)

**Factual Grounding**: 0.826 (over 608 applicable samples)

**Single-Truth Recall**: 0.668 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.807

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.999
- **Precision**: 1.000
- **Recall**: 0.998
- **Accuracy**: 0.999
- TP=607, FP=0, FN=1, TN=128

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.992
- **Abstain Recall**: 1.000
- **Abstain F1**: 0.996
- **Specificity**: 0.998
- Abstain TP=128, FP=1, FN=0, TN=607


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (57 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.995
- **GR F1** *(used in CATS)*: 0.997
- **Behavior**: 0.929 (n=154)
- **Grounding**: 0.854 (n=154)
- **Recall**: 0.808 (n=154)
- **CATS**: 0.897

### Type 2: Complementary Info

- **Samples**: 221 (45 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.852 (n=176)
- **Grounding**: 0.820 (n=176)
- **Recall**: 0.526 (n=156)
- **CATS**: 0.799

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.344 (n=96)
- **Grounding**: 0.869 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.737

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.669 (n=145)
- **Grounding**: 0.822 (n=145)
- **Recall**: 0.686 (n=140)
- **CATS**: 0.794

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.649 (n=37)
- **Grounding**: 0.640 (n=37)
- **Recall**: 0.622 (n=37)
- **CATS**: 0.727


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 1989

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

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1
- **Claim**: No, weight lifting does not cause high blood pressure

### Sample conflictingqa_0a05aabca56a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: d3, d5
- **Claim**: Yes, anime is a form of cartoon — it is animation produced in Japan, though it originated in the United States

### Sample conflictingqa_287debe5bb90

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: No, we cannot know anything beyond our minds

### Sample conflictingqa_2c0ea18839df

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: d2
- **Claim**: These rare, specific marks are the exception rather than the rule researchers continue to investigate the mechanisms by which epigenetic information can be transmitted across generations

### Sample conflictingqa_3601f7480501

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3, d1
- **Supporting Docs Found**: None
- **Claim**: Overall, the evidence suggests that unlimited vacation time can be beneficial for employees in specific contexts — for example, knowledge workers in creative fields where output rather than hours on the clock drives performance — but that a fixed-vacation-day policy may work better for most organizations

### Sample conflictingqa_37ebad668bb7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: d1
- **Claim**: The amount of data needed depends heavily on the complexity of the problem, the type of model being used the degree of tolerance for error

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d1, d5
- **Claim**: However, it is worth noting that some sources suggest the carbon savings are only significant if the real tree is used for at most 10 years that artificial trees are not inherently non-recyclable if properly disposed of

### Sample conflictingqa_4786f87b62be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: On the other side, progressive and feminist perspectives argue that the gap is the result of sexist discrimination and that women's choices are shaped by structural barriers and societal norms

### Sample conflictingqa_52181cd092aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3, d5
- **Claim**: Yes, the Great Pacific Garbage Patch is larger than Texas

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Yes, patents do apply to software. The U.S. Patent Office issues thousands of software-related patents every year the Supreme Court's 2014 decision in Alice v

### Sample conflictingqa_52e01830d2fe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: However, not all software is patentable: the U.S. Patent Office's own guidelines exclude 'computer programs as such' from patentability the Supreme Court's 2020 decision in Athena v

### Sample conflictingqa_52e01830d2fe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Mayo further narrowed the scope of patentable subject matter

### Sample conflictingqa_6988dd820a61

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3, d1
- **Claim**: The retrieved evidence presents conflicting views. Some sources argue that major earthquakes are more likely during full and new moons, while others argue that the evidence does not support this conclusion

### Sample conflictingqa_734859937e46

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: This means that ISPs can track and sell data like browsing history and location, but they must remove personally identifying information (PII) — such as names, addresses phone numbers — before sharing it with third parties

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d5
- **Claim**: The retrieved evidence presents competing views. Some sources argue that the Catholic Church is the true church because it claims to be the one, holy, catholic apostolic Church established by Jesus Christ, while others argue that the Catholic Church is not clearly mentioned in the Bible and that other denominations also claim to be the true church, making the question of which is the 'one true church' a matter of genuine theological debate

### Sample conflictingqa_894f4a4b9552

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1, d5
- **Supporting Docs Found**: d4
- **Claim**: Unlike normal matter, dark matter does not interact with the electromagnetic force, making it difficult to detect directly, but researchers have been able to infer its existence from the gravitational effect it exerts on visible matter

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Research using more than 100,000 audio recordings from around the world found that bird species' habitat and geography influence the frequency of the sounds they make larger birds tend to produce lower frequency sounds than smaller ones , further adding to the diversity of bird vocalizations

### Sample conflictingqa_8cf8ebb94554

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Yes

### Sample conflictingqa_9275911a2961

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: Calcium kidney stones are the most common type of kidney stone research has shown that antacids containing calcium can cause these stones to form if too much calcium builds up in the kidneys

### Sample conflictingqa_a507c5b61631

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3, d1, d5
- **Claim**: Yes; volcanic activity—specifically, elevated levels of mercury relative to organic carbon in North Sea sediments—has been identified as a direct proxy for volcanism multiple studies confirm its role as the dominant carbon source driving the Paleocene-Eocene Thermal Maximum (PETM)

### Sample conflictingqa_c418fecfc1e2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5
- **Claim**: Yes, plants can grow without sunlight, though not all plants are equally suited to low light conditions. Some plants, such as Chinese evergreen (Aglaonema), cast iron plant (Aspidistra elatior), ZZ plant (Zamioculcas), monstera (Monstera deliciosa) lucky bamboo (Dracaena sanderiana), are particularly good at growing in low light or even artificial light, while others like algae, mushrooms yeast can use electricity-driven 'dark photosynthesis' to generate food

### Sample conflictingqa_dc6f972e8447

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Additionally, exchanges themselves can engage in manipulation — for example, Bitfinex has been accused of liquidation hunting, where manipulators push prices just enough to trigger margin calls and liquidations, magnifying losses for leveraged traders

### Sample conflictingqa_dd426f7706e8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The retrieved evidence is divided. Some sources argue that werewolves can transform during a full moon, while others argue that full moon transformations are largely a product of cinematic storytelling and not rooted in ancient myths

### Sample conflictingqa_e60bdcf75ba7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d54
- **Supporting Docs Found**: d4
- **Claim**: However, some research presents a more nuanced picture: a comparative analysis found organic yields lower than conventional for row crops and fruit crops but higher for vegetables a long-term study found that organic yields can match conventional yields with appropriate management practices , suggesting that the yield gap is not inherent to organic farming itself but can be reduced with better farming techniques and crop selection

### Sample conflictingqa_f22b389be1d6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4, d5
- **Claim**: The retrieved evidence presents a genuine barefoot running debate. Some sources argue that running barefoot is healthier than running with shoes, while others argue that shoes provide arch support and cushioning that barefoot running does not, making shoes the healthier option

### Sample conflictingqa_f3b163170581

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Yes, humans did evolve from apes — specifically from a common ancestor with chimpanzees and other primates

### Sample conflictingqa_f4811561af0c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The Dutch were among the first to explore and map Australia, but they cannot be said to have 'discovered' it in the traditional sense

### Sample freshqa_0436c0b3a9d7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Aryna Sabalenka (6-3, 7-6(3))

### Sample freshqa_047057d22309

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d3, d5
- **Supporting Docs Found**: d2
- **Claim**: Harry's dukedom was never formally revoked — and that the title 'Duke of Sussex' is still his, though he no longer holds the HRH prefix

### Sample freshqa_122352ad92e3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d4
- **Claim**: In 2026, the first Seder is held on the evening of April 1 the second Seder is held on April 2

### Sample freshqa_150b9ed24a07

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Hillary Clinton enacted at least 1 executive order during her tenure as Secretary of State, though the specific number is not explicitly stated in the available evidence

### Sample freshqa_1ef881d26e2e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Maryam Mirzakhani (1977–2017), an Iranian mathematician who became the first woman and the first Iranian to be awarded a Fields Medal in 2014

### Sample freshqa_263eca8e024e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: This feat, with Box Office India reporting ₹1,810 crore the Times of India noting ₹1,750 crore

### Sample freshqa_31ad09b9cd22

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Samara Joy won the latest Grammy Award for Best Jazz Performance, taking home the award for "Twinkle Twinkle Little Me" featuring Sullivan Fortner at the 67th Annual Grammy Awards in 2025

### Sample freshqa_3227ea6c6056

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The latest major version of the .NET Framework is 4.8.1, released on August 9, 2022

### Sample freshqa_3227ea6c6056

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by the official Microsoft .NET Framework download page, which lists 4.8.1 as the latest release in the .NET Framework 4.x series

### Sample freshqa_354e7097602d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d4
- **Claim**: The U.S. Army conducted the test at approximately 5:30 a.m., detonating a plutonium-powered implosion device atop a 100-foot steel tower, releasing 18.6 kilotons of energy

### Sample freshqa_42796b35e143

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: 3 seasons — Season 1 premiered November 12, 2019, Season 2 premiered October 30, 2020 Season 3 premiered March 1, 2023

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The Federal Reserve cut interest rates by 50 basis points from August to December 2022, bringing the federal funds rate down to 3.5% — a 1.5 percentage point decline from its September 2024 peak of 4.25%

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: 2023

### Sample freshqa_7dce5d575302

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The last player to win the Ballon d'Or before the Messi–Ronaldo dominance was Luka Modric, who claimed the award in 2018

### Sample freshqa_7e63fcff2dea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Laika

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d4
- **Claim**: George R.R. Martin was born in Bayonne, New Jersey

### Sample freshqa_8f302f0bfe82

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The most recent Nebula Award for Best Novel was won by *The Incandescent* by Emily Tesh, published by Tor/Orbit UK, in 2025

### Sample freshqa_97f3c1fe1fd4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Eminem

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Frank Rosenblatt died in a boating accident on July 28, 1971

### Sample freshqa_a47283064972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1
- **Claim**: No, the Raptors do not have a winning record in the 2023–24 NBA season

### Sample freshqa_a50d0f1f3cdf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d4, d1
- **Claim**: Queen Elizabeth II of England died on 8 September 2022, at Balmoral Castle in Scotland, where she had been staying with her family

### Sample freshqa_a50d0f1f3cdf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d4, d1, d5
- **Claim**: She was 96 years old

### Sample freshqa_a5492f36ca23

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: He died at his home in New York surrounded by his family, with his son Duncan Jones confirming his passing

### Sample freshqa_b99c189f2222

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d4
- **Claim**: Jeff Bezos did not fully sell Amazon; rather, he executed a series of stock sales over time

### Sample freshqa_c3f10dc1632d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Jiangsu and Zhejiang provinces border Shanghai to the north

### Sample freshqa_c479e83e408f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: 8

### Sample freshqa_c4976f8629cb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d2
- **Claim**: The Komodo dragon's status as the heaviest reptile, surpassing all other species — including crocodiles, lizards snakes — in mass despite some being longer in length

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5
- **Claim**: 12

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The NFL announced that the Bills vs. Bengals game on January 2, 2023 resumed play on January 6, 2023

### Sample freshqa_e502143179d6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1, d5
- **Supporting Docs Found**: d4
- **Claim**: Musk's acquisition of Twitter marked the end of Twitter as an independent company and the beginning of X

### Sample freshqa_f6ac249bdf53

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d4, d5
- **Claim**: Ta-Nehisi Coates

### Sample freshqa_f6cc6071caa5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d4
- **Claim**: The number of Nazca Lines geoglyphs discovered so far is approximately 893, according to the most recent count reported by Yamagata University researchers in 2024

### Sample healthcontradict_0060

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d1
- **Claim**: Children with scarlet fever should not be sponged down with tepid water, as the bacteria causing scarlet fever can spread to other areas of the body ; similarly, the American Academy of Family Physicians advises that sponging does not work

### Sample healthcontradict_0060

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: Instead, the NHS recommends giving children paracetamol or ibuprofen to reduce fever the AAFP suggests using a cool (not cold) sponge bath for children over 5

### Sample hotpotqa_0083

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d7
- **Claim**: 2016

### Sample hotpotqa_0155

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: 2011

### Sample hotpotqa_0186

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d10
- **Supporting Docs Found**: None
- **Claim**: Additionally, a cover version of the song was recorded by Girls' Generation in 2007, though that version is a separate composition from the original

### Sample hotpotqa_0192

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d7
- **Supporting Docs Found**: None
- **Claim**: The operation also included more than 1,600 non-scientist personnel, bringing the total to over 3,200 individuals recruited from post-Nazi Germany

### Sample qacc_0b75ed799d46

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1, d5
- **Supporting Docs Found**: d4
- **Claim**: The ceremony streamed live on Netflix at 8:00 p.m. EST / 5:00 p.m. PST, with Kristen Bell hosting for the third time

### Sample qacc_160a528ae07e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2, d4
- **Claim**: Tom Brady has won the NFL MVP award three times, in 2007, 2010 2017, when he was with the New England Patriots

### Sample qacc_19ca08790764

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3, d5
- **Claim**: Oliver Stark plays Buck on 9-1-1

### Sample qacc_1b95727cc286

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: Wood Harris plays Ace/Azie Faison, Mekhi Phifer plays Mitch/Rich Porter Cam'ron plays Rico/Alpo Martinez in the 2002 film

### Sample qacc_252987b8054c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5
- **Claim**: Tori Spelling played Violet Anne Bickerstaff in Saved by the Bell

### Sample qacc_3c1297608017

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d4, d5
- **Claim**: The retrieved evidence suggests that crossing fingers for luck has its roots in pre-Christian pagan beliefs and early Christian practices, though the exact origins remain uncertain

### Sample qacc_41c44ecfd0f0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Peyer's patches

### Sample qacc_5eeb46719843

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: The 1931 Statute of Westminster also allowed Canada to sign international treaties independently, such as the Treaty of Versailles participate in the League of Nations

### Sample qacc_67b35f41ba84

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Alice Kremelberg plays Bill Pullman's wife in Season 4 of _The Sinner_

### Sample qacc_6837d86d03ac

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5
- **Claim**: The next in line to be the monarch of England is King Charles III, who ascended to the throne upon the death of Queen Elizabeth II on September 8, 2022

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: October 1, 1968

### Sample qacc_8dd2323c077d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: A third source, History Adventuring, corroborates that the first McDonald's in Phoenix represents a milestone in the evolution of fast food and the growth of McDonald's as a global brand

### Sample qacc_8ef7b3cf5c3f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: Additionally, Argentina and Uruguay both have large urban middle classes and relatively even income distributions the Southern Cone—the region encompassing these two countries—is characterized by high literacy rates and strong democratic stability

### Sample qacc_946ecfb478b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4, d5
- **Claim**: Justin Timberlake, Max Martin Shellback wrote the song; specifically, Timberlake wrote the lyrics and melody, while Martin and Shellback provided the music and production

### Sample qacc_9b16fd6882f3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5
- **Claim**: "God Gave Rock and Roll to You" is sung by multiple artists, but the song was written by Russ Ballard of Argent in 1971

### Sample qacc_9fbf28f5786f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: The International Space Station (ISS) was launched into space on November 20, 2000, marking the first continuous human presence in orbit since 1987

### Sample qacc_a44267c115d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d4
- **Claim**: This intracellular water is constantly moving across the cell membrane in response to osmotic gradients, with sodium ions as the predominant osmole driving water distribution

### Sample qacc_a4dde83c35da

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: This governmental structure was further marked by the use of severe punishments against perceived threats to imperial power and the abolition of the Mongol-era social segregation system

### Sample qacc_a6df0af8c2ba

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d4, d5
- **Claim**: Hosanna is an expression of praise and a cry for salvation, meaning "save us now" or "help, please." It is most commonly associated with Palm Sunday, when crowds greeted Jesus riding into Jerusalem with the cry, "Hosanna!

### Sample qacc_b0346f60b6ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Celebrity Big Brother is not on any major U.S. channel; it airs on ITV in the UK

### Sample qacc_b198a514fff8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: d5
- **Claim**: This followed decades of territorial status and political maneuvering, including the 1850 creation of the New Mexico Territory and the 1910 drafting of a state constitution

### Sample qacc_bc34664caee4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Rio de Janeiro, Brazil; Puerto Rico; and California's Mojave Desert

### Sample qacc_c264cb69676e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: This joint allows for movement in two planes and is surrounded by a joint capsule filled with synovial fluid, which lubricates the joint and reduces friction

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Carter Pewterschmidt is Lois's father, voiced by Seth MacFarlane

### Sample qacc_d00b0063e747

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: The twins were born on November 1, 2022 are both healthy and happy

### Sample qacc_d60bf850c4ff

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: Cadbury's products are sold in over 50 countries across five continents, though the exact number varies by product and year

### Sample qacc_d7c6682b5335

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: 1996

### Sample qacc_d8b24beb2f90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The balance sheet is the financial statement that involves all aspects of the accounting equation, showing the sum of assets, liabilities equity (Assets = Liabilities + Equity)

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d5
- **Claim**: They are officially called cuotas (fees) drivers pay in Mexican pesos

### Sample qacc_ea469c846404

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d2, d4
- **Claim**: 2022-2023

### Sample qacc_ff2cb00f4c03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d4, d5
- **Claim**: Season 7 of Game of Thrones consists of seven episodes — confirmed by HBO's episode listing, which places all seven episodes on the schedule for the week of July 16, 2017

### Sample qacc_ff2cb00f4c03

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Entertainment Weekly and Vanity Fair, meanwhile, note that Season 7's average episode length is actually longer than usual, with Episode 6 running for 71 minutes and the finale at 81 minutes , making the season's total runtime roughly equivalent to that of a full 10-episode season

### Sample situatedqa_geo_082db791e263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2, d4, d5
- **Claim**: In the United States, you must be at least 18 years of age to purchase a shotgun, though the federal minimum age requirement varies by gun type. The ATF has made it clear that individuals over 18 can own shotguns and rifles, but not pistols, so at a minimum you must be 18 to purchase a shotgun

### Sample situatedqa_geo_09a6a048cfbf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3, d4
- **Supporting Docs Found**: d5
- **Claim**: Elsewhere in the UK, the law is the same: 18 is the minimum age to buy and consume alcohol anyone under 18 is prohibited from purchasing alcohol

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: For larger commercial vehicles, the minimum age to drive is typically 18 years of age in some states, the minimum age to obtain a CDL is 21 years of age

### Sample situatedqa_geo_48e19f6b37bc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Alaska is the 3rd largest state in the United States by area, covering approximately 665,384 square miles (1,723,337 km²)

### Sample situatedqa_geo_48e19f6b37bc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4, d5, d2, d1
- **Supporting Docs Found**: None
- **Claim**: All available evidence consistently confirms this ranking, with no contradictions across sources

### Sample situatedqa_geo_5f5c20228969

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: The retrieved evidence places the number of fronts fought by the Axis in World War II at three: the Eastern Front (also called the Russia Front), the Western Front (also called the France and the Low Countries Front) the Mediterranean Front (also called the Italy Front)

### Sample situatedqa_geo_6cca7fdb9b41

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The official Guinness World Records title for the 'land farthest from the sea' is held by the Eurasian pole of inaccessibility in northwestern China, over 500km from any ocean

### Sample situatedqa_geo_80c72c2797bf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: This is a representative democracy, meaning citizens vote for representatives who then make laws on their behalf

### Sample situatedqa_geo_85af31651715

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This was confirmed by Andy Kerr, Scotland's former health minister, who spearheaded the smoking ban campaign

### Sample situatedqa_geo_864fc7e36714

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: Pew Research estimates that in 2065, those who identify as Asian will make up some 38% of all immigrants, surpassing the 31% share of Hispanics , while the Bush Institute notes that job opportunities and affordable housing are among the factors driving immigrant settlement patterns across the country

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1, d5
- **Supporting Docs Found**: d4
- **Claim**: This deployment was part of the broader U.S. foreign policy during the Cold War, aimed at preventing the spread of communism in Southeast Asia

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5
- **Claim**: The California flag is a bear flag, specifically the California grizzly bear flag. The bear is the official state animal of California the flag itself is the official state flag of California

### Sample situatedqa_geo_f2031e426cee

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: The present Law Minister of India is Shri Kiren Rijiju, who serves as the Minister of State for Law and Justice in the Government of India. He is a senior BJP leader and has held the position since 2020

### Sample situatedqa_geo_f26078ec6467

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d4
- **Claim**: The war was conducted across multiple theaters including Cuba and the Philippines apart from guaranteeing Cuba's independence, the U.S. also acquired Guam, Puerto Rico the Philippines

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5
- **Claim**: August 24, 1814

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: At the international level, the United Nations Framework Convention on Climate Change (UNFCCC) and the Paris Agreement establish a framework for global environmental policy, though the U.S. has been a notable laggard in meeting its commitments

### Sample situatedqa_temp_14f29fe7ff15

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: This victory by the British set the stage for the conquest of Philadelphia, the American capital at the time

### Sample situatedqa_temp_19badef7553b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: 1999

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: 2025

### Sample situatedqa_temp_45cbc4dc3e28

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d5
- **Claim**: The current Chief Justice of the Sindh High Court is Justice Zafar Ahmed Rajput, who became the acting Chief Justice on 06 December 2025. He is the 12th incumbent to serve as Chief Justice of the SHC since its inception in 1993

### Sample situatedqa_temp_4ead2f4cd2d5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: Jordan Ridgeway on Days of Our Lives (Dool) and Bethany Bryant on The Young and the Restless (Y&R)

### Sample situatedqa_temp_5a59faf24972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This version brings Live Updates, lock screen widgets, grouped notifications improved performance for larger-screen devices

### Sample situatedqa_temp_5a59faf24972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d4
- **Claim**: This as the most recent Android release, with d2 and d3 being outdated and superseded by the June 2025 update

### Sample situatedqa_temp_603f0dc417ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: September 18, 2022 (Season 6 of SEAL Team premiered on September 18, 2022 on Paramount+ in the US.)

### Sample situatedqa_temp_61a79d74d827

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d4
- **Claim**: This is confirmed by the official Tour de France 2017 route map, which places the start in Düsseldorf is corroborated by the Wikipedia article on the 2017 edition

### Sample situatedqa_temp_6a8aed3e3d5f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: Goku's Super Saiyan 3 transformation is also confirmed across multiple sources, including the Dragon Ball Fandom wiki and the Mugen database his attainment of the form is further corroborated by additional analysis

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d1, d5
- **Claim**: Todd Monken

### Sample situatedqa_temp_7a7a7f43e575

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: The name's prolific use is further corroborated by the fact that it is the most common city name in the world, appearing in over 1,700 places across 6 continents that it was the first city named in Massachusetts in 1636

### Sample situatedqa_temp_7bb7fe6c9287

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Yes, kennings are commonly used in Beowulf to describe characters and events in a metaphorical and memorable way. Some examples include "whale-road" for the sea, "twilight-spoiler" for Grendel "bone-house" for the human body

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The health minister of India in 2013 is Jagat Prakash Nadda, who served as Minister of Health and Family Welfare in the Government of India. He is a senior BJP leader and has held the position since 2019

### Sample situatedqa_temp_7f07f7a6c607

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3, d1, d5
- **Claim**: Mohamed Salah

### Sample situatedqa_temp_8808c106a115

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1, d5
- **Supporting Docs Found**: None
- **Claim**: This deficiency leads to the abnormal accumulation of GM2-ganglioside in brain and nerve cells, eventually causing the progressive deterioration of the central nervous system

### Sample situatedqa_temp_901be1437bc7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_956e1f6e518e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: d3, d1, d5
- **Claim**: This date, noting that no human has been to the moon since then

### Sample situatedqa_temp_9a91e43d8136

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Rohit Sharma

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d5
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_b3ad3248b7d4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d1
- **Claim**: The retrieved evidence places the Inca Empire's founding/settlement date at 1438, when Pachacuti expanded the Kingdom of Cusco into Tawantinsuyu (the four united regions of the Inca Empire)

### Sample situatedqa_temp_b6c75b99a2f4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: Beyond these long-wavelength visible light waves, the next band is near-infrared, which lies just beyond the human visual spectrum and is typically defined as the range of wavelengths from approximately 750 nm to 1100 nm

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d4, d5
- **Claim**: Rhys Ifans plays Eyeball Paul in Kevin & Perry Go Large

### Sample situatedqa_temp_dc4d0c4e24ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Riyad Mahrez won the PFA Player of the Year award for 2015–16, beating team-mates Jamie Vardy and N'Golo Kante, Tottenham's Harry Kane, West Ham's Dimitri Payet and Arsenal's Mesut Ozil

### Sample situatedqa_temp_f099d9d1452b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: Multiple authoritative sources, including the official CIA website and the White House, confirm his tenure has continued through at least May 2025

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: 1982–83

### Sample wikirevision_0047

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3, d1
- **Supporting Docs Found**: None
- **Claim**: Argentina's victory gave the country its third title and the first for any non-European nation since 2002

### Sample wikirevision_0049

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: The current Indian Premier League champion is Chennai Super Kings, who defeated Gujarat Titans by five wickets (DLS method) in the 2023 final to win their fifth league title. This result is consistently confirmed across multiple sources, with Chennai Super Kings' victory confirmed across the 2023 season, making them the current champions

### Sample wikirevision_0067

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: This win was confirmed across multiple sources, with Dembélé beating the likes of Lionel Messi and Cristiano Ronaldo to claim the award

### Sample wikirevision_0071

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Benjamin Netanyahu is the current Prime Minister of Israel, having assumed office on 29 December 2022. This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of Prime Minister of Israel, the list of Israeli prime ministers the older and newer Wikipedia revisions of Alternate Prime Minister of Israel

### Sample wikirevision_0096

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current Chief Justice of India is Surya Kant, who became the incumbent on 24 November 2025. He is the 53rd Chief Justice to serve since the Supreme Court of India was established in 1950

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: This victory gave Australia their sixth title, the most successful record held by any team in the history of the tournament

### Sample wikirevision_0109

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The current Prime Minister of Canada is Mark Carney, who assumed office on 14 March 2025. He is the 24th and current Prime Minister, serving as the official head of government of Canada. This is consistent across multiple sources, including the older and newer Wikipedia revisions of Prime Minister of Canada, as well as the list of Canadian prime ministers

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current President of Germany is Frank-Walter Steinmeier, who has been in office since 19 March 2017. He is the Federal President of the Federal Republic of Germany, serving as the country's head of state, with Bellevue Palace in Berlin as his official residence. This is consistent across multiple sources, including the current Wikipedia revision of the President of Germany article, which confirms his incumbency from 19 March 2017

### Sample wikirevision_0134

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current Prime Minister of Australia is Anthony Albanese, who became incumbent on 23 May 2022. He is the 31st person to serve in the role since the office was created in 1901. This is consistent across multiple sources, including the official Australian Government website and Wikipedia's list of Australian prime ministers

### Sample wikirevision_0135

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: The current Wimbledon men's singles champion is Jannik Sinner, who defeated Julian Cash in the 2026 final on 12 July 2026, claiming his first Wimbledon title and third major overall. The 2026 Wimbledon Championships were the 139th edition of the Wimbledon Championships, held at the All England Lawn Tennis and Croquet Club in Wimbledon, London, England from 29 June to 12 July 2026 — the first time video reviews were used in Wimbledon history

### Sample wikirevision_0135

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This victory also marked Sinner's third major overall, completing the career Grand Slam

### Sample wikirevision_0141

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Sinner's victory gave him his first major title and the 2026 Wimbledon Championships are scheduled to take place from 29 June to 12 July 2026, marking the first time video reviews will be used in the tournament's history

### Sample wikirevision_0157

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The current President of India is Droupadi Murmu, who became the incumbent on 25 July 2022. She is the 15th President of India since the post was established in 1950 and serves as the head of state and supreme commander of the Indian Armed Forces. This is consistent across multiple sources, including the official Government of India website and Wikipedia's list of presidents of India

### Sample wikirevision_0162

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: Argentina's victory gave the country its third title, 34 years after its last one in 1986

### Sample wikirevision_0172

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: The latest French Open men's singles champion is Carlos Alcaraz, who defeated world No. 1 Jannik Sinner in the 2025 final to win his second French Open title and fifth major. The 2025 French Open was the 124th edition of the tournament, held at the Stade Roland Garros in Paris, France, from 25 May to 8 June 2025, with Carlos Alcaraz defending his title from the 2024 champion Jannik Sinner

### Sample wikirevision_0172

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d4
- **Claim**: 2025 is the most recent French Open held Alcaraz's victory there makes him the latest champion


================================================================================

*Report generated by CATS v2.0*
