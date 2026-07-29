# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**GR Accuracy**: 0.694 (over 49 samples)

**GR F1** *(used in CATS)*: 0.819

**Behavior Adherence**: 0.469 (over 49 applicable samples)

**Factual Grounding**: 0.020 (over 49 applicable samples)

**Single-Truth Recall**: 0.735 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.511

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.819
- **Precision**: 0.694
- **Recall**: 1.000
- **Accuracy**: 0.694
- TP=34, FP=15, FN=0, TN=0


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19
- **GR Accuracy**: 0.632
- **GR F1** *(used in CATS)*: 0.774
- **Behavior**: 0.526 (n=19)
- **Grounding**: 0.000 (n=19)
- **Recall**: 0.875 (n=12)
- **CATS**: 0.544

### Type 2: Complementary Info

- **Samples**: 15
- **GR Accuracy**: 0.533
- **GR F1** *(used in CATS)*: 0.696
- **Behavior**: 0.533 (n=15)
- **Grounding**: 0.067 (n=15)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.432

### Type 3: Conflicting Opinions

- **Samples**: 10
- **GR Accuracy**: 0.900
- **GR F1** *(used in CATS)*: 0.947
- **Behavior**: 0.100 (n=10)
- **Grounding**: 0.000 (n=10)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.349

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.800 (n=5)
- **Grounding**: 0.000 (n=5)
- **Recall**: 0.400 (n=5)
- **CATS**: 0.550


================================================================================

## Cost Summary

**Total Cost**: $0.0713

**Decisions Made**: 160

**Average Cost per Decision**: $0.000446


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 160
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0714
- **Total Requests**: 160
- **Average Cost per Request**: $0.000447


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0015

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d2, d9, d3
- **Claim**: **Answer:** The most recently confirmed winner (Season 22) is **Abi Carter**

### Sample #0015

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d4, d6
- **Claim**: The winner of Season 23 (if this year refers to 2025) is not yet mentioned in the retrieved documents

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d5, d2, d8, d4, d3, d6
- **Claim**: **Conclusion**: Television can support language development in preschoolers when paired with adult guidance and high-quality, interactive content

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d10, d2, d12
- **Claim**: However, it is no substitute for direct social interactions, which remain the most effective foundation for language acquisition

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d10, d2, d9, d12, d11, d6
- **Claim**: For younger children, minimizing passive screen time and prioritizing conversational engagement is key

### Sample #0042

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d10, d2, d8, d4, d9, d3, d6
- **Claim**: The song "Gone, Gone, Gone She Been Gone So Long" is performed by **Chilliwack**

### Sample #0042

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d9
- **Claim**: It is part of their 1981 hit "My Girl (Gone, Gone, Gone)", as detailed in the retrieved documents

### Sample #0042

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d9, d2, d6
- **Claim**: The lyrics and background of the song are consistently attributed to the Canadian band Chilliwack across multiple sources, including Wikipedia and Spotify

### Sample #0102

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d5, d2, d8, d4, d9, d3, d6
- **Claim**: The PIN code system in India was introduced on **15th August 1972**

### Sample #0102

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d2, d4, d9
- **Claim**: This system was implemented to streamline the sorting and delivery of mail across the country's vast and complex postal network

### Sample #0102

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d6
- **Claim**: The initiative was led by **Shriram Bhikaji Velankar**, an additional secretary in the Union Ministry of Communications and a senior member of the Posts and Telegraphs Board

### Sample #0102

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: The six-digit PIN code was designed to address challenges like duplicate place names, language barriers the sheer volume of mail handled by India's postal service

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d5, d10, d2, d8, d4, d9, d3, d6
- **Claim**: The most recent woman to become President of Peru is **Dina Boluarte**

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d5, d10, d2, d8, d4, d9, d3, d6
- **Claim**: She assumed office on December 7, 2022, following the impeachment of her predecessor, Pedro Castillo, making her the first and, to date, only female president in Peru's history

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d5, d10, d2, d8, d4, d9, d3, d6
- **Claim**: This is consistently documented across multiple sources, including her Wikipedia page , BBC , CNN The New York Times , which all confirm her historic role as Peru's first female leader

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Her presidency continues as of the latest references in 2024

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d5, d8, d4, d9, d3
- **Claim**: While older documents (e.g., from late 2023 or 2024) mention Novak Djokovic or Rafael Nadal, the most up-to-date rankings consistently name **Jannik Sinner** as the ATP world No. 1

### Sample #0127

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The company's first arrival in India occurred later, in **1608**, when it established a trading post at Surat

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d2, d4, d6
- **Claim**: **Conclusion**: While stress is not the main cause of stomach ulcers, it can indirectly contribute to their development or worsen symptoms, particularly when combined with other risk factors like H. pylori infection or NSAID use

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Stress-related ulcers are a separate category linked to severe physical stress

### Sample #0139

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Since the user's instruction is to use only the retrieved

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d2, d8, d4, d3, d6
- **Claim**: **Conclusion**: Heated gemstones are typically less valuable than unheated ones, particularly in high-end markets where natural rarity is prized

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d5, d10, d8, d2, d4, d9, d11
- **Claim**: However, heating is a common and accepted practice in many gemstones its impact on value varies by type, treatment method consumer demand

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d8
- **Claim**: Always verify treatment history through reputable certifications to assess true value

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3, d2, d5
- **Claim**: ### **Conclusion**
The evidence suggests **some forms of innate knowledge exist**, particularly in foundational cognitive abilities like numerical perception, object permanence possibly language structures

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d8, d6
- **Claim**: However, philosophers like Locke and modern empiricists argue that even these abilities are shaped by experience

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3, d2
- **Claim**: The debate remains unresolved, with contemporary research (e.g., on infants and AI) continuing to explore the interplay between biology and environment in knowledge formation

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d8, d4, d9, d3
- **Claim**: ### Summary:
- **18 is the standard legal age** in most states, but laws vary

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d8, d4, d9, d3, d6
- **Claim**: Always check specific state regulations, as some still allow exceptions for minors

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Recent advocacy efforts have led to stricter laws in several states to end child marriage

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d5, d10, d8, d4, d9, d11, d3, d6
- **Claim**: **Conclusion**: While not *absolutely* required for every startup—especially small or bootstrapped ventures—a business plan (or lean equivalent) is highly advisable for most

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d5, d10, d4, d9, d11, d3, d6
- **Claim**: It is essential for securing funding, clarifying strategy navigating challenges

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d5, d10, d8, d4, d9, d11, d3, d6
- **Claim**: Startups aiming for scalability, investor backing operating in competitive markets should prioritize crafting a detailed plan

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d4, d9, d11, d3
- **Claim**: For others, a simplified, flexible approach may suffice, but even these benefit from some form of strategic framework

### Sample #0203

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d10, d2, d4, d9, d3, d6
- **Claim**: Some documents (e.g., d3, d5) reference other entities named "Channel 5" (e.g., a Cleveland-based TV station or unrelated anniversaries), but the UK terrestrial channel's start date is consistently cited as **30 March 1997**

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d5, d10, d8, d4, d9, d11, d3, d6
- **Claim**: **Recommendation**: Focus on reducing prolonged sedentary behavior (e.g., taking hourly movement breaks) while prioritizing smoking cessation as a higher-impact health intervention

### Sample #0229

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d2, d4, d3, d6
- **Claim**: In summary, smoking is a modifiable risk factor strongly linked to RA onset and progression, underscoring the importance of smoking cessation, especially for those with a family history of RA

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d5, d2, d8, d4, d9, d3, d6
- **Claim**: **Conclusion**: While driving is typically faster in most cities, public transportation can match or exceed driving speeds in areas with dedicated infrastructure (e.g., BRT, subways)

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d8, d4, d9, d3, d6
- **Claim**: Systemic improvements are needed to make transit a competitive option

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d6, d9, d5
- **Claim**: **Conclusion**: Dogs do not comprehend human language in its entirety

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d8, d4, d3
- **Claim**: Their understanding is rooted in associative learning, emotional cues contextual signals

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d5, d8, d4, d9, d3, d6
- **Claim**: While they can grasp specific words and intentions, their communication with humans remains a simplified, learned system rather than true linguistic understanding

### Sample #0301

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d5, d8, d4, d3, d6
- **Claim**: For context, the 2023 winner was *Dorfromantik: The Board Game*, but the latest (2024) winner is clearly documented in the provided sources ()

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d6
- **Claim**: While all octopuses are venomous, the majority of species have venom that is either weak or not dangerous to humans

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d4, d1, d6
- **Claim**: Only a small number, such as the blue-ringed octopus, have venom potent enough to be deadly

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d13, d14, d1, d5, d10, d4, d9, d12, d11, d3
- **Claim**: In summary, the scientific consensus is that melting sea ice does not directly cause significant sea level rise, but land ice melt and thermal expansion are the dominant factors

### Sample #0324

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d5, d8, d2, d6
- **Claim**: The latest Nobel Prize in Literature, awarded in 2024, was won by **Han Kang**, a South Korean author

### Sample #0324

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d5, d8, d2
- **Claim**: She was recognized "for her intense poetic prose that confronts historical traumas and exposes the fragility of human life." This information is confirmed across multiple sources, including the official Nobel Prize website () and Wikipedia

### Sample #0324

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Han Kang is the first South Korean to receive this honor

### Sample #0333

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3
- **Claim**: The last person appointed to the U.S. Supreme Court is **Ketanji Brown Jackson**

### Sample #0333

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: She was nominated by President Joe Biden on February 25, 2022 confirmed by the Senate on April 7, 2022

### Sample #0333

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d5
- **Claim**: She took her seat on June 30, 2022, succeeding Justice Stephen Breyer, who retired

### Sample #0333

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3
- **Claim**: This information is confirmed in documents **d3**, **d4** **d5**

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d5, d2, d8, d4, d9
- **Claim**: While some sources (e.g., Goodreads) list additional titles, most authoritative references (e.g., her official website, Wikipedia academic profiles) confirm these three as her primary published books

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6
- **Claim**: The discrepancy may arise from including edited works, articles different editions

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5
- **Claim**: Thus, the original version of Spider-Man relied on mechanical web-shooters, not organic ones

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d5, d2, d4, d9, d6
- **Claim**: The documents do not list all 56 names in their snippets, but they confirm the total number, the signing date key figures involved

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d9, d6
- **Claim**: For a full list of signers, additional resources would be required

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The discrepancy between the 2025 figure (8.2 billion) and earlier 2022 estimates (8 billion) reflects ongoing population growth, averaging roughly **80-90 million new people per year**

### Sample #0392

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d9, d5
- **Claim**: The United States (USA) is currently ranked as the top nation in the FIBA Men's World Ranking

### Sample #0392

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: Timestamped 2024-11-29), the USA has solidified its position as the number one ranked team, followed by Serbia in second place and Germany in third

### Sample #0392

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: This ranking reflects updates after the FIBA Continental Cup Qualifiers and confirms the USA's continued dominance at the top of the standings

### Sample #0394

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5
- **Claim**: The host stadium for this year's Super Bowl (Super Bowl LVIII) is **Allegiant Stadium** in Las Vegas, Nevada

### Sample #0394

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5
- **Claim**: The game was played on February 11, 2024, as confirmed by the documents

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d5, d2, d4, d6
- **Claim**: **Conclusion**: A vegan diet can be safe during pregnancy if meticulously planned with supplements and professional oversight

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d5, d2, d4, d3, d6
- **Claim**: However, it is not inherently suitable without addressing potential nutrient gaps

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d5, d2, d6
- **Claim**: Pregnant women following a vegan diet should work closely with healthcare providers to mitigate risks

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d5, d8, d4, d9, d3
- **Claim**: Other countries produce sparkling wines using different methods and names (e.g., Prosecco from Italy, Cava from Spain "sparkling wine" from other regions), but these cannot be labeled as "champagne" outside of France

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d10, d8
- **Claim**: Legal frameworks, including the Treaty of Versailles and the Madrid System, ensure this exclusivity

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d5, d2, d8, d4, d9, d3, d6
- **Claim**: As noted in the documents, even if a sparkling wine is made using similar techniques, it must originate in the Champagne region to bear the name "champagne."

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d5, d2, d8, d4, d9, d6
- **Claim**: Thus, while the 1851 covenant established it as a commandment in practice, its formal enforcement as a requirement for temple and priesthood privileges solidified its mandatory status in the early 20th century

### Sample #0470

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d5
- **Claim**: In summary, the documents validate club soda’s use for stain removal through user experiences and mentions of its application, but they do not delve into the chemical or physical reasons for its effectiveness

### Sample #0499

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Other documents highlight the inherent absorbency of cotton versus microfiber (e.g., **d5**) but do not directly address the effect of washing on absorbency

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The key insight comes from **d3**, which ties increased absorbency specifically to the washing process for high-quality cotton towels

### Sample #0509

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d2, d4, d3
- **Claim**: However, these details are outdated

### Sample #0509

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: To find the current schedule, please check the official *America's Got Talent* website, NBC's schedule a trusted TV guide for the most up-to-date air times

### Sample #0517

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: In summary, the brain registers most

### Sample #0531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: In summary, the documents suggest that socialism and communism are related but distinct concepts within Marxist theory, with socialism acting as a stepping stone to communism

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, they do not provide a detailed comparison of economic systems, governance other key aspects typically used to differentiate the two ideologies

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: For a thorough answer, additional sources would be required

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The retrieved documents do not explicitly state who wrote the jazz classic "I Got Rhythm." While the song is mentioned in document **d4** (related to a 1930 recording), the authorship is not addressed in the provided snippets

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Additional sources would be required to confirm the composer(s)

### Sample #0561

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The television series featuring Jamie Oliver as a cast member is **"Jamie's School Dinners"**

### Sample #0561

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: This four-episode documentary series, broadcast on Channel 4 in the UK in 2005, showcases Jamie Oliver's efforts to improve the quality and nutrition of school meals at Kidbrooke School in Greenwich

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3
- **Claim**: In summary, night vision is green because it optimizes visibility in low light by leveraging the human eye's sensitivity to green wavelengths, the efficiency of phosphor materials in amplifying light practical considerations like contrast and energy use

### Sample #0592

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Kenya and Jamaica each won **10 gold medals** apiece at the 2014 Commonwealth Games, tying for the most gold medals in the event

### Sample #0592

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Australia secured 8 gold medals, while England led in total medals (27) but with 13 of those being silver

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d2, d4, d3
- **Claim**: In summary, in-play odds are a blend of real-time probability analysis, balancing bets profit-margin maintenance

### Sample #0609

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Bookmakers rely on algorithms, expert insights betting trends to adjust odds swiftly during events

### Sample #0638

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: The retrieved documents do not explicitly state where Dorothy lived in *The Wizard of Oz*

### Sample #0638

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: While they describe her journey through the Land of Oz, her desire to return home various inspirations for the story (e.g., Peekskill, New York California in later adaptations), none directly mention her home location (traditionally Kansas in the original novel)

### Sample #0638

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: The documents focus on her adventures in Oz rather than her origin

### Sample #0644

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Additional titles like *The Nose From Jupiter* and others referenced in the snippets (e.g., *A Nose For Adventure*, *The Way to Schenectady*) are attributed to different authors (e.g., Scrimger) in the documents and are not linked to Richler here

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d3, d2
- **Claim**: Other documents (e.g., **d2**, **d4**) corroborate that salt-based systems are a primary method for water softening, contrasting them with alternatives like reverse osmosis or magnetic systems

### Sample #0654

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: While some documents discuss advanced or metaphorical concepts (e.g., "data gravity" in d2 or theoretical models in d1/d4), the core definition remains rooted in the mutual attraction between masses


================================================================================

*Report generated by CATS v2.0*
