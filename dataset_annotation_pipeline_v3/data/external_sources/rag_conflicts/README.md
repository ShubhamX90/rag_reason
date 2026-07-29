

# CONFLICTS

This repo includes the dataset described in the paper: [DRAGged into CONFLICTS: Detecting and Addressing Conflicting Sources in Search-Augmented LLMs]()


Each instance includes the fields:

```json
{
  "source": "", 
  "question": "",
  "search_results": [
    {
      "title": "",
      "url": "",
      "snippet": "",
      "date": "",
      "response_str": "", // full text from the webpage
      "short_text": "",  // 512-token window
    }
  ],
  "conflict_type": "", 
  "correct_answer": "" 
}
```

# Citation

```
@article{cattan2025dragged,
  title={DRAGged into CONFLICTS: Detecting and Addressing Conflicting Sources in Search-Augmented LLMs},
  author={Arie Cattan and Alon Jacovi and Ori Ram and Jonathan Herzig and Roee Aharoni and Sasha Goldshtein and Eran Ofek and Idan Szpektor and Avi Caciularu},
  journal={ArXiv},
  year={2025},
  volume={abs/2506.08500},
}
```


# Disclaimer

This is not an official Google's product. 
