
Welcome to the LLM Challenge/Contest @ Algorithms Summit 2023.
This repository represents a starter kit for participants to get started with the challenge.
This challenge explores one aspect of LLM workflows, namely prompt engineering :hammer_and_wrench: :nut_and_bolt: :computer:.

---
#### Get Started

Please go through the following documents to get started with the challenge.

1. [Setting Python Environment/Dependencies](https://gitlab.analog.com/aaldujai/llm_challenge_as23/-/blob/main/docs/PYENV.md)
2. [Running Notebooks and Python Scripts](https://gitlab.analog.com/aaldujai/llm_challenge_as23/-/blob/main/docs/RUN.md)
2. [Challenge Dataset and Submission Format](https://gitlab.analog.com/aaldujai/llm_challenge_as23/-/blob/main/docs/DATA.md)
3. [Challenge Submission and Evaluation](https://gitlab.analog.com/aaldujai/llm_challenge_as23/-/blob/main/docs/EVAL.md)

We have created some videos to go along the above documents and understand the task of the challenge better. These videos can be found [here](https://analogdevices.hosted.panopto.com/Panopto/Pages/Sessions/List.aspx?folderID=3538a09f-201f-4bef-8177-b03d01447d8b). We recommend watching `Code Setup` and `OpenAI API Key Setup` before `LLM Challenge Overview`. The slides that go with the last video can be found [here](https://gitlab.analog.com/aaldujai/llm_challenge_as23/-/blob/main/docs/Challenge-Overview.pptx).



---
#### Ideas / Suggestions to beat the baselines

- Do not take embeddings at face value. Train an ML model that improves embedding-based retrieval.
- Vector store is not the only way to have an embedding-based retrieval. Check [Exemplar SVM](https://icml.cc/2012/papers/946.pdf) for an example of how instance-based ranking of text snippets can be done.
- A chain of LLM calls (e.g., LangChain's stuff, map-reduce, map-rank, map-refine).
- Other ideas discussed in the `LLM Challenge Overview` video linked above and listed in the accompanying slides.

---
#### Feedback / Questions / Bug Reports

Feel free to share your experience on the [TEAMS group](https://teams.microsoft.com/l/team/19%3AsPkLqzyDM78MDQq5LbWL2doQgc9LjbGEMjlReS6MkT01@thread.tacv2/conversations?groupId=aa7e9374-9bed-4cf3-9c88-b387c82f30da&tenantId=eaa689b4-8f87-40e0-9c6f-7228de4d754a). Or simply raise an issue for this repository.

---
#### ADI Generative AI Policy

As we go through the nuts and bolts of Prompt Engineering, let's remind ourselves with ADI GenAI policy and understand what can and can not be done with these LLMs for ADI. As long as the information that we share with GenAI tools is publicly available, there should not be an problem with ADI's policy. More details can be found [here](https://thecircuit.web.analog.com/about-adi/Shared%20Documents/Policies-and-Procedures/Generative%20AI%20Policy%20-%20Final.pdf).


---
#### Acknowledgement

We would like to thank Nick Moran, Wenjie Lu, Steve Wacks, Sefa Demirtas, Tao Yu, Matt Crivello, Andrew Fitzell, Marc Light, Dave Boland, and Chris Cianciolo.




**Happy prompt-engineering :sunglasses:**


