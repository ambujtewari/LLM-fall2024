
**STATS 700, Fall 2024**

The Attention mechanism and the Transformer architecture have completely changed the landscape of AI, deep learning, and NLP research in the past few years. This course will be a selective review of the fast growing literature on Transformers and Large Language Models (LLMs) with a preference towards theoretical and mathematical analyses. We will study the limitations and capabilities of the transformer architecture. We will discuss empirical phenomena such as neural scaling laws and emergence of skills as the models are scaled up in size. LLMs also raise issues around copyright, trust, safety, fairness, and watermarking. We will look at alignment to human values and techniques such as RLHF (reinforcement learning with human feedback) as well as adaptation of LLMs to downstream tasks via few shot fine-tuning and in-context learning. Towards the end, we might look at the impact that LLMs are having in disciplines such as Cognitive Science, Linguistics, and Neuroscience. We might also discuss ongoing efforts to build LLMs and foundation models for science and mathematics. This course is inspired by the Special Year ([Part 1](https://simons.berkeley.edu/programs/special-year-large-language-models-transformers-part-1), [Part 2](https://simons.berkeley.edu/programs/special-year-large-language-models-transformers-part-2) and an earlier [workshop](https://simons.berkeley.edu/workshops/large-language-models-transformers)) on LLMs and Transformers being hosted by the Simons Institute at UC Berkeley and may be tweaked to better align with it as the Special Year progresses.

_Note_: This course is primarily meant for Statistics PhD students. Others will need instructor's permission to enroll. Graduate coursework in statistics, theoretical computer science, mathematics, or related disciplines required. Students will be expected to possess that hard-to-define quality usually referred to as "mathematical maturity". 

* TOC
{:toc}

# Logistics

Days and Times: Tuesdays and Thursdays, 11:30 am-1:00 pm  
Location: [USB2260](https://mclassrooms.umich.edu/rooms/2113853)  

# Courses / Blogs

[Stanford](https://stanford-cs324.github.io/winter2022/)    
[Princeton](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/)    
[Berkeley](https://rdi.berkeley.edu/understanding_llms/s24)    
[Michigan EECS](https://www.dropbox.com/scl/fi/xx8bu60mpn2rg84txmr9x/EECS598_LLM_syllabus.pdf?rlkey=q4lgtwtlce8gkbr07tqje0srj&dl=0)    
Borealis AI blog series:
- [Intro to LLMs](https://www.borealisai.com/research-blogs/a-high-level-overview-of-large-language-models/)
- [Transformers I](https://www.borealisai.com/en/blog/tutorial-14-transformers-i-introduction/)
- [Transformers II](https://www.borealisai.com/en/blog/tutorial-16-transformers-ii-extensions/)
- [Transformers III](https://www.borealisai.com/en/blog/tutorial-17-transformers-iii-training/)
- [Neural Natural Language Generation: Decoding Algorithms](https://www.borealisai.com/research-blogs/tutorial-6-neural-natural-language-generation-decoding-algorithms/)
- [Neural Natural Language Generation: Sequence Level Training](https://www.borealisai.com/research-blogs/tutorial-7-neural-natural-language-generation-sequence-level-training/)
- [Training and fine-tuning LLMs](https://www.borealisai.com/research-blogs/training-and-fine-tuning-large-language-models/)
- [Speeding up inference in LLMs](https://www.borealisai.com/research-blogs/speeding-up-inference-in-transformers/)

# Schedule

J&M = [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) (3rd ed. draft), Jurafsky and Martin
M&S = [Foundations of Statistical Natural Language Processing](https://nlp.stanford.edu/fsnlp/), Manning and Schütze
C&T = [Elements of Information Theory](https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X) (2nd ed.), Cover and Thomas

1. N-gram language models, J&M Chapter 3
2. Essential Information Theory, M&S Section 2.2
3. C&T, Chapters 2-6?
4. Vector Semantics and Embeddings, J&M Chapter 6
5. Neural Networks and Neural Language Models, J&M Chapter 7
6. RNNs and LSTMs, J&M Chapter 9
7. Transformers and Large Language Models, J&M Chapter 10

# Topics

## Word Embeddings
[Analogies Explained: Towards Understanding Word Embeddings](https://proceedings.mlr.press/v97/allen19a/allen19a.pdf)    

## Attention

[What can a Single Attention Layer Learn? A Study Through the Random Features Lens](https://arxiv.org/pdf/2307.11353.pdf)    
[Inductive Biases and Variable Creation in Self-Attention Mechanisms](https://arxiv.org/pdf/2110.10090.pdf)    

## Implicit Regularization

[Implicit Regularization of Gradient Flow on One-Layer Softmax Attention](https://arxiv.org/pdf/2403.08699.pdf)    

## Basics of Transformers

[Formal Algorithms for Transformers](https://arxiv.org/pdf/2207.09238.pdf)    

## NTK Theory for Transformers

[Infinite attention: NNGP and NTK for deep attention networks](https://arxiv.org/pdf/2006.10540.pdf)    
[Tensor Programs II: Neural Tangent Kernel for Any Architecture](https://arxiv.org/pdf/2006.14548.pdf)  
[A Kernel-Based View of Language Model Fine-Tuning](https://proceedings.mlr.press/v202/malladi23a/malladi23a.pdf)

## Capabilities and Limitations of Transformers

[On the Turing Completeness of Modern Neural Network Architectures](https://arxiv.org/pdf/1901.03429.pdf)    
[Are Transformers universal approximators of sequence-to-sequence functions?](https://arxiv.org/pdf/1912.10077.pdf)    
[From Self-Attention to Markov Models: Unveiling the Dynamics of Generative Transformers](https://arxiv.org/pdf/2402.13512.pdf)    
[On the Ability and Limitations of Transformers to Recognize Formal Languages](https://aclanthology.org/2020.emnlp-main.576.pdf)    
[Theoretical Limitations of Self-Attention in Neural Sequence Models](https://arxiv.org/pdf/1906.06755.pdf)    
[Self-Attention Networks Can Process Bounded Hierarchical Languages](https://arxiv.org/pdf/2105.11115.pdf)    
[On Limitations of the Transformer Architecture](https://arxiv.org/pdf/2402.08164.pdf)    
[Transformers Learn Shortcuts to Automata](https://arxiv.org/pdf/2210.10749)  

## Beyond PAC Learning: Learning Distributions and Grammars

[On the Learnability of Discrete Distributions](https://dl.acm.org/doi/pdf/10.1145/195058.195155)  
[Grammatical Inference: Learning Automata and Grammars](https://www.cambridge.org/us/universitypress/subjects/computer-science/pattern-recognition-and-machine-learning/grammatical-inference-learning-automata-and-grammars?format=HB&isbn=9780521763165)  
[Mathematical Linguistics](https://link.springer.com/book/10.1007/978-1-84628-986-6) especially Chapter 7 (Complexity) and Chapter 8 (Linguistic pattern recognition)  

## Emergence

[Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/pdf/2304.15004.pdf)    
[A Theory for Emergence of Complex Skills in Language Models](https://arxiv.org/pdf/2307.15936.pdf)    

## In-Context Learning

[What Can Transformers Learn In-Context? A Case Study of Simple Function Classes](https://arxiv.org/pdf/2208.01066.pdf)    
[The Learnability of In-Context Learning](https://arxiv.org/pdf/2303.07895.pdf)    
[Supervised Pretraining Can Learn In-Context Reinforcement Learning](https://arxiv.org/pdf/2306.14892.pdf)    
[Large Language Models can Implement Policy Iteration](https://proceedings.neurips.cc/paper_files/paper/2023/file/60dc7fa827f5f761ad481e2ad40b5573-Paper-Conference.pdf)    
[Trainable Transformer in Transformer](https://arxiv.org/pdf/2307.01189)
[Trained Transformers Learn Linear Models In-Context](https://www.jmlr.org/papers/volume25/23-1042/23-1042.pdf)    
[The Evolution of Statistical Induction Heads: In-Context Learning Markov Chains](https://arxiv.org/pdf/2402.11004)  

## Assessing Model Uncertainty 

[Distinguishing the Knowable from the Unknowable with Language Models](https://arxiv.org/pdf/2402.03563)  
[Conformal Language Modeling](https://arxiv.org/pdf/2306.10193)  
[Language Models with Conformal Factuality Guarantees](https://arxiv.org/pdf/2402.10978)  

## RLHF

[Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155.pdf)    
[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290.pdf)    
[Efficient Exploration for LLMs](https://arxiv.org/pdf/2402.00396.pdf)    

## State Space Models

[Mamba: Linear-time sequence modeling with selective state spaces](https://arxiv.org/ftp/arxiv/papers/2312/2312.00752.pdf)    
[Repeat After Me: Transformers are Better than State Space Models at Copying](https://arxiv.org/pdf/2402.01032.pdf)    

## Transformers as Interacting Particle Systems

[A mathematical perspective on Transformers](https://arxiv.org/pdf/2312.10794.pdf)

## Transformers in RL

[Transformers in Reinforcement Learning: A Survey](https://arxiv.org/pdf/2307.05979.pdf)

## LLMs and Cognitive Science, Linguistics, Neuroscience

[The debate over understanding in AI’s large language models](https://www.pnas.org/doi/abs/10.1073/pnas.2215907120)    
[Language models and linguistic theories beyond words](https://www.nature.com/articles/s42256-023-00703-8)    
[Noam Chomsky: The False Promise of ChatGPT](https://www.nytimes.com/2023/03/08/opinion/noam-chomsky-chatgpt-ai.html)    
[Modern language models refute Chomsky’s approach to language](https://lingbuzz.net/lingbuzz/007180)    
[Dissociating language and thought in large language models](https://arxiv.org/pdf/2301.06627.pdf)  
[Embers of Autoregression: Understanding Large Language Models Through the Problem They are Trained to Solve](https://arxiv.org/pdf/2309.13638.pdf)  
[Shared computational principles for language processing in humans and deep language models](https://doi.org/10.1038/s41593-022-01026-4)  
[The neural architecture of language: Integrative modeling converges on predictive processing](https://doi.org/10.1073/pnas.2105646118)  
[Predictive Coding or Just Feature Discovery? An Alternative Account of Why Language Models Fit Brain Data](https://doi.org/10.1162/nol_a_00087)  

## LLMs and Foundation Models for Science and Mathematics

[On the Opportunities and Risks of Foundation Models](https://crfm.stanford.edu/assets/report.pdf)    
[MIDAS Symposium 2024](https://midas.umich.edu/ai-se-annual-symposium/)    
[MICDE Symposium 2024](https://micde.umich.edu/news-events/annual-symposia/2024-symposium/)    
