
**STATS 700, Fall 2024**

The Attention mechanism and the Transformer architecture have completely changed the landscape of AI, deep learning, and NLP research in the past few years. This advanced graduate level course will consist of two parts. In the first part, we will review foundational material in information theory, statistical NLP, and deep learning theory. In the second part, student project teams will explore topics from the fast growing literature on Transformers and Large Language Models (LLMs) especially papers that provide theoretical and mathematical analyses. Topics include, but are not limited to:

- limitations and capabilities of the transformer architecture
- neural scaling laws and emergence of skills as the models are scaled up in size
- issues around copyright, trust, safety, fairness, and watermarking
- alignment to human values and techniques such as RLHF (reinforcement learning with human feedback)
- adaptation of LLMs to downstream tasks via few shot fine-tuning and in-context learning.

If there is time, we might look at the impact that LLMs are having in disciplines such as Cognitive Science, Linguistics, and Neuroscience. We might also discuss ongoing efforts to build LLMs and foundation models for science and mathematics.

_Note_: This course is primarily meant for Statistics PhD students. Others will need instructor's permission to enroll. Graduate coursework in statistics, theoretical computer science, mathematics, or related disciplines required. Students will be expected to possess that hard-to-define quality usually referred to as "mathematical maturity". 

* TOC
{:toc}

# Logistics & Schedule

Days and Times: Tuesdays and Thursdays, 11:30 am-1:00 pm  
Location: [USB2260](https://mclassrooms.umich.edu/rooms/2113853)  

J&M = [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) (3rd ed. draft), Jurafsky and Martin  
C&T = [Elements of Information Theory](https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X) (2nd ed.), Cover and Thomas  
DLT = [Deep Learning Theory Lecture Notes](https://mjt.cs.illinois.edu/dlt/index.pdf), Matus Telgarsky

## Part 1

1. 8/27 Introduction [slides](https://docs.google.com/presentation/d/1ozkV1Kk4wPucriWIT_kk6QQabwaG3xl2/edit?usp=sharing&ouid=105036821118529706206&rtpof=true&sd=true)
2. 8/29 N-gram Language Models, J&M Chapter 3 [annotated chapter](https://www.dropbox.com/scl/fi/787j6oay929yawgfeypce/3.pdf?rlkey=g4y673lj11d8xh7zy7883okm9&st=c6qlxrpi&dl=0)
3. 9/3 Entropy, Relative Entropy, and Mutual Information, C&T Chapter 2, Sections 2.1-2.5 [notes](https://www.dropbox.com/scl/fi/pumev44yqz2tyca3pw6gc/CT-Sec-2-1-to-2-5.pdf?rlkey=d2c92e0w41yiqkxkxl1eraw7j&st=knnx8u3g&dl=0)  
4. 9/5 Entropy, Relative Entropy, and Mutual Information, C&T Chapter 2, Sections 2.6-2.10 [notes](https://www.dropbox.com/scl/fi/6in60dt7n4c02gimbjjl4/CT-Sec-2-6-to-2-10.pdf?rlkey=apx8z8oma104ktarn6h2i39vi&st=zl6mdfte&dl=0)  
5. 9/10 Asymptotic Equipartition Property, C&T Chapter 3
6. 9/12 Entropy Rates of a Stochastic Process, C&T Chapter 4, Sections 4.1-4.3
7. 9/17 Entropy Rates of a Stochastic Process, C&T Chapter 4, Sections 4.4-4.5  
8. Data Compression, C&T Chapter 5
9. Gambling and Data Compression, C&T Chapter 6
10. Vector Semantics and Embeddings, J&M Chapter 6
11. Neural Networks, J&M Chapter 7
12. RNNs and LSTMs, J&M Chapter 8
13. Transformers, J&M Chapter 9
14. Large Language Models, J&M Chapter 10
15. Masked Language Models, J&M Chapter 11
16. Model Alignment, Prompting, and In-Context Learning, J&M Chapter 12
17. Universal Approximation, DLT Chapter 2
18. Infinite-Width Networks and the Barron Norm, DLT Chapter 3
19. Neural Tangent Kernel, DLT Chapter 4
20. Benefits of Depth, DLT Chapter 5
21. Implicit Bias, DLT Chapter 10
22. Rademacher Complexity, DLT Chapters 13-14
23. Covering Numbers, DLT Chapters 15-16

## Part 2

TBD

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

Special Year on LLMs and Transformers at the Simons Institute, UC Berkeley:
- [Part 1](https://simons.berkeley.edu/programs/special-year-large-language-models-transformers-part-1)
- [Part 2](https://simons.berkeley.edu/programs/special-year-large-language-models-transformers-part-2)
- An earlier [workshop](https://simons.berkeley.edu/workshops/large-language-models-transformers)  

# Classics

Markov (1913) [An Example of Statistical Investigation of the Text _Eugene Onegin_ Concerning the Connection of Samples in Chains](https://drive.google.com/file/d/1eMDPSx0kWQOw0Sv7ulbr3Z_TV1G_m8Qe/view?usp=sharing)  
Shannon (1951) [Prediction and Entropy of Printed English](https://drive.google.com/file/d/1jOyEgx1paD4qBtb7NPdz0bXKnWlSnByn/view?usp=sharing)  
Zipf (1935) _The Psycho-Biology of Language_ Houghton, Mifflin. ([Reprinted by MIT Press](https://mitpress.mit.edu/9780262740029/the-psycho-biology-of-language/) in 1965)  
Good(-Turing) (1953) [The Population Frequencies of Species and the Estimation of Population Parameters](https://drive.google.com/file/d/1tqVVS_T73b6jYyWF0HSWuzlakwk3pWxD/view?usp=sharing)  

# Topics

## Word Embeddings
[Neural Word Embedding as Implicit Matrix Factorization](https://papers.nips.cc/paper_files/paper/2014/file/feab05aa91085b7a8012516bc3533958-Paper.pdf)  
[A Latent Variable Model Approach to PMI-based Word Embeddings](https://doi.org/10.1162/tacl_a_00106)  
[Skip-Gram – Zipf + Uniform = Vector Additivity](https://aclanthology.org/P17-1007.pdf)  
[Analogies Explained: Towards Understanding Word Embeddings](https://proceedings.mlr.press/v97/allen19a/allen19a.pdf)    

## Attention

[What can a Single Attention Layer Learn? A Study Through the Random Features Lens](https://arxiv.org/pdf/2307.11353.pdf)    
[Inductive Biases and Variable Creation in Self-Attention Mechanisms](https://arxiv.org/pdf/2110.10090.pdf)    

## Implicit Regularization

[Implicit Regularization of Gradient Flow on One-Layer Softmax Attention](https://arxiv.org/pdf/2403.08699.pdf)    

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
[Thinking Like Transformers](https://srush.github.io/raspy/)  

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

## Hallucinations

[Calibrated Language Models Must Hallucinate](https://arxiv.org/pdf/2311.14648.pdf)  

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

## Language Modeling, Prediction, and Compression

[On prediction by data compression](https://link.springer.com/chapter/10.1007/3-540-62858-4_69)  
[Language Modeling Is Compression](https://arxiv.org/pdf/2309.10668)  
[Prediction by Compression](https://arxiv.org/pdf/1008.5078)  

## LLMs, Online Learning, and Regret

[Do LLM Agents Have Regret? A Case Study in Online Learning and Games](https://arxiv.org/pdf/2403.16843)  

## Transformers in RL

[Transformers in Reinforcement Learning: A Survey](https://arxiv.org/pdf/2307.05979.pdf)

## LLMs and Causality

[Causal Reasoning and Large Language Models: Opening a New Frontier for Causality](https://arxiv.org/pdf/2305.00050)  

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
