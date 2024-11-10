
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

## Part 1

- 8/27 Introduction [slides](https://docs.google.com/presentation/d/1ozkV1Kk4wPucriWIT_kk6QQabwaG3xl2/edit?usp=sharing&ouid=105036821118529706206&rtpof=true&sd=true)
- 8/29 N-gram Language Models, J&M Chapter 3 [annotated chapter](https://www.dropbox.com/scl/fi/787j6oay929yawgfeypce/3.pdf?rlkey=g4y673lj11d8xh7zy7883okm9&st=c6qlxrpi&dl=0)
- 9/3, 9/5 Entropy, Relative Entropy, and Mutual Information, C&T Chapter 2, Sections 2.1-2.5 [notes](https://www.dropbox.com/scl/fi/pumev44yqz2tyca3pw6gc/CT-Sec-2-1-to-2-5.pdf?rlkey=d2c92e0w41yiqkxkxl1eraw7j&st=knnx8u3g&dl=0)  
- 9/10, 9/12 Entropy, Relative Entropy, and Mutual Information, C&T Chapter 2, Sections 2.6-2.10 [notes](https://www.dropbox.com/scl/fi/6in60dt7n4c02gimbjjl4/CT-Sec-2-6-to-2-10.pdf?rlkey=apx8z8oma104ktarn6h2i39vi&st=zl6mdfte&dl=0)
- 9/17, 9/19 Paper presentation [notes](https://www.dropbox.com/scl/fi/96ly78ynmazx0b4vux6id/single-layer-self-attention-and-communication-complexity.pdf?rlkey=er31wvchscsrkp1bh7jaghpy3&st=x80j6vdu&dl=0)  
  - [Representational Strengths and Limitations of Transformers](https://arxiv.org/pdf/2306.02896)
  - [On Limitations of the Transformer Architecture](https://arxiv.org/pdf/2402.08164)
  - [One-layer transformers fail to solve the induction heads task](https://arxiv.org/pdf/2408.14332)  
- 9/19 (wrap up) Entropy, Relative Entropy, and Mutual Information, C&T Chapter 2, Sections 2.6-2.10 [notes](https://www.dropbox.com/scl/fi/6in60dt7n4c02gimbjjl4/CT-Sec-2-6-to-2-10.pdf?rlkey=apx8z8oma104ktarn6h2i39vi&st=zl6mdfte&dl=0)  
- 9/24 Vector Semantics and Embeddings, J&M Chapter 6 [annotated chapter](https://www.dropbox.com/scl/fi/yee8frm5qeo3vzwrxrgrd/6.pdf?rlkey=rpmjt59exuxjdz0fi6f9iz4xm&st=3r6f026q&dl=0)  
- 9/26, 10/1 Asymptotic Equipartition Property, C&T Chapter 3, Sections 3.1-3.3 [notes](https://www.dropbox.com/scl/fi/1c8xloiluxr2j4znol40y/CT-Sec-3-1-to-3-3.pdf?rlkey=ktlrvtahq5x6fc0pvam8edz44&st=qx0cso0n&dl=0)  
- 10/3 Entropy Rates of a Stochastic Process, C&T Chapter 4, Sections 4.1-4.3 [notes](https://www.dropbox.com/scl/fi/hcg1kig528gl4b3n75r7w/CT-Sec-4-1-to-4-3.pdf?rlkey=zkb1lnlsszs6sc6zwk4qldp0n&st=r5ne9ust&dl=0)  
- 10/10 Project Pitches  
- 10/15 FALL BREAK  
- 10/17 Project Pitches
- 10/22 Entropy Rates of a Stochastic Process, C&T Chapter 4, Sections 4.4-4.5 [notes](https://www.dropbox.com/scl/fi/lg3924896ze0jzr9wfhie/CT-Sec-4-4-to-4-5.pdf?rlkey=hi4z6zhuqptp8zlogsocrs2zj&st=dm2428zt&dl=0)  
- 10/24 Guest Lecture by Vinod Raman: [Generation through the lens of learning theory](https://arxiv.org/pdf/2410.13714)
- 10/28 Project Proposals Due  

**Supplementary Material**

- Neural Networks, J&M Chapter 7 [annotated chapter](https://www.dropbox.com/scl/fi/1ynppiktk1f2hy8srkolv/7.pdf?rlkey=z9sz3d13mbirbm1g43653bsql&st=4zjhh7d3&dl=0)  
  - RNNs and LSTMs, J&M Chapter 8 [annotated chapter](https://www.dropbox.com/scl/fi/pki1mdcxqtv1s64f768h9/8.pdf?rlkey=cjs5m9ubvb77sxe6tskt7kju4&st=6rzi9ycs&dl=0)  
- Transformers, J&M Chapter 9  
  - Large Language Models, J&M Chapter 10  

## Part 2

- 10/29
  - T1
    - [AI models collapse when trained on recursively generated data](https://doi.org/10.1038/s41586-024-07566-y)
    - [The Curse of Recursion: Training on Generated Data Makes Models Forget](https://arxiv.org/pdf/2305.17493)
    - [How Bad is Training on Synthetic Data? A Statistical Analysis of Language Model Collapse](https://arxiv.org/pdf/2404.05090)  
  - T2
    - [Prediction-Powered Ranking of Large Language Models](https://arxiv.org/pdf/2402.17826)
    - [AutoEval Done Right: Using Synthetic Data for Model Evaluation](https://arxiv.org/pdf/2403.07008)
- 10/31
  - T4
    - [The Last JITAI?](https://arxiv.org/pdf/2402.08658)
    - [Thompson Sampling for Noncompliant Bandits](https://arxiv.org/pdf/1812.00856)  
    - [Neural Thompson Sampling](https://arxiv.org/pdf/2010.00827)  
  - T5 [On the Dimensionality of Word Embeddings](https://arxiv.org/pdf/1812.04224)
- 11/5
  - T9 [SpectraTr: A novel deep learning model for qualitative analysis of drug spectroscopy based on transformer structure](https://worldscientific.com/doi/epdf/10.1142/S1793545822500213)
  - T17 [Distinguishing the Knowable from the Unknowable with Language Models](https://arxiv.org/pdf/2402.03563)
- 11/7
  - T12 [Conformal Language Modeling](https://arxiv.org/pdf/2306.10193)
  - T13
    - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929)
    - [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020)
    - [Indication as Prior Knowledge for Multimodal Disease Classification in Chest Radiographs with Transformers](https://arxiv.org/pdf/2202.06076)
- 11/12
  - T14
    - [Large Language Models Are Zero-Shot Time Series Forecasters](https://arxiv.org/pdf/2310.07820v3)
    - [Time-LLM: Time Series Forecasting by Reprogramming Large Language Models](https://openreview.net/pdf?id=Unb5CVPtae)
  - T15
    - [AI agents that matter](https://arxiv.org/abs/2407.01502)
    - [Are more LLM calls all you need? Towards scaling laws of compound inference systems](https://arxiv.org/pdf/2403.02419)
- 11/14
  - T7 [Improving Pretraining Data Using Perplexity Correlations](https://arxiv.org/pdf/2409.05816)
  - T16 [Controllable Neural Symbolic Regression](https://arxiv.org/pdf/2304.10336)
- 11/19
  - T3 [A statistical framework for weak-to-strong generalization](https://arxiv.org/pdf/2405.16236v1)
  - T6 [Large Language Models as General Pattern Machines](https://doi.org/10.48550/arXiv.2307.04721)
- 11/21
  - T8
    - [GNOT: A General Neural Operator Transformer for Operator Learning](https://proceedings.mlr.press/v202/hao23c/hao23c.pdf)
    - [Understanding Deep Neural Networks with Rectified Linear Units](https://arxiv.org/pdf/1611.01491)
  - T11 [Repeat After Me: Transformers are Better than State Space Models at Copying](https://arxiv.org/pdf/2405.13798)
- 11/26
  - T10 [Slaves to the Law of Large Numbers: An Asymptotic Equipartition Property for Perplexity in Generative Language Models](https://arxiv.org/pdf/2405.13798)
- 11/28 THANKSGIVING BREAK  

**Dec 3, Poster Session I: Understanding and Improving LLMs & Transformers**

T1 Soham, Sunrit: Evolution of Iteratively Trained Generative AI Models  
T2 Xuanyu, Yiling: Context-Aware Ranking of Large Language Models via Pairwise Comparison  
T3 Elvin, Jason, Kellen, Mihir: Is Linear Probing better than Fine-tuning for Weak-to-Strong Generalization?  
T5 Jake, Jaylin, Noah: Dimension Decisions: The Impact of Embedding Size on LLMs  
T7 Eduardo, Felipe, Harry, Xinhe: shIRT: similarity heuristics for data selection using IRT  
T10 Mojtaba, Tara: Beat LLMs in Their Own Game: Statistics-based Methods for AI Detection with Theoretical Guarantees  
T11 Unique, Vinod: The Limitations of Self-Attention for Information Retrieval  
T12 Paolo, Sahana: Budget-Constrained Learning to Defer for Autoregressive Models  
T17 Yuezhou: Classifying Uncertainty with In-Context Learning Methods  

**Dec 5, Poster Session II: Applying LLMs & Transformers**

T4 Gabe, Ki, Marc: Online RL Considerations for LLM-Assisted JITAIs  
T6 Andrej: Modern autoregressive architectures for the collective variable problem  
T8 Abhiti, Julian, Yash: daptive Spectral Neural Operators  
T9 Joe: Applying Transformers to Spectral Data  
T13 Qiyuan, Zhilin: A Unified Framework for Multimodal Learning: Integrating Image, Text, and Crowd-Sourced Annotations  
T14 Victor: Forecasting Solar Flares Using Time Series Foundation Models and LLM-Based Models  
T15 Jiwoo, Yue: AI Agent Evaluation  
T16 Daniel: Incorporating Domain Knowledge in Transformer-based models for Symbolic Regression  

**Dec 13, Project Reports Due**


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

Basics
- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) This is the paper that started the transformers revolution
- [Formal Algorithms for Transformers](https://arxiv.org/pdf/2207.09238)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)  

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

## Reverse Engineering Transformers

[Transformer Circuits Thread Project](https://transformer-circuits.pub/)  
[Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/pdf/2211.00593)  
[Progress Measures for Grokking Via Mechanistic Interpretability](https://arxiv.org/pdf/2301.05217)  

## Capabilities and Limitations of LLMs and Transformers

[On the Turing Completeness of Modern Neural Network Architectures](https://arxiv.org/pdf/1901.03429.pdf)    
[Are Transformers universal approximators of sequence-to-sequence functions?](https://arxiv.org/pdf/1912.10077.pdf)    
[Theoretical Limitations of Self-Attention in Neural Sequence Models](https://arxiv.org/pdf/1906.06755.pdf)    
[On the Ability and Limitations of Transformers to Recognize Formal Languages](https://aclanthology.org/2020.emnlp-main.576.pdf)    
[Self-Attention Networks Can Process Bounded Hierarchical Languages](https://arxiv.org/pdf/2105.11115.pdf)    
[Transformers Learn Shortcuts to Automata](https://arxiv.org/pdf/2210.10749)  
[Representational Strengths and Limitations of Transformers](https://arxiv.org/pdf/2306.02896)  
[On Limitations of the Transformer Architecture](https://arxiv.org/pdf/2402.08164.pdf)    
[Transformers, parallel computation, and logarithmic depth](https://arxiv.org/pdf/2402.09268)  
[From Self-Attention to Markov Models: Unveiling the Dynamics of Generative Transformers](https://arxiv.org/pdf/2402.13512.pdf)    
[Slaves to the Law of Large Numbers: An Asymptotic Equipartition Property for Perplexity in Generative Language Models](https://arxiv.org/pdf/2405.13798)  
[One-layer transformers fail to solve the induction heads task](https://arxiv.org/pdf/2408.14332)  
[Thinking Like Transformers](https://srush.github.io/raspy/)  

## Beyond PAC Learning: Learning Distributions and Grammars, Learning to Generate

[On the Learnability of Discrete Distributions](https://dl.acm.org/doi/pdf/10.1145/195058.195155)  
[Near-optimal Sample Complexity Bounds for Robust Learning of Gaussian Mixtures via Compression Schemes](https://dl.acm.org/doi/pdf/10.1145/3417994)  
[Distribution Learnability and Robustness](https://proceedings.neurips.cc/paper_files/paper/2023/file/a5321f64005b0d4a94d0b18e84e19f48-Paper-Conference.pdf)  
[Inherent limitations of dimensions for characterizing learnability of distribution classes](https://proceedings.mlr.press/v247/lechner24a/lechner24a.pdf)  
[Grammatical Inference: Learning Automata and Grammars](https://www.cambridge.org/us/universitypress/subjects/computer-science/pattern-recognition-and-machine-learning/grammatical-inference-learning-automata-and-grammars?format=HB&isbn=9780521763165)  
[Mathematical Linguistics](https://link.springer.com/book/10.1007/978-1-84628-986-6) especially Chapter 7 (Complexity) and Chapter 8 (Linguistic pattern recognition)  
[Language Generation in the Limit](https://arxiv.org/pdf/2404.06757)  

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

[Formal grammar and information theory: together again?](https://doi.org/10.1098/rsta.2000.0583)  
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
