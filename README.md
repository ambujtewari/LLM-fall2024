
The Attention mechanism and the Transformer architecture have completely changed the landscape of AI, deep learning, and NLP research in the past few years. This course will be a selective review of the fast growing literature on Transformers and Large Language Models (LLMs) with a preference towards theoretical and mathematical analyses. We will study the limitations and capabilities of the transformer architecture. We will discuss empirical phenomena such as neural scaling laws and emergence of skills as the models are scaled up in size. LLMs also raise issues around copyright, trust, safety, fairness, and watermarking. We will look at alignment to human values and techniques such as RLHF (reinforcement learning with human feedback) as well as adaptation of LLMs to downstream tasks via few shot fine-tuning and in-context learning. Towards the end, we might look at the impact that LLMs are having in disciplines such as Cognitive Science, Linguistics, and Neuroscience. We might also discuss ongoing efforts to build LLMs and foundation models for science and mathematics. This course is inspired by the Special Year ([Part 1](https://simons.berkeley.edu/programs/special-year-large-language-models-transformers-part-1), [Part 2](https://simons.berkeley.edu/programs/special-year-large-language-models-transformers-part-2) and an earlier [workshop](https://simons.berkeley.edu/workshops/large-language-models-transformers)) on LLMs and Transformers being hosted by the Simons Institute at UC Berkeley and may be tweaked to better align with it as the Special Year progresses.

# List of Courses / Blogs

[Stanford](https://stanford-cs324.github.io/winter2022/)    
[Princeton](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/)    
[Berkeley](https://rdi.berkeley.edu/understanding_llms/s24)    
[Michigan EECS](https://www.dropbox.com/scl/fi/xx8bu60mpn2rg84txmr9x/EECS598_LLM_syllabus.pdf?rlkey=q4lgtwtlce8gkbr07tqje0srj&dl=0)    
Borealis AI blog series:
- [Intro to LLMs](https://www.borealisai.com/research-blogs/a-high-level-overview-of-large-language-models/)
- [Transformers I](https://www.borealisai.com/en/blog/tutorial-14-transformers-i-introduction/)
- [Transformers II](https://www.borealisai.com/en/blog/tutorial-16-transformers-ii-extensions/)
- [Transformers III](https://www.borealisai.com/en/blog/tutorial-17-transformers-iii-training/)
- [Training and fine-tuning LLMs](https://www.borealisai.com/research-blogs/training-and-fine-tuning-large-language-models/)
- [Speeding up inference in LLMs](https://www.borealisai.com/research-blogs/speeding-up-inference-in-transformers/)

# List of Papers

## Word Embeddings
[Analogies Explained: Towards Understanding Word Embeddings](https://proceedings.mlr.press/v97/allen19a/allen19a.pdf)    

## Attention

[What can a Single Attention Layer Learn? A Study Through the Random Features Lens](https://arxiv.org/pdf/2307.11353.pdf)    
[Inductive Biases and Variable Creation in Self-Attention Mechanisms](https://arxiv.org/pdf/2110.10090.pdf)    

## Implicit Rregularization

[Implicit Regularization of Gradient Flow on One-Layer Softmax Attention](https://arxiv.org/pdf/2403.08699.pdf)    

## Basics of Transformers

[Formal Algorithms for Transformers](https://arxiv.org/pdf/2207.09238.pdf)    

## NTK Theory for Transformers

[Infinite attention: NNGP and NTK for deep attention networks](https://arxiv.org/pdf/2006.10540.pdf)    
[Tensor Programs II: Neural Tangent Kernel for Any Architecture](https://arxiv.org/pdf/2006.14548.pdf)

## Capabilities and Limitations of Transformers

[On the Turing Completeness of Modern Neural Network Architectures](https://arxiv.org/pdf/1901.03429.pdf)    
[Are Transformers universal approximators of sequence-to-sequence functions?](https://arxiv.org/pdf/1912.10077.pdf)    
[From Self-Attention to Markov Models: Unveiling the Dynamics of Generative Transformers](https://arxiv.org/pdf/2402.13512.pdf)    
[On the Ability and Limitations of Transformers to Recognize Formal Languages](https://aclanthology.org/2020.emnlp-main.576.pdf)    
[Theoretical Limitations of Self-Attention in Neural Sequence Models](https://arxiv.org/pdf/1906.06755.pdf)    
[Self-Attention Networks Can Process Bounded Hierarchical Languages](https://arxiv.org/pdf/2105.11115.pdf)    
[On Limitations of the Transformer Architecture](https://arxiv.org/pdf/2402.08164.pdf)    

## Emergence

[Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/pdf/2304.15004.pdf)    
[A Theory for Emergence of Complex Skills in Language Models](https://arxiv.org/pdf/2307.15936.pdf)    

## In-Context Learning

[What Can Transformers Learn In-Context? A Case Study of Simple Function Classes](https://arxiv.org/pdf/2208.01066.pdf)    
[The Learnability of In-Context Learning](https://arxiv.org/pdf/2303.07895.pdf)    
[Supervised Pretraining Can Learn In-Context Reinforcement Learning](https://arxiv.org/pdf/2306.14892.pdf)    
[Large Language Models can Implement Policy Iteration](https://proceedings.neurips.cc/paper_files/paper/2023/file/60dc7fa827f5f761ad481e2ad40b5573-Paper-Conference.pdf)    

## RLHF

[Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155.pdf)
[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290.pdf)

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

## LLMs and Foundation Models for Science and Mathematics

[On the Opportunities and Risks of Foundation Models](https://crfm.stanford.edu/assets/report.pdf)    
[MIDAS Symposium 2024](https://midas.umich.edu/ai-se-annual-symposium/)    
[MICDE Symposium 2024](https://micde.umich.edu/news-events/annual-symposia/2024-symposium/)    
