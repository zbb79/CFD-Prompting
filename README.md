# CFD-Prompting

- cdf_prompt.py: This file implements the main function for the CFD-Prompting framework.
- CaseStudy.pdf: Contains two examples based on the proposed model, one on the HotpotQA dataset and the other on the MuSiQue dataset.

### encoder
Encoder.py: Contains the encoder component used by CFD-Prompting.
### llm
getllm.py: Implements the interface and utilities for interacting with large language models (LLMs).
### models
This folder contains the encoder and its associated fine-tuning functions.
### untils 
1. E_gen.py: Constructs counterfactual external knowledge for CFD-Prompting framework.
2. kmeans.py: Implements K-means clustering used in answer aggregation.
3. find_final_answer.py: Returns the final answer based on ranked candidates.
4. probability.py: Calculates answer probabilities from model outputs.
