# CFD-Prompting
File Explanation
data:Store the test dataset.
encoder/Encoder.py:Call the encoder file.
llm/getllm.py:Store the large model interface and the calling file.
model:Place the encoder and the corresponding fine-tuning function folder,The original encoder model we use is the bert-base-uncased model.
untils/
  E_gen.py:Counterfactual context construction file.
  kmeans.py:K-means construction file.
  find_final_answer.py:Return the final answer.
cdf_prompt.py:The CFD_Prompting main function file.
train.py:Dataset test file.
