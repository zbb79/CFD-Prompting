# CFD-Prompting
## File Explanation
cdf_prompt.py: The CFD_Prompting main function file.<br>
cdf_prompt.py: The main function of the CFD_Prompting method without clustering-based filtering, used for ablation studies.<br>
train.py: Dataset test file.<br>
data: Store the test dataset.<br>
### encoder
Encoder.py: Call the encoder file,and in the ablation study, the corresponding encoder model can be replaced simply by changing the “model_path” parameter in the file.<br>
### llm
getllm.py: Store the large model interface and the calling file.<br>
### model
  Place the encoder and the corresponding fine-tuning function folder,The original encoder model we use is the bert-base-uncased model.<br>
### untils <br>
1. E_gen.py: Counterfactual context construction file.<br>
2. kmeans.py: K-means construction file.<br>
3. find_final_answer.py: Return the final answer.<br>
4. probability.py: calculate probability <br>

