from encoder.Encoder import *

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from llm.getllm import generate_cot



def p_C_A(a, A, E, t):

    B, answers = generate_cot(E, t, 10)
    all_texts = [a] + B


    cot_matrix = encoder(all_texts)
    vector_a = cot_matrix[0]
    vectors_B = cot_matrix[1:]


    if vector_a.ndim == 1:
        vector_a = vector_a.reshape(1, -1)
    if vectors_B.ndim == 1:
        vectors_B = vectors_B.reshape(1, -1)


    similarities = cosine_similarity(vector_a, vectors_B).flatten()


    threshold = 0.9
    high_sim_indices = np.where(similarities >= threshold)[0]
    high_sim_count = len(high_sim_indices)
    high_sim_ratio = high_sim_count / len(B)
    high_sim_chains = [B[idx] for idx in high_sim_indices]
    high_sim_answers = [answers[idx] for idx in high_sim_indices]

    if high_sim_count > 0:
        different_count = sum(1 for ans in high_sim_answers if ans != A)
        different_ratio = different_count / high_sim_count
    else:
        different_ratio = 0


    return high_sim_ratio,different_ratio






