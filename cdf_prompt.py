from llm.getllm import *
from untils.E_gen import alt_E
from untils.probability import *
from untils.find_final_answer import *
from untils.Kmeans import *

def CFD_CoT(T,E):
    M=30
    K=5
    A_ace=[]
    CoT,Answers=generate_cot(E,T,M)

    cluster_centers_CoT,Answer=get_most_different_chains_by_clustering(CoT, Answers,K)

    Alt_E, Prob = alt_E(T,E)
    for i in range(len(cluster_centers_CoT)):
        P_Ci = 0
        for j in range(len(Alt_E)):
            P_C_te,P_A_tce=p_C_A(cluster_centers_CoT[i], Answer[i], Alt_E[j], T)

            P_ej=Prob[j]

            P_Ci+=P_A_tce*P_C_te*P_ej

        A_ace.append(P_Ci)
    final_vote_answer = find_best_answer(Answer,A_ace)
    return final_vote_answer








