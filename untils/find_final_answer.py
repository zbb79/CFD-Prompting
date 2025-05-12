def find_max_index(prob_list,Answers):
    if not prob_list:
        return None

    max_index = prob_list.index(max(prob_list))
    best_answer=Answers[max_index]
    return best_answer







def find_best_answer(A, B):
    if len(A) != len(B):
        raise ValueError("wrong")

    probability_dict = {}

    for element, probability in zip(A, B):
        if element in probability_dict:
            probability_dict[element] += probability
        else:
            probability_dict[element] = probability
    sorted_probabilities = sorted(probability_dict.items(), key=lambda x: x[1], reverse=True)

    if sorted_probabilities and sorted_probabilities[0][0] == None:
        if len(sorted_probabilities) > 1:
            return sorted_probabilities[1][0]
        else:
            return None
    elif sorted_probabilities:
        return sorted_probabilities[0][0]
    else:
        return None




