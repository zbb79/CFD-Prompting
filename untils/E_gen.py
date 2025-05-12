from llm.getllm import Entity_extract, AltEntity_extract
import re
import itertools


def calculate_combination_probability(weights, Entity, k):
    n = len(Entity)
    elements = list(range(n))

    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    comb_probabilities = {}
    select_entity = []
    sort = []


    for comb in itertools.combinations(elements, k):
        prob = 1.0
        remaining = 1.0
        used_weights = []
        for idx in comb:
            if remaining > 0:  
                prob *= normalized_weights[idx] / remaining
                used_weights.append(normalized_weights[idx])
                remaining -= normalized_weights[idx]
            else:
                prob = 0
                break
        comb_probabilities[comb] = prob
        select_entity.append([Entity[i] for i in comb])
        sort.append(list(comb))


    Prob = []
    total_prob = sum(comb_probabilities.values())
    if total_prob > 0:
        for comb in comb_probabilities:
            comb_probabilities[comb] /= total_prob
            Prob.append(comb_probabilities[comb])
    else:
        Prob = [1.0 / len(comb_probabilities)] * len(comb_probabilities)

    return select_entity, Prob, sort


def replace_keywords(text, original_entities, new_entities):

    for orig, new in zip(original_entities, new_entities):
        text = text.replace(orig, new)
    return text



def extract_entities(text):

    match = re.search(r'\[([^\]]+)\]', text)
    if match:
        entities = [entity.strip() for entity in match.group(1).split(',')]
        return entities
    else:

        list_match = re.search(r'List:\s*(.*)', text)
        if list_match:
            entities = [entity.strip() for entity in list_match.group(1).split(',')]
            return entities
        else:
            return []


def alt_E(question, E):

    alt_entity = extract_entities(Entity_extract(E, question))
    n = len(alt_entity)
    if n < 2:
        return [], []

    alt_gen_entity = extract_entities(AltEntity_extract(alt_entity))

    weights = [1.0 - i * 0.1 for i in range(n)]

    k = n - 1
    select_entity, Prob, Sort = calculate_combination_probability(weights, alt_gen_entity, k)

    ALT_E = []
    for i in range(len(Sort)):
        original_subset = [alt_entity[idx] for idx in Sort[i]]
        new_subset = select_entity[i]
        alt_e = replace_keywords(E, original_subset, new_subset)
        ALT_E.append(alt_e)

    return ALT_E, Prob




if __name__ == "__main__":
    question = "Compounds that are capable of accepting electrons, such as o 2 or f2, are called what?"
    E='''
    Oxidants and Reductants Compounds that are capable of accepting electrons, such as O 2 or F2, are calledoxidants (or oxidizing agents) because they can oxidize other compounds.
    In the process of accepting electrons, an oxidant is reduced. Compounds that are capable of donating electrons, such as sodium metal or cyclohexane (C6H12),
    are calledreductants (or reducing agents) because they can cause the reduction of another compound. In the process of donating electrons, a reductant is oxidized.
    These relationships are summarized in Equation 3.30: Equation 3.30 Saylor URL: http://www. saylor. org/books.'''
    alt_e, probs = alt_E(question, E)
    print("Constructed texts:", alt_e)
    print("Probabilities:", probs)
    print("Sum of probabilities:", sum(probs))
