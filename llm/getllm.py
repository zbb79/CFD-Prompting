from openai import OpenAI
import pandas as pd
import os
import re
import asyncio

os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = ""

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)
Module=""
def extract_cot_and_answer(response: str):
    pattern = r'\[([^\[\]]+)\]'
    match = re.search(pattern, response)
    
    if match:

        answer = match.group(1)
    else:

        pattern = r'the final answer is\s*:?\s*([^.]+)'
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
        else:
            answer = None

    return answer



def generate_cot(E,t,M):
    prompt=f'''
     Based on the context, answer the question step by step and provide the final answer in the end enclosed in square brackets.
    '''
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"context:\n{E}question:{t}\n"}
    ]
    response = client.chat.completions.create(
          model=Module,
          messages=messages,
          n=M,
          top_p=0.9,
          temperature=0.7
          )
    Path=[]
    Answer=[]
    for i in range(M):
        path=response.choices[0].message.content.strip()
        Path.append(path)
        answer=extract_cot_and_answer(path)
        Answer.append(answer)
    return Path,Answer

def Entity_extract(E,question):
    prompt = f'''
    Based on the input question, you need to extract 5 entities from the provided text that are most likely to be used to answer the question. The entities should be listed in descending order of relevance to the question.
    '''
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"context:{E}question:{question}\n"}
    ]
    response = client.chat.completions.create(
        model=Module,
        messages=messages,
        top_p=0.9,
        temperature=0.7
    )
    triplets = response.choices[0].message.content.strip()
    return triplets


def AltEntity_extract(entity):
    prompt = f'''
    Based on the input list, sequentially replace the entities in the list with entities that are similar in meaning while maintaining the same form as the original entities. Finally, return the result in the form of a list.

    '''
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"input:{entity}\noutput:"}
    ]
    response = client.chat.completions.create(
        model=Module,
        messages=messages,
        top_p=0.9,
        temperature=0.7
    )
    triplets = response.choices[0].message.content.strip()
    return triplets






