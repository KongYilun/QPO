import openai
import os
import re
import math
from concurrent.futures import ThreadPoolExecutor
from evaluation.math_equivalence import *
from requests.exceptions import RequestException
import time
####3.5: sk-uINRZAXtNohaulzOC6C46294B81341Df862b5fD8C85325D9
####4: sk-F1e0GxsRv8OzwosJ17FeBcE53b26434bA38a6d077c8536D2

from openai import OpenAI
from evaluation.nlp_data.gsm8k import GSM8K


def extract_gsm8k_golden(input_string):
    match = re.search(r'#### (\d+(\.\d+)?)', input_string)
    if match:
        return match.group(1)
    else:
        print('Error! NO Extracted golden answer!')

def call_gpt4_api(client,prompt,instruction):
    new_prompt=f"{prompt}\n{instruction}"
    max_try=5
    for i in range(max_try):
        retries = 0
        while retries < 20 :
            try:
                completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "The final answer should be expressed as a single number, prefixed by 'Final Answer: '."},#
                    {"role": "user", "content": new_prompt}
                ],
                logprobs=True,
                temperature=0,
                top_p=1,
                n=1
                )
                break
            except (openai.OpenAIError, openai.InternalServerError, RequestException) as e:
                print(f"Error, Retrying in 0.1 seconds...")
                time.sleep(0.1)
                retries += 1
        if retries==20:
            print('Totally Wrong!')

        if completion.choices[0].logprobs!=None:
            break
    
    full_result=completion.choices[0].message.content
    pred=completion.choices[0].message.content
    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    if len(pred)==0:
        clean_result=None
    else:
        clean_result=pred[-1]
        if clean_result[-1] == '.':
            clean_result = clean_result[:-1] 
        if '.' in clean_result:
            integer_part, decimal_part = clean_result.split('.')
            if decimal_part == '0' * len(decimal_part):
                clean_result=integer_part

    log_probs=[]

    if completion.choices[0].logprobs!=None:
        for i in completion.choices[0].logprobs.content:
            log_probs.append(i.logprob)

        log_prob_sum = sum(log_probs)
        mean_log_prob = log_prob_sum / len(log_probs)
        perplexity = -mean_log_prob
    else:
        perplexity=None

    final_answer_perplexity=0
    return full_result,perplexity,clean_result,final_answer_perplexity

def gsm8k_eval_per_instance(client,prompt,full_answer,instruction):
    full_result,perplexity,clean_result,clean_perplexity=call_gpt4_api(client,prompt,instruction)
    clean_answer=extract_gsm8k_golden(full_answer)
    new_clean_result,correctness=is_equiv(clean_result,clean_answer)

    return full_result,new_clean_result,perplexity,correctness,prompt,full_answer,clean_answer,clean_perplexity,instruction

def svamp_eval_per_instance(client,prompt,full_answer,instruction):
    full_result,perplexity,clean_result,clean_perplexity=call_gpt4_api(client,prompt,instruction)
    clean_answer=full_answer
    new_clean_result,correctness=is_equiv(clean_result,clean_answer)

    return full_result,new_clean_result,perplexity,correctness,prompt,full_answer,clean_answer,clean_perplexity,instruction


