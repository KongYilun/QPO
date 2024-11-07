from typing import Tuple, Dict, Any
import argparse
import torch
import sys
import os
import numpy as np
import yaml
import datasets
from evaluation.nlp_data import *
from evaluation.decoders import *
from evaluation.metrics import *
from evaluation.target_models import *
from evaluation.templates import *
from evaluation.utils import *
import time
from evaluation.metrics.utils import quasi_exact_match
import json
import random
import copy
from evaluation.openai_evaluate import *
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

def get_metric_name_config(metric_config) -> Tuple[str, Dict[str, Any]]:
    with open(metric_config, "r") as f:
        metric_config = yaml.safe_load(f)
        metric_name = list(metric_config.keys())[0]
    return metric_name, metric_config


def collect(args):
    print(f'\n #########  Start Collecting New Dataset for Round {args.round+1} #########\n')
    device=args.device
    base_seed=100+args.seed
    if args.round==1:
        train_mode='scratch'
        i_train_iter=100
    else:
        train_mode='finetune'
        i_train_iter=20
    model_path=f'{args.save_dir}/{args.task}_{args.data_component}_round{args.round}_{train_mode}_{args.metric}_seed{args.seed}.pth'
    print('DT model:\t',model_path)
    target_model=args.target_model
    print('target model:\t',target_model)
    dataset_name=args.task
    print('task:\t',dataset_name)
    print('task mode:\t',args.metric)
    model=torch.load(model_path).to(device)
    model.eval()
    expected_rtg=args.expected_rtg
    # target_model = get_model(target_model)
    nlp_dataset = get_dataset(dataset_name)  
    mode=args.generate_mode
    offline_dataset=[]
    q=[]
    p=[]
    print('collect set')
    for i in range(10):
        print('####Collecting times: ',i)
        num_test_instances= 100 ##len(dataset.splits['train'])
        seed=base_seed+i+args.round
        querys = nlp_dataset.sample_instances("collect", num_test_instances, seed=seed, max_words=False)
        if nlp_dataset.name == "ag_news":
            query_texts=[q['text'] for q in querys]
        if nlp_dataset.name == "anli":
            query_texts=["[Hypothesis]: "+q["hypothesis"]+" [Premise]: "+q["premise"] for q in querys]
        if nlp_dataset.name == "boolq":
            query_texts=["[Question]: "+q['question']+" [Passage]: "+q['passage'] for q in querys]
        if nlp_dataset.name == "imdb":
            query_texts=[q['text'] for q in querys]
        if nlp_dataset.name == "tweet_emotion":
            query_texts=[q['text'] for q in querys]
        if nlp_dataset.name == "cosmos_qa":
            query_texts=["[Question]: "+q['question']+" [Context]: "+q['context'] for q in querys]
        if nlp_dataset.name == "hellaswag":
            query_texts=["[Activity_Label]: "+q['activity_label']+" [Context]: "+q['ctx'] for q in querys]
        if nlp_dataset.name == 'gsm8k':
            query_texts=[q['question'] for q in querys]
        if nlp_dataset.name == 'svamp':
            query_texts=[q['question_concat'] for q in querys]
    
        demonstrations_list = []
        num_combinations=5
        num_demonstrations=6
        if args.metric!='zero_shot':
            for demo_seed in range(num_combinations):
                demonstration_instances = nlp_dataset.sample_instances("train", num_demonstrations, seed=demo_seed)
                demonstrations_list.append(demonstration_instances)

        rtg=[expected_rtg]*len(query_texts)
        zsa_metric_name, zsa_metric_config=get_metric_name_config('evaluation/configs/metric/query_based_zero_shot_accuracy_defaults.yaml')
        fsa_metric_name, fsa_metric_config=get_metric_name_config('evaluation/configs/metric/query_based_few_shot_accuracy_defaults.yaml')
        with torch.no_grad():
            if mode=='sample':
                instruction=model.generate_sample(query_texts, rtg, 2, 0.9)
            elif mode=='greedy':
                instruction=model.generate_greedy(query_texts, rtg)
            else:
                print('Wrong Mode')

        instruction=instruction['sample_tokens']
        instruction_list=[''.join(sublist).strip() for sublist in instruction]
        instruction_list=[instruct.replace('<|endoftext|>','') for instruct in instruction_list]


        if nlp_dataset.name=='gsm8k' or nlp_dataset.name=='svamp':
            if nlp_dataset.name=='gsm8k':
                q_k,a_k='question','answer'
            if nlp_dataset.name=='svamp':
                q_k,a_k='question_concat','Answer'
            os.environ["OPENAI_API_KEY"] ='sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
            # os.environ["OPENAI_BASE_URL"] = "https://xiaoai.plus/v1"
            print(os.environ["OPENAI_BASE_URL"])
            client = OpenAI()
            avg_list=[]
            results=[]
            with ThreadPoolExecutor(max_workers=100) as executor:
            # 提交所有提示词处理任务到线程池
                future_to_prompt = {executor.submit(svamp_eval_per_instance, client,instance[q_k],instance[a_k],instruction): (instance[q_k],instance[a_k],instruction) for instance,instruction in zip(querys,instruction_list)}
                
                # 获取每个任务的结果
                for future in future_to_prompt:
                    full_result,clean_result,perplexity,correctness,prompt,full_answer,clean_answer,clean_perplexity, instruct = future.result()
                    avg_list.append(correctness)
                    instruct+='<|endoftext|>'
                    if perplexity==None:
                        continue
                    new_dic={'prompt_method':f"PGDT Round {args.round}",
                            'query':prompt,
                            'gt_answer':clean_answer,
                            'prompt':instruct,
                            'GPT-3.5':{'zero_shot_answer':clean_result,
                                    'full_answer':full_result,
                                    'zero_shot_perplexity':perplexity*10,
                                    # 'answer_perplexity':clean_perplexity,
                                    'zero_shot_correct':correctness}}
                    offline_dataset.append((new_dic))

        else:
            prompt_template_dir='evaluation/configs/default_prompts'
            prompt_template = QueryBasedInstructionBasedFewShotTemplate(
                        jinja2_file_path=os.path.join(prompt_template_dir, dataset_name + ".j2")
                    )
            decoder_name = query_based_decoder_name(nlp_dataset.task_type)
            decoder = get_decoder(decoder_name, prompt_template, nlp_dataset)

            if args.metric=='zero_shot':
                metric = get_metric(zsa_metric_name, target_model, nlp_dataset, prompt_template, decoder, zsa_metric_config)
                zsa_results = metric.evaluate(instruction_list, querys)
            elif args.metric=='few_shot':
                metric = get_metric(fsa_metric_name, target_model, nlp_dataset, prompt_template, decoder, fsa_metric_config)
                inputs=(demonstrations_list, querys)
                fsa_results = metric.evaluate(instruction_list, inputs)
            elif args.metric=='both':
                metric = get_metric(zsa_metric_name, target_model, nlp_dataset, prompt_template, decoder, zsa_metric_config)
                zsa_results = metric.evaluate(instruction_list, querys)
                metric = get_metric(fsa_metric_name, target_model, nlp_dataset, prompt_template, decoder, fsa_metric_config)
                inputs=(demonstrations_list, querys)
                fsa_results = metric.evaluate(instruction_list, inputs)
            else:
                print('Wrong task mode!')
                return
            
            for t in range(num_test_instances):
                log_dict={}
                log_dict['prompt_method']=f"PGDT Round {args.round}"
                if args.task =='ag_news':
                    log_dict['query']=querys[t]['text']
                    log_dict['label']=querys[t]['label']
                if args.task =='anli':
                    log_dict['premise']=querys[t]['premise']
                    log_dict['hypothesis']=querys[t]['hypothesis']
                    log_dict['reason']=querys[t]['reason']
                    log_dict['label']=querys[t]['label']
                    log_dict['query']="[Hypothesis]: "+querys[t]["hypothesis"]+" [Premise]: "+querys[t]["premise"]
                if args.task =='boolq':
                    log_dict['question']=querys[t]['question']
                    log_dict['passage']=querys[t]['passage']
                    log_dict['answer']=querys[t]['answer']
                    log_dict['label']=1 if querys[t]['answer']==False else 0
                    log_dict['query']="[Question]: "+querys[t]['question']+" [Passage]: "+querys[t]['passage']
                if args.task =='imdb':
                    log_dict['query']=querys[t]['text']
                    log_dict['label']=querys[t]['label']
                if args.task =='tweet_emotion':
                    log_dict['query']=querys[t]['text']
                    log_dict['label']=querys[t]['label']
                if args.task =='cosmos_qa':
                    log_dict['context']=querys[t]['context']
                    log_dict['question']=querys[t]['question']
                    log_dict['answer0']=querys[t]['answer0']
                    log_dict['answer1']=querys[t]['answer1']
                    log_dict['answer2']=querys[t]['answer2']
                    log_dict['answer3']=querys[t]['answer3']
                    log_dict['label']=querys[t]['label']
                    log_dict['query']="[Question]: "+querys[t]['question']+" [Context]: "+querys[t]['context']
                if args.task =='hellaswag':
                    log_dict['ctx']=querys[t]['ctx']
                    log_dict['activity_label']=querys[t]['activity_label']
                    log_dict['endings']=querys[t]['endings']
                    log_dict['label']=int(querys[t]['label'])
                    log_dict['query']="[Activity_Label]: "+querys[t]['activity_label']+" [Context]: "+querys[t]['ctx']
                log_dict['prompt']=instruction_list[t]+'<|endoftext|>'
                log_dict[f'{args.target_model}']={}
                if args.metric=='zero_shot':
                    log_dict[f'{args.target_model}']['zero_shot_answer']=int(zsa_results['zero_shot_predicted_outputs'][t][0])           #####对每个query的结果，体现query-conditioned
                    log_dict[f'{args.target_model}']['zero_shot_perplexities']=zsa_results['zero_shot_predicted_outputs'][t][1]     #####对每个query的结果，体现query-conditioned
                    log_dict[f'{args.target_model}']['zero_shot_correct']=zsa_results['zero_shot_correctness_indicators'][t]        #####对每个query的结果，体现query-conditioned
                    log_dict[f'{args.target_model}']['zero_shot_avg_acc']=zsa_results["zero_shot_accuracy"]                         #####对所有query取平均，体现prompt本身的能力
                elif args.metric=='few_shot':
                    log_dict[f'{args.target_model}']['few_shot_accuracy']=fsa_results['per_instance_few_shot_accuracies'][t]
                    log_dict[f'{args.target_model}']['few_shot_avg_acc']=fsa_results['few_shot_accuracy']
                offline_dataset.append(log_dict)

    file_name=f'data/{args.task}/{args.task}_{args.target_model}_{args.data_component}_round{args.round}_collected_{args.metric}_seed{args.seed}.json'    
    with open(file_name, "w", encoding='utf-8') as f:
        json.dump(offline_dataset, f, ensure_ascii=False, indent=2)   
        #f.write("\n")  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, required=False, default="cuda")
    parser.add_argument("--target_model", default='llama7b', type=str, help="Model name")
    parser.add_argument("--round", default=1, type=int, help="Model name")
    parser.add_argument("--generate_mode", type=str, required=False, default='sample')
    parser.add_argument("--data_component", type=str, required=False, default='ocm')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--task", default='anli', type=str, help="Dataset name")
    parser.add_argument("--expected_rtg", type=int, required=False, default=100)
    parser.add_argument("--save_dir", type=str, required=False)
    parser.add_argument("--metric", default='zero_shot', type=str, help="Metric config file")
    parser.add_argument( "--prompt_template_dir", type=str, default="configs/default_prompts", help="Directory containing prompt templates for each dataset")
    args = parser.parse_args()
    collect(args)

        
        

