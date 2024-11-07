from typing import Tuple, Dict, Any
import argparse
import numpy as np
import json

def statistics(data):
    _p_l=[]
    _q_l=[]
    for i in data:
        _q_l.append(i['query'])
        _p_l.append(i['prompt'])
    return len(set(_q_l)),len(set(_p_l))

def merge(args):
    print(f'\n #########  Start Merging Dataset for Round {args.round+1} #########\n')
    new_path=f'data/{args.task}/{args.task}_{args.target_model}_{args.data_component}_round{args.round}_collected_{args.metric}_seed{args.seed}.json'
    if args.round==1:
        past_path=f'data/{args.task}/{args.task}_{args.target_model}_{args.data_component}_round{args.round}_{args.metric}_expert.json'
    else:
        past_path=f'data/{args.task}/{args.task}_{args.target_model}_{args.data_component}_round{args.round}_{args.metric}_expert_seed{args.seed}.json'
    with open(new_path,'r') as file:
        _new_data=json.load(file)
    with open(past_path,'r') as file:
        past_data=json.load(file)
    with open('data/config.json','r') as file:
        config=json.load(file)
    expert_list=[]
    zero_shot_avg_acc=0
    zero_shot_correct_list=[]
    few_shot_avg_acc=0
    few_shot_perturbed_avg_acc=0
    if args.task=='gsm8k' or args.task=='svamp':
        for i in _new_data:
            if args.metric=='zero_shot':
                zero_shot_correct_list.append(i['GPT-4o']['zero_shot_correct'])
        zero_shot_avg_acc=sum(zero_shot_correct_list)/len(zero_shot_correct_list)
    else:
        for i in _new_data:
            if args.metric=='zero_shot':
                zero_shot_avg_acc+=i[args.target_model]['zero_shot_avg_acc']
            elif args.metric=='few_shot':
                few_shot_avg_acc+=i[args.target_model]["few_shot_avg_acc"]
                # few_shot_perturbed_avg_acc+=i[args.target_model]["few_shot_perturbed_avg_acc"]
            else:
                pass
            # zero_shot_avg_acc+=i[args.target_model]['zero_shot_avg_acc']
            # few_shot_avg_acc+=i[args.target_model]["few_shot_avg_acc"]
            # few_shot_perturbed_avg_acc+=i[args.target_model]["few_shot_perturbed_avg_acc"]
        zero_shot_avg_acc/=len(_new_data)
        few_shot_avg_acc/=len(_new_data)
        few_shot_perturbed_avg_acc/=len(_new_data)

    # for i in _new_data:
    #     if args.metric=='zero_shot':
    #         i[args.target_model]['zero_shot_avg_acc']=zero_shot_avg_acc
    #     elif args.metric=='few_shot':
    #         i[args.target_model]["few_shot_avg_acc"]=few_shot_avg_acc
    #         # i[args.target_model]["few_shot_perturbed_avg_acc"]=few_shot_perturbed_avg_acc
    #     else:
    #         pass
            # i[args.target_model]['zero_shot_avg_acc']=zero_shot_avg_acc
            # i[args.target_model]["few_shot_avg_acc"]=few_shot_avg_acc
            # i[args.target_model]["few_shot_perturbed_avg_acc"]=few_shot_perturbed_avg_acc
    
    new_=[]
    for i in _new_data:
        if i not in new_:
            new_.append(i)
    _new_data=new_

    for i in _new_data:
        if args.metric=='zero_shot':
            if args.task=='gsm8k' or args.task=='svamp':
                rtg=i['GPT-4o']["zero_shot_correct"]*10+i['GPT-4o']['zero_shot_avg_acc']*10-i['GPT-4o']["zero_shot_perplexity"]
            else:
                rtg=i[args.target_model]["zero_shot_correct"]*10+i[args.target_model]['zero_shot_avg_acc']*10-i[args.target_model]["zero_shot_perplexities"][i["label"]]
        elif args.metric=='few_shot':
            fs=(i['llama7b']["few_shot_accuracy"]+i['llama7b']["few_shot_avg_acc"])*10
            rtg=fs
        else:
            pass
            # zs=i[args.target_model]["zero_shot_correct"]*10-i[args.target_model]["zero_shot_perplexities"][i["label"]]+i[args.target_model]['zero_shot_avg_acc']*10
            # fs=(i['llama7b']["few_shot_accuracy"]+i['llama7b']["few_shot_perturbed_accuracy"]+i['llama7b']["few_shot_perturbed_avg_acc"]+i['llama7b']["few_shot_avg_acc"])*10/4
            # sensitivity_metric=(i['llama7b']["few_shot_demo_selectional_sensitivity"]+i['llama7b']["few_shot_query_sensitivity"]+i['llama7b']["few_shot_demo_permutational_sensitivity"])*10/3
            # rtg=zs+fs-sensitivity_metric
        if rtg>config[args.task][args.metric]["min"]:
            expert_list.append(i)

    q_past,p_past=statistics(past_data)
    q_new,p_new=statistics(expert_list)
    print('original querys:',q_past,'original prompts:',p_past)
    print('new querys:',q_new,'new prompts:',p_new)
    data=past_data+expert_list
    q_merge,p_merge=statistics(data)
    print('merged querys:',q_merge,'merged prompts:',p_merge)
    save_path=f'data/{args.task}/{args.task}_{args.target_model}_{args.data_component}_round{args.round+1}_{args.metric}_expert_seed{args.seed}.json'
    with open(save_path,'w') as file:
        json.dump(data,file,indent=2)
    print('saved new dataset:',save_path)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model", default='llama7b', type=str, help="Model name")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--round", default=1, type=int, help="Model name")
    parser.add_argument("--task", default='anli', type=str, help="Dataset name")
    parser.add_argument("--metric", default='zero_shot', type=str, help="Metric config file")
    parser.add_argument("--data_component", type=str, required=False, default='ocm')
    args = parser.parse_args()
    merge(args)