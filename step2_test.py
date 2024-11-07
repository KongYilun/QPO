import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import argparse
from model.pgdt import PGDT
import openai
from evaluation.nlp_data import *
from evaluation.target_models import *
from evaluation.utils import *
from evaluation.templates import *
from evaluation.decoders import *
from evaluation.openai_evaluate import *
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import json

def evaluate(model, dataset, dataset_name, target_model, rtg, args):
    model.eval()
    seed=args.seed
    mode=args.generate_mode
    metric=args.metric
    print('Mode: ', args.test_mode)
    if dataset.name=='gsm8k' or dataset.name=='svamp':
        num_test_instances= 200
        querys = dataset.sample_instances(f"{args.test_mode}", num_test_instances, seed=seed, max_words=False)
    else:
        num_test_instances= 100 ##len(dataset.splits['train'])
        querys = dataset.sample_instances(f"{args.test_mode}", num_test_instances, seed=seed)
    
    
    
    if dataset.name == "ag_news":
        query_texts=[q['text'] for q in querys]
    if dataset.name == "anli":
        query_texts=["[Hypothesis]: "+q["hypothesis"]+" [Premise]: "+q["premise"] for q in querys]
    if dataset.name == "boolq":
        query_texts=["[Question]: "+q['question']+" [Passage]: "+q['passage'] for q in querys]
    if dataset.name == "imdb":
        query_texts=[q['text'] for q in querys]
    if dataset.name == "tweet_emotion":
        query_texts=[q['text'] for q in querys]
    if dataset.name == "cosmos_qa":
        query_texts=["[Question]: "+q['question']+" [Context]: "+q['context'] for q in querys]
    if dataset.name == "hellaswag":
        query_texts=["[Activity_Label]: "+q['activity_label']+" [Context]: "+q['ctx'] for q in querys]
    if dataset.name == 'gsm8k':
        query_texts=[q['question'] for q in querys]
    if dataset.name == 'svamp':
        query_texts=[q['question_concat'] for q in querys]
    
    demonstrations_list = []
    num_combinations=10
    num_demonstrations=6
    if metric!='zero_shot':
        for demo_seed in range(num_combinations):
            demonstration_instances = dataset.sample_instances("train", num_demonstrations, seed=demo_seed)
            demonstrations_list.append(demonstration_instances)

    rtg=[rtg]*len(query_texts)
    zsa_metric_name, zsa_metric_config=get_metric_name_config('evaluation/configs/metric/query_based_zero_shot_accuracy_defaults.yaml')
    fsa_metric_name, fsa_metric_config=get_metric_name_config('evaluation/configs/metric/query_based_few_shot_accuracy_defaults.yaml')
    with torch.no_grad():
        if mode=='sample':
            instruction=model.generate_sample(query_texts, rtg, 2, 0.9)
        elif mode=='greedy':
            print(mode)
            instruction=model.generate_greedy(query_texts, rtg)
        else:
            print('Wrong Mode')
    instruction=instruction['sample_tokens']
    instruction_list=[''.join(sublist).strip() for sublist in instruction]
    instruction_list=[instruct.replace('<|endoftext|>','') for instruct in instruction_list]
    save_dic={'query':querys,'generated_prompt':instruction_list}
    with open(f'test_round{args.round}_seed{seed}.json','w') as file:
        json.dump(save_dic,file,indent=2)
    exit()
    for i in range(len(set(instruction_list))):
        print(list(set(instruction_list))[i])
    # print(instruction_list)
    torch.cuda.empty_cache()
    if dataset.name=='gsm8k' or dataset.name=='svamp':
        if dataset.name=='gsm8k':
            q_k,a_k='question','answer'
            eval_func=gsm8k_eval_per_instance
        if dataset.name=='svamp':
            q_k,a_k='question_concat','Answer'
            eval_func=svamp_eval_per_instance
        os.environ["OPENAI_API_KEY"] ='sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
        # os.environ["OPENAI_BASE_URL"] = "https://xiaoai.plus/v1"
        client = OpenAI()
        avg_list=[]
        results=[]
        with ThreadPoolExecutor(max_workers=100) as executor:
        # 提交所有提示词处理任务到线程池
            future_to_prompt = {executor.submit(eval_func, client,instance[q_k],instance[a_k],instruction): (instance[q_k],instance[a_k],instruction) for instance,instruction in zip(querys,instruction_list)}
            
            # 获取每个任务的结果
            for future in future_to_prompt:
                full_result,clean_result,perplexity,correctness,prompt,full_answer,clean_answer,clean_perplexity,instruct = future.result()
                avg_list.append(correctness)
                new_dic={'query':prompt,
                        'gt_answer':clean_answer,
                        'prompt':instruct,
                        'GPT-4o':{'zero_shot_answer':clean_result,
                                'full_answer':full_result,
                                'zero_shot_perplexity':perplexity,
                                # 'answer_perplexity':clean_perplexity,
                                'zero_shot_correct':correctness}}
                results.append((new_dic))
        
        avg_acc=sum(avg_list)/len(avg_list)
        print("zero_shot_accuracy:",avg_acc)
        
    else:
        prompt_template_dir='evaluation/configs/default_prompts'
        prompt_template = QueryBasedInstructionBasedFewShotTemplate(
                    jinja2_file_path=os.path.join(prompt_template_dir, dataset_name + ".j2")
                )
        decoder_name = query_based_decoder_name(dataset.task_type)
        decoder = get_decoder(decoder_name, prompt_template, dataset)

        with torch.no_grad():
            if metric=='zero_shot':
                metric = get_metric(zsa_metric_name, target_model, dataset, prompt_template, decoder, zsa_metric_config)
                zsa_results = metric.evaluate(instruction_list, querys)
                print("zero_shot_accuracy:",zsa_results["zero_shot_accuracy"])
            elif metric=='few_shot':
                metric = get_metric(fsa_metric_name, target_model, dataset, prompt_template, decoder, fsa_metric_config)
                inputs=(demonstrations_list, querys)
                fsa_results = metric.evaluate(instruction_list, inputs)
                print("few_shot_accuracy:",fsa_results["few_shot_accuracy"])
            elif metric=='both':
                metric = get_metric(zsa_metric_name, target_model, dataset, prompt_template, decoder, zsa_metric_config)
                zsa_results = metric.evaluate(instruction_list, querys)
                print("zero_shot_accuracy:",zsa_results["zero_shot_accuracy"])
                metric = get_metric(fsa_metric_name, target_model, dataset, prompt_template, decoder, fsa_metric_config)
                inputs=(demonstrations_list, querys)
                fsa_results = metric.evaluate(instruction_list, inputs)
                print("few_shot_accuracy:",fsa_results["few_shot_accuracy"])
            else:
                print('Wrong task mode!')
                return

    #

def test(args):
    print(f'\n #########  Start Testing Round {args.round}  #########\n')
    device=args.device
    if args.round==1:
        train_mode='scratch'
        i_train_iter=100
    else:
        train_mode='finetune'
        i_train_iter=20
    model_path=f'{args.save_dir}/{args.task}_{args.data_component}_round{args.round}_{train_mode}_{args.metric}_seed{args.model_seed}.pth'
    print('seed:', args.seed)
    
    target_model=args.target_model
    print('target model:\t',target_model)
    dataset_name=args.task
    print('task:\t',dataset_name)
    print('metric:\t',args.metric)
    print('tested model:\t',model_path)
    model=torch.load(model_path).to(device)
    expected_rtg=args.expected_rtg
    print('Expected RTG:',expected_rtg)
    # target_model = get_model(target_model)
    nlp_dataset = get_dataset(dataset_name)  
    evaluate(model,nlp_dataset,dataset_name,target_model, expected_rtg,  args)         
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, required=False, default="cuda")
    parser.add_argument("--round", type=int, required=False, default=1)
    parser.add_argument("--seed", type=int, required=False, default=0)
    parser.add_argument("--model_seed", type=int, required=False, default=0)
    parser.add_argument("--target_model", type=str, required=False, default="llama7b", help="Model name")
    parser.add_argument("--task", type=str, required=False, default="ag_news")
    parser.add_argument("--metric", type=str, required=False, default="zero_shot")
    parser.add_argument("--expected_rtg", type=int, required=False, default=100)
    parser.add_argument("--data_component", type=str, required=False)
    parser.add_argument("--save_dir", type=str, required=False)
    parser.add_argument("--test_mode", type=str, required=False)
    parser.add_argument("--generate_mode", type=str, required=False, default='greedy')
    args = parser.parse_args()
    test(args)