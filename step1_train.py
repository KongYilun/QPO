import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import argparse
import json
#from model.pgdt import PGDT
from model.pgdt import PGDT
#from model.pgdt import PGDT
from evaluation.nlp_data import *
from evaluation.target_models import *
from evaluation.utils import *
from evaluation.templates import *
from evaluation.decoders import *
import time


def rtg_norm(min_s,max_s,rtg):
    rtg=(rtg-min_s)/(max_s-min_s)*100
    return rtg



def train(args):
    seed=args.seed
    print('seed:',seed)
    torch.manual_seed(seed)
    device=args.device
    if args.train_mode=='scratch':
        print('initiate a new model')
        model=PGDT(frozen=False,device=device).to(device)
        max_train_iters=100
        optimizer = torch.optim.AdamW(
                            model.parameters(),
                            lr=1e-3,
                            weight_decay=1e-4
                        )
    elif args.train_mode=='finetune':
        if args.round==2:
            path_=f'{args.save_dir}/{args.task}_{args.data_component}_round{args.round-1}_scratch_{args.metric}_seed{seed}.pth'
        else:
            path_=f'{args.save_dir}/{args.task}_{args.data_component}_round{args.round-1}_finetune_{args.metric}_seed{seed}.pth'
        print('load model:',path_)
        model=torch.load(path_)
        max_train_iters=20
        optimizer = torch.optim.AdamW(
                            model.parameters(),
                            lr=1e-4,
                            weight_decay=1e-4
                        )
    else:
        print("No training mode")
        return 
    
    if args.max_train_epochs != None:
        max_train_iters=args.max_train_epochs

    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=20, gamma=0.65)
    if args.round==1:
        data_files=f'data/{args.task}/{args.task}_{args.target_model}_{args.data_component}_round{args.round}_{args.metric}_expert.json'
    else:
        data_files=f'data/{args.task}/{args.task}_{args.target_model}_{args.data_component}_round{args.round}_{args.metric}_expert_seed{args.seed}.json'
    # dataset=load_dataset("json", data_files=data_files)
    with open(data_files,'r') as file:
        dataset=json.load(file)    
    print('offline dataset size:', len(dataset))
    train_loader=DataLoader(dataset, batch_size=128, shuffle=True)
    if args.metric=='zero_shot':
        metric='zero_shot'
    elif args.metric=='few_shot':
        metric='few_shot'
    elif args.metric=='both':
        metric='both'
    else:
        raise Exception('Wrong Metric!')
    print('metric:',metric)
    print('training dataset:',data_files)
    with open('data/config.json','r') as file:
        config=json.load(file)
    min_s=config[args.task][args.metric]["min"]
    max_s=config[args.task][args.metric]["max"]
    print(f'min: {min_s}, max: {max_s}')
    print(f'#########  Start Training Round {args.round} #########')
    i_train_iter=0
    # for i_train_iter in range(1,max_train_iters+1):
    start_time=time.time()
    while i_train_iter<max_train_iters:
        i_train_iter+=1
        total_loss = 0
        total_prompt_loss=0
        total_rtg_loss=0
        total_rate=0
        model.train()
        for iter, dic in enumerate(train_loader):
            if metric=='zero_shot':
                if args.task=='gsm8k' or args.task=='svamp':
                    zero_shot_metric=dic['GPT-4o']["zero_shot_correct"]*10+dic['GPT-4o']['zero_shot_avg_acc']*10-dic['GPT-4o']['zero_shot_perplexity']
                    rtg=zero_shot_metric
                else:
                    zero_shot_perplexity=torch.stack([dic['llama7b']["zero_shot_perplexities"][index][i] for i, index in enumerate(dic['label'])])
                    zero_shot_metric=dic['llama7b']["zero_shot_correct"]*10+dic['llama7b']['zero_shot_avg_acc']*10-zero_shot_perplexity
                    rtg=zero_shot_metric#+few_shot_metric-sensitivity_metric
            elif metric=='few_shot':
                few_shot_metric=(dic['llama7b']["few_shot_accuracy"]+dic['llama7b']["few_shot_avg_acc"])*10   #
                # sensitivity_metric=(dic['llama7b']["few_shot_demo_selectional_sensitivity"]+dic['llama7b']["few_shot_query_sensitivity"]+dic['llama7b']["few_shot_demo_permutational_sensitivity"])*10/3
                rtg=few_shot_metric#-sensitivity_metric
            else:
                zero_shot_perplexity=torch.stack([dic['llama7b']["zero_shot_perplexities"][index][i] for i, index in enumerate(dic['label'])])
                zero_shot_metric=dic['llama7b']["zero_shot_correct"]*10-zero_shot_perplexity+dic['llama7b']['zero_shot_avg_acc']*10#
                few_shot_metric=(dic['llama7b']["few_shot_accuracy"]+dic['llama7b']["few_shot_perturbed_accuracy"]+dic['llama7b']["few_shot_perturbed_avg_acc"]+dic['llama7b']["few_shot_avg_acc"])*10/4   #
                sensitivity_metric=(dic['llama7b']["few_shot_demo_selectional_sensitivity"]+dic['llama7b']["few_shot_query_sensitivity"]+dic['llama7b']["few_shot_demo_permutational_sensitivity"])*10/3
                rtg=zero_shot_metric+few_shot_metric-sensitivity_metric

            if args.reward_norm==True:
                rtg=rtg_norm(min_s,max_s, rtg)
            loss,logits, prompt_loss, rtg_loss=model.forward(dic['query'], dic['prompt'], rtg)
            optimizer.zero_grad()
            total_loss += loss.item()
            total_prompt_loss += prompt_loss.item()
            total_rtg_loss += rtg_loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
        scheduler.step()
        print('Epoch:', i_train_iter, '\t average loss:', total_loss/len(train_loader), '\t prompt loss:', total_prompt_loss/len(train_loader), '\t rtg loss:', total_rtg_loss/len(train_loader))#, '\t rate:', total_rate/len(train_loader))
        if i_train_iter  == max_train_iters:
            _path=f'{args.save_dir}/{args.task}_{args.data_component}_round{args.round}_{args.train_mode}_{args.metric}_seed{seed}.pth'
            if args.max_train_epochs != None:
                _path=f'{args.save_dir}/{args.task}_{args.data_component}_round{args.round}_{args.train_mode}_{args.max_train_epochs}_{args.metric}_seed{seed}.pth'
            print('########  Model Saved  #########')
            print(_path)
            torch.save(model, _path)
    print('Time:',time.time()-start_time)

            
            


            

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, required=False, default="cuda")
    parser.add_argument("--round", type=int, required=True, default=1)
    parser.add_argument("--seed", type=int, required=False, default=0)
    parser.add_argument("--max_train_epochs", type=int, required=False)
    parser.add_argument("--target_model", type=str, required=False, default="llama7b", help="Model name")
    parser.add_argument("--task", type=str, required=False, default="ag_news")
    parser.add_argument("--metric", type=str, required=False, default="zero_shot")
    #parser.add_argument("--dataset", type=str, required=False, default="data/ag_news/ag_news_llama7b_ocm_round1_zs_expert.json", help="Dataset name")
    parser.add_argument("--decoder", type=str, required=False, help="Decoder name")
    parser.add_argument("--reward_norm", action='store_true')
    parser.add_argument("--data_component", type=str, required=False)
    parser.add_argument("--save_dir", type=str, required=False, default="model_saved")
    parser.add_argument("--train_mode", type=str, required=False, default="scratch")
    args = parser.parse_args()
    train(args)