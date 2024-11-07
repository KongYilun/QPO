import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer

from model.utils import _top_k_logits, _top_p_logits

class PGDT(nn.Module):


    def __init__(
            self,
            hidden_size=768,
            q_max_length=40,
            p_max_length=40,
            frozen=False,
            device='cpu',

    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.device=device
        self.q_max_length=q_max_length
        self.p_max_length=p_max_length
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.model=GPT2LMHeadModel.from_pretrained("gpt2")
        if frozen==True:
            for param in self.model.transformer.parameters():
                param.requires_grad = False
        #self.model.transformer.config.pad_token_id = self.model.transformer.config.eos_token_id
        self.tokenizer=GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.embed_component = nn.Embedding(3, hidden_size)
        #nn.Embedding(vocab_size, hidden_size)
        self.embed_token = self.model.transformer.wte
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.predict_return = torch.nn.Linear(hidden_size, 1)
        self.mlp_layer=torch.nn.Linear(hidden_size, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.rtg_loss=nn.MSELoss()
        self.prompt_loss=nn.CrossEntropyLoss()

    def _predict_prompt_forward(self, hidden_state):
        x=self.mlp_layer(hidden_state)
        x=F.relu(x)
        logits = self.model.lm_head(x)
        return logits

    def forward(self, questions, prompts, returns_to_go):####prompts, answers,

        # embed each modality with a different head
        component_r_embeddings = self.embed_component(torch.tensor(0, dtype=torch.long, device=self.device))
        component_q_embeddings = self.embed_component(torch.tensor(1, dtype=torch.long, device=self.device))
        component_p_embeddings = self.embed_component(torch.tensor(2, dtype=torch.long, device=self.device))
        tk_q=self.tokenizer(questions, padding="max_length", max_length=self.q_max_length, truncation=True, return_tensors="pt").to(self.device)
        tk_p=self.tokenizer(prompts, padding="max_length", max_length=self.p_max_length, truncation=True, return_tensors="pt").to(self.device)
        # print(tk_q['input_ids'][:5,-1])
        batch_size, seq_length = tk_q['input_ids'].shape[0], tk_q['input_ids'].shape[1]
        returns_to_go=torch.tensor(returns_to_go, dtype=torch.float32, device=self.device).reshape(batch_size,1)
        returns_embeddings = self.embed_return(returns_to_go).unsqueeze(1)
        question_embeddings = self.embed_token(tk_q['input_ids'])
        prompt_embeddings = self.embed_token(tk_p['input_ids'])

        # component embedding
        returns_embeddings = returns_embeddings + component_r_embeddings
        question_embeddings = question_embeddings + component_q_embeddings
        prompt_embeddings = prompt_embeddings + component_p_embeddings

        # input embedding
        stacked_inputs = torch.cat((returns_embeddings, question_embeddings, prompt_embeddings), dim=1)     
        stacked_inputs = self.embed_ln(stacked_inputs)

        # attention mask 
        attention_mask_r = torch.ones((batch_size, 1), dtype=torch.long, device=self.device)
        stacked_attention_mask = torch.cat((attention_mask_r, tk_q['attention_mask'], tk_p['attention_mask']),dim=1)       
        # get generation cache
        transformer_outputs = self.model.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            use_cache=True
        )

        ####RTG Loss
        rtg_weight=transformer_outputs.last_hidden_state[:,0]
        rtg_pred=self.predict_return(rtg_weight)
        rtg_loss=self.rtg_loss(rtg_pred,returns_to_go)
        ####last_token_hidden_state = outputs.last_hidden_state[np.arange(input_ids.shape[0]), (input_lengths - 1)]  ###RLPrompt实现方式
        #rtg_weight=transformer_outputs.last_hidden_state[:,0].unsqueeze(1).expand(-1, self.p_max_length, self.hidden_size)
        last_hidden_state = transformer_outputs.last_hidden_state[:,self.q_max_length:self.q_max_length+self.p_max_length]#+rtg_weight
        # print('forward:\n', transformer_outputs.last_hidden_state[:5,self.q_max_length,:5])
        logits=self._predict_prompt_forward(last_hidden_state)
        
        mask=tk_p['attention_mask'].bool()
        logits=logits[mask]
        labels=tk_p['input_ids'][mask]
        
        prompt_loss=self.prompt_loss(logits, labels)
        if rtg_loss < 10:
            rate=0.5
        else:
            rate=0.1
        loss = prompt_loss + rate*rtg_loss
        return loss, logits, prompt_loss, rate*rtg_loss

    def generate_sample(self, questions, returns_to_go, top_k, top_p):####prompts, answers,

        # embed each modality with a different head
        component_r_embeddings = self.embed_component(torch.tensor(0, dtype=torch.long, device=self.device))
        component_q_embeddings = self.embed_component(torch.tensor(1, dtype=torch.long, device=self.device))
        component_p_embeddings = self.embed_component(torch.tensor(2, dtype=torch.long, device=self.device))
        tk_q=self.tokenizer(questions, padding="max_length", max_length=self.q_max_length, truncation=True, return_tensors="pt").to(self.device)

        batch_size, seq_length = tk_q['input_ids'].shape[0], tk_q['input_ids'].shape[1]
        returns_to_go = torch.tensor(returns_to_go, dtype=torch.float32, device=self.device).reshape(batch_size,1)

        returns_embeddings = self.embed_return(returns_to_go).unsqueeze(1)
        question_embeddings = self.embed_token(tk_q['input_ids'])

        # component embedding
        returns_embeddings = returns_embeddings + component_r_embeddings
        question_embeddings = question_embeddings + component_q_embeddings

        # input embedding
        stacked_inputs = torch.cat((returns_embeddings, question_embeddings),dim=1)     
        stacked_inputs = self.embed_ln(stacked_inputs)

        # attention mask 
        attention_mask_r = torch.ones((batch_size, 1), dtype=torch.long, device=self.device)
        stacked_attention_mask = torch.cat((attention_mask_r,tk_q['attention_mask']),dim=1)       

        # get generation cache
        transformer_outputs = self.model.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            use_cache=True
        )

        ####last_token_hidden_state = outputs.last_hidden_state[np.arange(input_ids.shape[0]), (input_lengths - 1)]  ###RLPrompt实现方式
        last_token_hidden_state = transformer_outputs.last_hidden_state[:,-1]  ## + 
        
        #rtg_weight=transformer_outputs.last_hidden_state[:,0]
        past_key_values = transformer_outputs.past_key_values

        sample_ids, sample_logits = [], []
        sample_tokens = [[] for _ in range(batch_size)]
        for _ in range(self.p_max_length):
            #last_token_hidden_state=last_token_hidden_state+rtg_weight

            logits = self._predict_prompt_forward(last_token_hidden_state)
            if top_k is not None:
                logits=_top_k_logits(logits, k=top_k)
            if top_p is not None:
                logits = _top_p_logits(logits, p=top_p)

            ids = (torch.distributions.categorical.Categorical(logits=logits).sample())  ##batch_size * 1
            tokens = [self.tokenizer.decode([a]) for a in ids.tolist()]
            #token_strs = [self.tokenizer.convert_tokens_to_string([t]) for t in tokens]
            
            for s,t in zip(sample_tokens, tokens):
                s.append(t)
            sample_ids.append(ids.unsqueeze(1))
            sample_logits.append(logits.unsqueeze(1))

            current_prompt_embedding=self.embed_token(ids).unsqueeze(1) + component_p_embeddings
            current_prompt_embedding=self.embed_ln(current_prompt_embedding)

            transformer_outputs = self.model.transformer(
                inputs_embeds=current_prompt_embedding,
                past_key_values=past_key_values,
                use_cache=True
            )

            last_token_hidden_state = transformer_outputs.last_hidden_state[:,-1]
            past_key_values = transformer_outputs.past_key_values

        # [batch_size, prompt_length]
        sample_ids = torch.cat(sample_ids, dim=1)
        # [batch_size, prompt_length, vocab_size]
        sample_logits = torch.cat(sample_logits, dim=1)

        #sample_sentence=
        output = dict(sample_tokens=sample_tokens,
                      sample_logits=sample_logits,
                      sample_ids=sample_ids)
        return output

    def generate_greedy(self, questions, returns_to_go):####prompts, answers,

        # embed each modality with a different head
        component_r_embeddings = self.embed_component(torch.tensor(0, dtype=torch.long, device=self.device))
        component_q_embeddings = self.embed_component(torch.tensor(1, dtype=torch.long, device=self.device))
        component_p_embeddings = self.embed_component(torch.tensor(2, dtype=torch.long, device=self.device))
        tk_q=self.tokenizer(questions, padding="max_length", max_length=self.q_max_length, truncation=True, return_tensors="pt").to(self.device)
        # print(tk_q['input_ids'][:5,-1])
        batch_size, seq_length = tk_q['input_ids'].shape[0], tk_q['input_ids'].shape[1]
        returns_to_go = torch.tensor(returns_to_go, dtype=torch.float32, device=self.device).reshape(batch_size,1)

        returns_embeddings = self.embed_return(returns_to_go).unsqueeze(1)
        question_embeddings = self.embed_token(tk_q['input_ids'])

        # component embedding
        returns_embeddings = returns_embeddings + component_r_embeddings
        question_embeddings = question_embeddings + component_q_embeddings

        # input embedding
        stacked_inputs = torch.cat((returns_embeddings, question_embeddings),dim=1)     
        stacked_inputs = self.embed_ln(stacked_inputs)

        # attention mask 
        attention_mask_r = torch.ones((batch_size, 1), dtype=torch.long, device=self.device)
        stacked_attention_mask = torch.cat((attention_mask_r,tk_q['attention_mask']),dim=1)       
        # get generation cache
        transformer_outputs = self.model.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            use_cache=True
        )
        ####last_token_hidden_state = outputs.last_hidden_state[np.arange(input_ids.shape[0]), (input_lengths - 1)]  ###RLPrompt实现方式
        last_token_hidden_state = transformer_outputs.last_hidden_state[:,-1]  ## + 
        past_key_values = transformer_outputs.past_key_values
        sample_ids, sample_logits = [], []
        sample_tokens = [[] for _ in range(batch_size)]
        for i in range(self.p_max_length):
            #last_token_hidden_state=last_token_hidden_state+rtg_weight
            logits = self._predict_prompt_forward(last_token_hidden_state)
            ids = logits.argmax(dim=-1)  ##batch_size * 1
            tokens = [self.tokenizer.decode([a]) for a in ids.tolist()]
            #token_strs = [self.tokenizer.convert_tokens_to_string([t]) for t in tokens]
            
            for s,t in zip(sample_tokens, tokens):
                s.append(t)
            sample_ids.append(ids.unsqueeze(1))
            sample_logits.append(logits.unsqueeze(1))

            current_prompt_embedding=self.embed_token(ids).unsqueeze(1) + component_p_embeddings
            current_prompt_embedding=self.embed_ln(current_prompt_embedding)
            
            transformer_outputs = self.model.transformer(
                inputs_embeds=current_prompt_embedding,
                past_key_values=past_key_values,
                use_cache=True
            )

            last_token_hidden_state = transformer_outputs.last_hidden_state[:,-1]
            past_key_values = transformer_outputs.past_key_values

        # [batch_size, prompt_length]
        sample_ids = torch.cat(sample_ids, dim=1)
        # [batch_size, prompt_length, vocab_size]
        sample_logits = torch.cat(sample_logits, dim=1)

        #sample_sentence=
        output = dict(sample_tokens=sample_tokens,
                      sample_logits=sample_logits,
                      sample_ids=sample_ids)
        return output



        