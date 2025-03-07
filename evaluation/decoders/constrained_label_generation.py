from typing import Tuple, List, Dict, Any, Union, Optional

import numpy as np
import torch

from evaluation.decoders.decoder import Decoder
from evaluation.target_models.causal_lm import CausalLM
from evaluation.templates.few_shot_template import FewShotTemplate


class ConstrainedLabelGeneration(Decoder):
    """
        Decoder that uses the language model to find the lowest perplexity label
        from a static set of labels. Ideal for classification tasks with a fixed,
        known set of labels.

        Assumes the presence of a label_map in the template which maps huggingface
        labels to verbalizer strings. Uses the language model to find the lowest 
        perplexity verbalizer string among this set.
    """

    def __init__(self, template: FewShotTemplate):
        super().__init__(template)

    
    def decode(
        self,
        model: CausalLM,
        demonstrations: List[Dict[str, Any]],
        test_examples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        
        """
            model: model to use for decoding.
            demonstrations: list of in-context demonstrations to use for decoding.
            test_examples: list of test examples to decode.
        """

        def tokenize(text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
            return model.tokenizer(text).input_ids

        # get the huggingface labels and the corresponding verbalizers
        hf_labels = list(self.template.label_map.keys())
        verbalizers = list(self.template.label_map.values())

        # generate prompts for each test example and tokenize them
        prompts = [self.template.render(demonstrations, test_example) for test_example in test_examples]
        prompt_ids = tokenize(prompts)
        
        # get the longest common prefix of the prompts. Contains the few-shot demonstrations.
        lc_prefix_ids = self._longest_common_prefix(prompt_ids)
        past_key_values, past_last_logit = self._get_forward_cache(model, lc_prefix_ids)

        results = []
        for prompt in prompts:
            # candidate_completions correspond to answered test examples
            candidate_answered_prompts = [tokenize(prompt + verbalizer) for verbalizer in verbalizers]
            candidate_completions = [candidate_completion[len(lc_prefix_ids):] for candidate_completion in candidate_answered_prompts]
            
            # find index where the verbalizer begins in the candidate_completions
            label_idx = len(tokenize(prompt.rstrip())) - len(lc_prefix_ids)
            
            # get the perplexities of the verbalizers and compute prediction
            verbalizer_perplexities = self._get_verbalizer_perplexities(
                model,
                candidate_completions,
                label_idx,
                past_key_values,
                past_last_logit,
            )
            prediction = hf_labels[np.argmin(verbalizer_perplexities)]

            results.append({
                "prediction": prediction,
                "perplexities": verbalizer_perplexities,
            })
            
        return results

    def _get_forward_cache(
        self,
        model: CausalLM,
        input_ids: List[int],
    ) -> Tuple[Optional[Tuple[Tuple[torch.FloatTensor]]], Optional[torch.Tensor]]:
        # computes a forward pass on the input_ids, and returns the  
        # corresponding past_key_values and past_last_logit

        if len(input_ids) == 0:
            return None, None
        
        with torch.no_grad():
            input_ids = torch.tensor([input_ids], dtype=int).to(model.device)
            model_output = model.hf_model.forward(
                input_ids=input_ids,
                use_cache=True
            )

        past_key_values = model_output["past_key_values"]
        past_last_logit = model_output["logits"][:, -1, :]

        return past_key_values, past_last_logit

    def _get_verbalizer_perplexities(
        self,
        model: CausalLM,
        completions: List[List[int]],
        label_idx: int,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_last_logit: Optional[torch.Tensor] = None,
    ):

        if (past_key_values is None) ^ (past_last_logit is None):
            raise ValueError("Only one of past_key_values and past_last_logit were passed. Expected both or neither.")
        if past_last_logit is None:
            # dummy past_last_logit if it is not passed. (just to get indexing to line up properly)
            past_last_logit = torch.zeros((1, output["logits"].shape[2]), dtype=float).to(model.device)
        
        perplexities = []
        for completion in completions:
            with torch.no_grad():
                input_ids = torch.tensor([completion], dtype=int).to(model.device)
                output = model.hf_model.forward(input_ids=input_ids, past_key_values=past_key_values)
                
                logits = torch.concat([past_last_logit.unsqueeze(1), output["logits"]], axis=1)[0, :-1, :]
                label_ids = input_ids[0, label_idx:]
                label_logits = logits[label_idx:, :]

                probs = torch.softmax(label_logits.to(dtype=torch.float32), dim=-1)
                token_probs = probs[range(len(label_ids)), label_ids]
            perplexities.append(-torch.mean(torch.log(token_probs)).item())
        
        return perplexities

    def _longest_common_prefix(self, id_lists: List[List[int]]):
            if len(id_lists) == 1:
                return id_lists[0]
            ids_sorted = sorted(id_lists)
            first = ids_sorted[0]
            last = ids_sorted[-1]
            for i in range(min(len(first), len(last))):
                if first[i] != last[i]:
                    return first[:i]
            return first if len(first) < len(last) else last

    