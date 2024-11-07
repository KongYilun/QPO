# QPO: Query-dependent Prompt Optimization via Multi-Loop Offline Reinforcement Learning


## Install

This evaluation suite demands `torch==1.12.1` to ensure compatibility with `crfm_helm`. You will also need `transformers>=4.28.1` to ensure compatibility with the LLaMA models.

Set up a new Python 3.9 environment and install [PyTorch 1.12.1](https://pytorch.org/get-started/previous-versions/#v1121).
```bash
pip install -r requirements.txt --no-deps
```

## Usage

1. Download the task NLP dataset and save it to Path /evaluation/nlp_dataset/ or change the code in /evaluation/nlp_data/{ag_news}.py for example.
2. Modify the path to the target LLM in QPO/evaluation/target_models/llama7b.py.
3. Download the pre-trained GPT-2 model and save as ./gpt2  
4. run training code from step1 to step4.

## Acknowledgments

This repo benefits from [DT](https://github.com/kzl/decision-transformer) and [InstructEval](https://github.com/princeton-nlp/InstructEval). Thanks for their wonderful works!
