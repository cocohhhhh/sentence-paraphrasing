**Description of files**

- eda.ipynb:
  - To check the length distribution of the question pairs.
- t5_finetune.py
  - Use a small subset of the quora dataset to finetune a t5 paraphraser model.
  - The finetuned model can be seen and downloaded at: `coco101010/t5-paraphrase-quora-finetuned`
- paraphrasing.ipynb
  - To create the paraphases using different models. (synonym replacement, t5, finetuned_t5, gpt_t5, gemma3:1b)
- evaluation.ipynb
  - A quick evaluation with BLEU score and cosine similarity.
