# EditLens

This is the accompanying repository for the ICLR 2026 paper [EditLens: Quantifying the Extent of AI Editing in Text](https://arxiv.org/abs/2510.03154), which is the first paper to formalize the task of scoring text according to the extent of AI intervention in the text, as opposed to prior work that treated AI text detection as a binary (or occasionally ternary) classification task.

## Links

- **Paper:** [arXiv:2510.03154](https://arxiv.org/abs/2510.03154)
- **Models:** [pangram/models on HuggingFace](https://huggingface.co/pangram/models)
- **Dataset:** [pangram/editlens_iclr on HuggingFace](https://huggingface.co/datasets/pangram/editlens_iclr)

## Setup

```bash
pip install -r requirements.txt
```

## Training

Configs for both models are in `configs/`. The effective batch size for both models is 24.

### RoBERTa-Large (Single GPU)

```bash
python scripts/train.py -cn roberta
```

### Llama-3.2-3B QLoRA (8 GPUs)

Note that per-device batch size is 3 across 8 GPUs for an effective batch size of 24. Adjust if you are using fewer than 8 GPUs.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 scripts/train.py -cn llama
```

## Inference

Run inference on any HuggingFace dataset (remote or local). The script adds two columns to the output:
- `bucket_pred`: predicted bucket (int)
- `score_pred`: continuous score in [0, 1], the expected value of the bucket distribution

```bash
python scripts/inference.py \
  --checkpoint pangram/editlens_roberta-large \
  --model_name FacebookAI/roberta-large \
  --max_length 512 \
  --dataset pangram/editlens_iclr \
  --split test \
  --text_col text \
  --output predictions.jsonl
```

You can train an EditLens model with any number of classification buckets! This script will infer the num_buckets hyperparameter automatically from the model checkpoint.

## Citation
If you use the code, dataset, or models mentioned in this repository, please cite our paper as follows:

```bibtex
@misc{thai2025editlensquantifyingextentai,
      title={EditLens: Quantifying the Extent of AI Editing in Text},
      author={Katherine Thai and Bradley Emi and Elyas Masrour and Mohit Iyyer},
      year={2025},
      eprint={2510.03154},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.03154},
}
```

## License

This work is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
