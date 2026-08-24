"""Predict EditLens scores for arbitrary text read from stdin.

Single document:
    printf '%s' 'Some text...' | python scripts/predict_text.py \
        --checkpoint pangram/editlens_roberta-large \
        --base_model FacebookAI/roberta-large

Multiple documents may be separated by an ASCII form-feed (\f). Each result
is printed as one JSON object, making the output easy to consume as JSONL.

Add ``--leave_one_sentence_out`` to include sentence-level leave-one-out
attributions. This is substantially slower because each sentence is omitted
and the document is scored again.
"""

import argparse
import json
import sys

import numpy as np
import torch
from scipy.special import softmax
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from inference import infer_n_buckets, is_qlora_checkpoint
from leave_one_out import leave_one_sentence_out, score_from_probabilities
from preprocess import clean_text
from train import NormedLinear


def main() -> None:
    parser = argparse.ArgumentParser(description="Score arbitrary text with EditLens")
    parser.add_argument(
        "--checkpoint",
        default="pangram/editlens_roberta-large",
        help="Checkpoint name or path (default: pangram/editlens_roberta-large)",
    )
    parser.add_argument(
        "--base_model",
        default="FacebookAI/roberta-large",
        help="Base model name (default: FacebookAI/roberta-large)",
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--leave_one_sentence_out",
        action="store_true",
        help="Report how omitting each sentence changes the document score.",
    )
    args = parser.parse_args()

    # Form-feed is intentionally used as the multi-document separator: it is
    # uncommon in prose and does not destroy ordinary paragraph newlines.
    documents = [part.strip() for part in sys.stdin.read().split("\f")]
    documents = [doc for doc in documents if doc]
    if not documents:
        parser.error("No text received on stdin")

    n_buckets = infer_n_buckets(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    qlora = is_qlora_checkpoint(args.checkpoint)
    if qlora and not torch.cuda.is_available():
        raise RuntimeError("This QLoRA checkpoint requires a compatible CUDA device.")

    if qlora:
        from peft import PeftModel
        from transformers import BitsAndBytesConfig

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForSequenceClassification.from_pretrained(
            args.base_model, num_labels=n_buckets, quantization_config=quantization
        )
        base.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(base, "score") and isinstance(base.score, torch.nn.Linear):
            base.score = NormedLinear(
                base.config.hidden_size, n_buckets, device=next(base.parameters()).device
            )
        model = PeftModel.from_pretrained(base, args.checkpoint)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    model.eval()

    training_args = TrainingArguments(
        output_dir="/tmp/editlens_text_inference",
        per_device_eval_batch_size=args.batch_size,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    def score_texts(texts: list[str]) -> np.ndarray:
        encoded = [
            tokenizer(clean_text(text), truncation=True, max_length=args.max_length)
            for text in texts
        ]
        return softmax(trainer.predict(encoded).predictions, axis=1)

    probabilities = None if args.leave_one_sentence_out else score_texts(documents)

    for index, document in enumerate(documents):
        loo = None
        if args.leave_one_sentence_out:
            loo = leave_one_sentence_out(
                document, tokenizer, score_texts, args.max_length
            )
            probs = np.asarray(loo["probabilities"])
        else:
            probs = probabilities[index]
        ranking = sorted(
            ({"bucket": int(bucket), "probability": float(prob)}
             for bucket, prob in enumerate(probs)),
            key=lambda item: item["probability"], reverse=True,
        )
        result = {
            "text_preview": document[:80],
            "predicted_bucket": int(np.argmax(probs)),
            "score": score_from_probabilities(probs),
            "ranking": ranking,
        }
        if loo is not None:
            result["aggregation"] = {
                "method": "mean_probability_across_sentence_windows",
                "n_windows": loo["n_windows"],
            }
            result["sentence_attributions"] = loo["sentence_attributions"]
        print(json.dumps(result))


if __name__ == "__main__":
    main()
