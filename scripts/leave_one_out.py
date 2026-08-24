"""Sentence-level leave-one-out attribution for document classifiers.

The public function is model-agnostic: callers supply a tokenizer and a
function that returns one probability vector per input text. Long documents
are evaluated as sentence-preserving windows and their window probabilities
are averaged.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np


ProbabilityScorer = Callable[[Sequence[str]], np.ndarray]


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences without an extra NLP-model dependency.

    This deliberately lightweight splitter handles normal punctuation. It is
    not intended for specialised tokenisation such as legal citations.
    """
    return [
        match.group().strip()
        for match in re.finditer(r"\S[\s\S]*?(?:[.!?]+(?=\s|$)|$)", text)
        if match.group().strip()
    ]


def sentence_windows(
    sentences: Sequence[str], tokenizer: object, max_length: int
) -> list[str]:
    """Group whole sentences into windows that fit the model context."""
    windows: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        candidate = " ".join([*current, sentence])
        token_count = len(tokenizer(candidate, add_special_tokens=True)["input_ids"])
        if current and token_count > max_length:
            windows.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)

    if current:
        windows.append(" ".join(current))
    return windows


def score_from_probabilities(probabilities: np.ndarray) -> float:
    """Convert a bucket-probability vector to the EditLens 0--1 score."""
    buckets = np.arange(len(probabilities))
    return float(probabilities @ buckets / (len(probabilities) - 1))


def leave_one_sentence_out(
    text: str,
    tokenizer: object,
    score_texts: ProbabilityScorer,
    max_length: int,
) -> dict:
    """Measure the effect of omitting every sentence from *text*.

    ``score_texts`` receives a sequence of windows and must return an array of
    shape ``(len(texts), n_buckets)``. A positive ``score_delta`` means that
    removing the sentence lowered the document score, so that sentence pushes
    the score in the AI direction under this approximation.
    """
    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("Text contains no sentences")

    variants: list[tuple[int | None, list[str]]] = [
        (None, sentence_windows(sentences, tokenizer, max_length))
    ]
    for index in range(len(sentences)):
        remaining = [sentence for i, sentence in enumerate(sentences) if i != index]
        if remaining:
            variants.append((index, sentence_windows(remaining, tokenizer, max_length)))

    # Send all windows through the model in one logical call so the caller can
    # batch them efficiently. ``boundaries`` maps them back to each variant.
    all_windows: list[str] = []
    boundaries: list[tuple[int | None, int, int]] = []
    for index, windows in variants:
        start = len(all_windows)
        all_windows.extend(windows)
        boundaries.append((index, start, len(all_windows)))
    window_probabilities = score_texts(all_windows)
    # Also score each sentence independently. This is intentionally reported
    # alongside (rather than replacing) leave-one-out: standalone sentence
    # scores are useful for highlighting, but omit document context.
    sentence_probabilities = score_texts(sentences)

    aggregate: dict[int | None, np.ndarray] = {
        index: window_probabilities[start:end].mean(axis=0)
        for index, start, end in boundaries
    }
    baseline = aggregate[None]
    baseline_score = score_from_probabilities(baseline)

    attributions = []
    for index, sentence in enumerate(sentences):
        without = aggregate.get(index)
        if without is None:  # A one-sentence document has no non-empty variant.
            attributions.append(
                {
                    "sentence_index": index,
                    "sentence": sentence,
                    "sentence_score": score_from_probabilities(sentence_probabilities[index]),
                    "sentence_bucket": int(np.argmax(sentence_probabilities[index])),
                    "sentence_bucket_probabilities": sentence_probabilities[index].tolist(),
                    "score_without": None,
                    "score_delta": None,
                }
            )
            continue
        score_without = score_from_probabilities(without)
        attributions.append(
            {
                "sentence_index": index,
                "sentence": sentence,
                "sentence_score": score_from_probabilities(sentence_probabilities[index]),
                "sentence_bucket": int(np.argmax(sentence_probabilities[index])),
                "sentence_bucket_probabilities": sentence_probabilities[index].tolist(),
                "score_without": score_without,
                "score_delta": baseline_score - score_without,
            }
        )

    return {
        "probabilities": baseline.tolist(),
        "score": baseline_score,
        "n_windows": len(variants[0][1]),
        "sentence_attributions": attributions,
    }
