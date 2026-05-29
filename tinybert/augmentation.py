"""
Data augmentation from TinyBERT Algorithm 1 (arXiv:1909.10351, Section 3.2).

For each WORD in a sentence:
  - If BERT's tokenizer keeps the word as a single piece: mask the whole word,
    take the top-K MLM predictions, sample one with probability p_t.
  - If BERT's tokenizer splits the word into multiple pieces: look up the K
    nearest GloVe neighbours of the whole word, sample one with probability p_t.

Default hyper-parameters from the paper: pt=0.4, Na=20, K=15.

GloVe is optional; without it, multi-piece words are left unchanged.
"""

import os
import random
import re
import string
from typing import Optional

import numpy as np
import torch
from transformers import BertTokenizer, BertForMaskedLM


_WORD_RE = re.compile(r"\S+")


def _load_glove(glove_path: str, vocab: set[str]) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    if not os.path.isfile(glove_path):
        return vectors
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab:
                vectors[word] = np.array(parts[1:], dtype=np.float32)
    return vectors


def _cosine_neighbours(word: str, glove: dict[str, np.ndarray], k: int) -> list[str]:
    if word not in glove or not glove:
        return []
    vec = glove[word]
    norm = np.linalg.norm(vec)
    if norm == 0:
        return []
    scores = {
        w: float(np.dot(vec, v) / (norm * np.linalg.norm(v) + 1e-9))
        for w, v in glove.items() if w != word
    }
    return sorted(scores, key=scores.get, reverse=True)[:k]


def _split_words(sentence: str) -> list[str]:
    """Whitespace tokenisation. Punctuation stays attached to the word."""
    return _WORD_RE.findall(sentence)


def _is_punct(word: str) -> bool:
    return all(c in string.punctuation for c in word)


class TinyBERTAugmenter:
    """
    Replicates Algorithm 1 from TinyBERT.

    Args:
        bert_model_name: HuggingFace model ID for BERT MLM.
        glove_path: Path to GloVe .txt file (optional).
        pt: Replacement probability threshold.
        na: Number of augmented samples per example.
        k: Candidate set size.
        device: torch device.
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        glove_path: Optional[str] = None,
        pt: float = 0.4,
        na: int = 20,
        k: int = 15,
        device: str = "cpu",
    ):
        self.pt = pt
        self.na = na
        self.k = k
        self.device = device

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.mlm = BertForMaskedLM.from_pretrained(bert_model_name)
        self.mlm.eval()
        self.mlm.to(device)

        self.glove: dict[str, np.ndarray] = {}
        if glove_path:
            print(f"Loading GloVe from {glove_path} ...")
            all_words = set(self.tokenizer.vocab.keys())
            self.glove = _load_glove(glove_path, all_words)
            print(f"  Loaded {len(self.glove)} GloVe vectors.")

    @torch.no_grad()
    def _bert_candidates(self, words: list[str], idx: int) -> list[str]:
        """Mask the whole word at position `idx`, return top-K MLM predictions."""
        masked = words.copy()
        masked[idx] = self.tokenizer.mask_token
        text = " ".join(masked)
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        mask_positions = (enc["input_ids"][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_positions) == 0:
            return []
        logits = self.mlm(**enc).logits[0]
        top_ids = logits[mask_positions[0]].topk(self.k).indices.tolist()
        candidates = self.tokenizer.convert_ids_to_tokens(top_ids)
        # drop BERT special tokens and any wordpiece continuations
        candidates = [c for c in candidates if not c.startswith("[") and not c.startswith("##")]
        return candidates[: self.k]

    def _word_candidates(self, words: list[str], idx: int) -> list[str]:
        """Dispatch to BERT-MLM or GloVe based on the word's subword count."""
        word = words[idx]
        if _is_punct(word):
            return []
        pieces = self.tokenizer.tokenize(word)
        if len(pieces) <= 1:
            return self._bert_candidates(words, idx)
        return _cosine_neighbours(word.lower(), self.glove, self.k)

    def augment(self, sentence: str) -> list[str]:
        """Return up to Na augmented versions of `sentence`."""
        words = _split_words(sentence)
        if not words:
            return [sentence] * self.na

        # Candidate sets are deterministic given (sentence, position) — compute
        # once per position, sample Na times. This is ~Na× faster than the
        # naïve inner-loop call.
        candidates_per_pos = [self._word_candidates(words, i) for i in range(len(words))]

        augmented: list[str] = []
        for _ in range(self.na):
            new_words = words.copy()
            for i, cands in enumerate(candidates_per_pos):
                if cands and random.random() <= self.pt:
                    new_words[i] = random.choice(cands)
            augmented.append(" ".join(new_words))
        return augmented

    def augment_dataset(self, sentences: list[str]) -> list[str]:
        """Augment a list of sentences, returning original + all augmentations."""
        result = list(sentences)
        for sent in sentences:
            result.extend(self.augment(sent))
        return result
