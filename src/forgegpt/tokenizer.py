"""Byte-level BPE tokenizer (the scheme GPT-2 uses), trained on our own corpus.

Byte-level: the starting alphabet is the 256 possible byte values, so any string can
be encoded and there is no "unknown token". BPE (byte-pair encoding) then repeatedly
merges the most frequent adjacent pair of tokens into a new token, until the
vocabulary reaches the target size. Frequent words ("transmission") end up as one
token; rare words are built from a few pieces.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"  # end-of-text: separates narratives when they are packed together
# Placeholders written by the cleaning step. The byte-level pre-tokenizer splits
# punctuation from letters, so BPE could never learn "[phone]" as one token on its
# own; we add them explicitly so each is always exactly one token.
PLACEHOLDERS = ["[phone]", "[email]", "[url]", "[vin]", "[xxx]"]


def train_tokenizer(texts: Iterable[str], vocab_size: int, length: int | None = None) -> Tokenizer:
    """Train a byte-level BPE tokenizer whose final vocabulary is `vocab_size`.

    `length` is only for the progress bar (number of texts, if known).
    """
    if vocab_size % 64:
        # The embedding matrix is (vocab_size x d_model); GPUs are faster when its
        # dimensions are multiples of 64.
        raise ValueError(f"vocab_size should be a multiple of 64, got {vocab_size}")

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        # Leave room for the placeholders added after training.
        vocab_size=vocab_size - len(PLACEHOLDERS),
        min_frequency=2,
        special_tokens=[EOT],  # gets id 0
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer, length=length)
    tok.add_tokens(PLACEHOLDERS)
    return tok


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))
