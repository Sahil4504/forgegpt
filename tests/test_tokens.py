import numpy as np

from forgegpt.data import encode_to_bin, load_tokens
from forgegpt.tokenizer import EOT, train_tokenizer

TEXTS = [
    "the brakes failed on the highway.",
    "the transmission slipped at [phone] mph.",
    "airbag light on.",
] * 40


def test_bin_roundtrip(tmp_path):
    tok = train_tokenizer(TEXTS, vocab_size=320)
    eot = tok.token_to_id(EOT)
    path = tmp_path / "train.bin"

    n_docs, n_tokens = encode_to_bin(TEXTS, tok, eot, path, batch_size=7)  # several batches
    ids = load_tokens(path)

    assert ids.dtype == np.uint16
    assert n_docs == len(TEXTS) and n_tokens == len(ids)
    assert path.stat().st_size == 2 * n_tokens
    assert ids[-1] == eot and int((ids == eot).sum()) == len(TEXTS)

    # Splitting the stream on EOT and decoding gives back exactly the original texts.
    docs, current = [], []
    for i in ids.tolist():
        if i == eot:
            docs.append(tok.decode(current))
            current = []
        else:
            current.append(i)
    assert docs == TEXTS
