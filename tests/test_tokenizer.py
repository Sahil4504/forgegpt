import pytest

from forgegpt.tokenizer import EOT, PLACEHOLDERS, load_tokenizer, train_tokenizer

E_ACUTE, DEGREE = chr(0xE9), chr(0xB0)

CORPUS = [
    "the brakes failed while driving on the highway at 60 mph.",
    "the transmission slipped and the check engine light came on.",
    "dealer replaced the airbag sensor but the warning light stayed on.",
    "call me at [phone] or email [email], vin [vin], see [url]. name [xxx].",
    f"the engine overheated at 105{DEGREE} near the caf{E_ACUTE}.",
] * 50


@pytest.fixture(scope="module")
def tok():
    return train_tokenizer(CORPUS, vocab_size=384)


def test_vocab_size_and_eot_id(tok):
    assert tok.get_vocab_size() == 384
    assert tok.token_to_id(EOT) == 0


def test_roundtrip_is_lossless(tok):
    for text in [
        *CORPUS[:5],
        "never seen before: zxqv!! 12345 \t tabs",
        "emoji " + chr(0x26A0) + " and unicode " + chr(0x2192),
        "call [phone] now, then [xxx] wrote [url]",
    ]:
        assert tok.decode(tok.encode(text).ids) == text


def test_placeholders_are_single_tokens(tok):
    for p in PLACEHOLDERS:
        ids = tok.encode(f"see {p} here").ids
        assert tok.token_to_id(p) in ids


def test_frequent_words_get_merged(tok):
    assert len(tok.encode("the transmission").ids) < len("the transmission")


def test_rejects_unaligned_vocab_size():
    with pytest.raises(ValueError):
        train_tokenizer(CORPUS, vocab_size=1000)


def test_save_and_load(tok, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(str(path))
    assert load_tokenizer(path).encode("the brakes").ids == tok.encode("the brakes").ids
