"""Cleaning rules for NHTSA complaint narratives.

Each rule is a small function so it can be tested on its own. `clean_narrative`
applies them in order; the order matters (e.g. personal info is masked before
lowercasing, and whitespace is collapsed last). Every rule here came from profiling
the real data; see docs/DATA.md for the numbers behind each one.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from collections.abc import Callable

MIN_CHARS = 20  # narratives shorter than this after cleaning are dropped

# --- characters -----------------------------------------------------------------------

_CHAR_MAP = {
    # curly quotes, dashes, bullets, odd spaces -> plain ASCII
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2033": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
    "\u2015": "-", "\u2212": "-", "\u2e3a": "-", "\u2e3b": "-",
    "\u2026": "...", "\u2022": "-", "\u25cf": "-", "\u00b7": "-",
    "\u00a0": " ", "\u2007": " ", "\u2009": " ", "\u202f": " ",
    "\u2028": " ", "\u2029": " ",
    "\u200b": "", "\u200c": "", "\u200d": "", "\u2060": "", "\ufeff": "", "\ufe0f": "",
    # Older records store smart punctuation with its high byte stripped
    # (e.g. U+2019 became 0x19). Found by looking at these bytes in context.
    "\x11": "'", "\x12": "'", "\x18": "'", "\x19": "'",
    "\x13": '"', "\x14": '"', "\x1c": '"', "\x1d": '"',
    "\x16": " - ",
}  # fmt: skip
_CHAR_TABLE = str.maketrans(_CHAR_MAP)


def unescape_html(text: str) -> str:
    """'&amp;' -> '&', '&#65279;' -> the character it encodes."""
    return html.unescape(text) if "&" in text else text


def normalize_characters(text: str) -> str:
    """Map smart punctuation to ASCII; drop control, format and private-use characters.

    Accented letters and symbols like the degree sign are kept: the byte-level
    tokenizer handles them fine.
    """
    text = text.translate(_CHAR_TABLE)
    if text.isprintable():  # fast path (C speed): nothing invisible left to remove
        return text
    return "".join(" " if unicodedata.category(ch) in ("Cc", "Cf", "Co") else ch for ch in text)


# --- NHTSA boilerplate ----------------------------------------------------------------

_FOIA_RE = re.compile(
    r"(?:parts of this document have been redacted to protect personally identifiable"
    r" information|information redacted) pursuant to the freedom of information act"
    r"(?:\s*\(foia\))?(?:,?\s*5\s*u\.?\s*s\.?\s*c\.?\s*(?:\u00a7\s*)?552\s*\(b\)\s*\(6\))?\.?",
    re.IGNORECASE,
)
# Analyst initials and markers: "*AK", "*SLC", chained "*DT*JB", wrapped "*DT*",
# "***NAR***". Uppercase only. "*NOT*" is real emphasis (the only common one), so keep it.
_STAFF_TAG_RE = re.compile(r"(?:\*+(?!NOT\*)[A-Z]{2,3}\b)+\**")
# Older records sometimes end with bare analyst initials "TT" after the starred tag.
_TRAILING_TT_RE = re.compile(r"(?:\s+TT)+\s*$")


def remove_foia_footer(text: str) -> str:
    return _FOIA_RE.sub(" ", text)


def remove_staff_tags(text: str) -> str:
    if "*" not in text and not text.endswith("TT"):
        return text
    return _TRAILING_TT_RE.sub("", _STAFF_TAG_RE.sub(" ", text))


# --- personal information ---------------------------------------------------------------

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+?(?=[.,;:!?)\]]*(?:\s|$))", re.IGNORECASE)
# 17-char VIN: no I, O or Q, and at least one digit (so long words never match).
_VIN_RE = re.compile(r"\b(?=[A-HJ-NPR-Z0-9]*\d)[A-HJ-NPR-Z0-9]{17}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?:\b1[-. ])?(?:\(\d{3}\)\s*|\b\d{3}[-. ]{1,3})\d{3}[-. ]{1,3}\d{4}\b")


def mask_emails(text: str) -> str:
    return _EMAIL_RE.sub("[email]", text) if "@" in text else text


def mask_urls(text: str) -> str:
    return _URL_RE.sub("[url]", text)


def mask_vins(text: str) -> str:
    return _VIN_RE.sub("[vin]", text)


def mask_phones(text: str) -> str:
    return _PHONE_RE.sub("[phone]", text)


# --- final normalization ------------------------------------------------------------------

_REDACTION_RE = re.compile(r"\[?\b[Xx]{3,}\b\]?")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_redactions(text: str) -> str:
    """NHTSA's own redaction marker appears as [XXX], XXX, XXXX...; unify to [xxx]."""
    return _REDACTION_RE.sub("[xxx]", text)


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


RULES: list[tuple[str, Callable[[str], str]]] = [
    ("html_entities", unescape_html),
    ("characters", normalize_characters),
    ("foia_footer", remove_foia_footer),
    ("staff_tags", remove_staff_tags),
    ("email", mask_emails),
    ("url", mask_urls),
    ("vin", mask_vins),
    ("phone", mask_phones),
    ("redaction_marker", normalize_redactions),
    ("lowercase", str.lower),
    ("whitespace", collapse_whitespace),
]


def clean_narrative(text: str, counts: Counter[str] | None = None) -> str:
    """Apply every rule in order. If `counts` is given, tally which rules changed text."""
    for name, rule in RULES:
        new = rule(text)
        if counts is not None and new != text:
            counts[name] += 1
        text = new
    return text
