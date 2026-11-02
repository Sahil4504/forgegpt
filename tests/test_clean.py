"""Cleaning rules, tested on (shortened) examples taken from the real data."""

from collections import Counter

from forgegpt.clean import (
    clean_narrative,
    mask_phones,
    mask_urls,
    mask_vins,
    normalize_characters,
    remove_foia_footer,
    remove_staff_tags,
)


def test_staff_tags_removed_at_end_and_middle():
    assert remove_staff_tags("ENGINE STALLED.  *AK *TS").strip() == "ENGINE STALLED."
    assert "*" not in remove_staff_tags("LOSS OF CONTROL. *AW  ALSO SEE ODINO 480080.")
    assert remove_staff_tags("LATCH FAILURE.     *AK  TT").strip() == "LATCH FAILURE."


def test_chained_and_wrapped_staff_tags_removed():
    assert remove_staff_tags("NOT AVAILABLE.*DT*JB").strip() == "NOT AVAILABLE."
    assert remove_staff_tags("UNKNOWN. *DT* THE END").split() == ["UNKNOWN.", "THE", "END"]
    assert remove_staff_tags("*TS ***NAR***  THE CAR").split() == ["THE", "CAR"]


def test_staff_tag_rule_keeps_emphasis_and_lowercase():
    assert remove_staff_tags("it did *NOT* stop") == "it did *NOT* stop"
    assert remove_staff_tags("cost *ab") == "cost *ab"


def test_foia_footer_removed():
    text = (
        "PEDAL WENT TO THE FLOOR. INFORMATION REDACTED PURSUANT TO THE FREEDOM OF "
        "INFORMATION ACT (FOIA), 5 U.S.C. 552(B)(6).  *TR"
    )
    assert clean_narrative(text) == "pedal went to the floor."


def test_foia_footer_long_variant():
    text = (
        "SERVICE STABILITRAK  *TR    PARTS OF THIS DOCUMENT HAVE BEEN REDACTED TO PROTECT "
        "PERSONALLY IDENTIFIABLE INFORMATION PURSUANT TO THE FREEDOM OF INFORMATION ACT "
        "(FOIA), 5 U.S.C. 552(B)(6)."
    )
    assert clean_narrative(text) == "service stabilitrak"


def test_foia_footer_without_citation():
    out = remove_foia_footer("x. Information redacted pursuant to the Freedom of Information Act")
    assert out.strip() == "x."


def test_smart_and_mangled_punctuation():
    assert (
        normalize_characters("I\u2019m \u201cfine\u201d \u2014 ok\u2026") == 'I\'m "fine" - ok...'
    )
    assert normalize_characters("WE WON\x12T FIX IT") == "WE WON'T FIX IT"
    assert normalize_characters("DID NOT \x13CATCH\x14") == 'DID NOT "CATCH"'
    assert normalize_characters("x" + chr(0xF0B7) + "y") == "x y"  # private-use bullet
    assert normalize_characters("a\u200bb\ufeff") == "ab"
    assert normalize_characters("caf\u00e9 90\u00b0") == "caf\u00e9 90\u00b0"


def test_phone_numbers_masked():
    assert mask_phones("CALL 404-261-7770,WK NO.") == "CALL [phone],WK NO."
    assert mask_phones("(619) 686-1775 VOICE PAGER") == "[phone] VOICE PAGER"
    assert mask_phones("1-800-555-1234") == "[phone]"
    assert mask_phones("ILL 60090    847   541-2898") == "ILL 60090    [phone]"
    assert mask_phones("RECALL 96V073000 AT 45,000 MILES") == "RECALL 96V073000 AT 45,000 MILES"


def test_vins_masked_but_not_long_words():
    assert mask_vins("VIN #1B3HD46T5SF683587.") == "VIN #[vin]."
    assert mask_vins("ELECTROMECHANICALS") == "ELECTROMECHANICALS"


def test_urls_masked_keeping_trailing_punctuation():
    assert mask_urls("see https://www.example.com/a/b). next") == "see [url]). next"
    assert mask_urls("go to www.nhtsa.gov.") == "go to [url]."


def test_full_pipeline_and_counts():
    counts = Counter()
    raw = "&#65279;SECOND VEHICLE VIN:[XXX].  EMAIL ME AT PGT1@NETCARRIER.COM  *AK"
    assert clean_narrative(raw, counts) == "second vehicle vin:[xxx]. email me at [email]"
    assert counts["staff_tags"] == 1 and counts["email"] == 1 and counts["lowercase"] == 1
    assert counts["phone"] == 0
