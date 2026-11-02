import zipfile

import pyarrow.parquet as pq

from forgegpt.data import COLUMNS, MIN_FIELDS, iter_lines, parse_line, write_parquet

VIN = "1FTFW1EF5FK"


def make_line(**overrides: str) -> str:
    """Build a realistic 51-field flat-file line."""
    fields = [""] * 51
    fields[0] = "1"
    fields[1] = "10000001"
    fields[3] = "FORD"
    fields[4] = "F-150"
    fields[5] = "2015"
    fields[11] = "ENGINE"
    fields[14] = VIN
    fields[16] = "20240105"
    fields[19] = "THE ENGINE STALLED ON THE HIGHWAY."
    for name, value in overrides.items():
        fields[COLUMNS[name]] = value
    return "\t".join(fields) + "\r\n"


def write_zip(path, lines):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("FLAT_CMPL.txt", "".join(lines).encode("utf-8"))


def test_parse_line_extracts_selected_fields():
    rec = parse_line(make_line())
    assert set(rec) == set(COLUMNS)
    assert rec["make"] == "FORD"
    assert rec["date_received"] == "20240105"
    assert rec["narrative"] == "THE ENGINE STALLED ON THE HIGHWAY."


def test_parse_line_never_keeps_personal_fields():
    assert VIN not in parse_line(make_line()).values()


def test_parse_line_rejects_short_lines():
    assert parse_line("a\tb\tc\n") is None


def test_parse_line_accepts_older_shorter_records():
    line = "\t".join(make_line().split("\t")[:MIN_FIELDS])
    assert parse_line(line)["narrative"] == "THE ENGINE STALLED ON THE HIGHWAY."


def test_stray_carriage_return_does_not_split_a_record(tmp_path):
    zip_path = tmp_path / "FLAT_CMPL.zip"
    write_zip(zip_path, [make_line(narrative="LINE ONE\rLINE TWO"), make_line()])
    assert len(list(iter_lines(zip_path))) == 2


def test_zip_to_parquet_roundtrip(tmp_path):
    zip_path = tmp_path / "FLAT_CMPL.zip"
    write_zip(zip_path, [make_line(odino=str(i), narrative=f"case {i} café") for i in range(5)])

    records = [parse_line(line) for line in iter_lines(zip_path)]
    assert records[2]["narrative"] == "case 2 café"

    out = tmp_path / "out.parquet"
    n = write_parquet(records, out, batch_size=2)  # forces several batches
    table = pq.read_table(out)
    assert n == table.num_rows == 5
    assert table.column_names == list(COLUMNS)
    assert not (tmp_path / "out.parquet.tmp").exists()
