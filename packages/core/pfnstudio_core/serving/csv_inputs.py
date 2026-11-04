"""`data_csv` → `{rows, columns}` for table-shaped predict inputs.

Callers have a spreadsheet, not a JSON matrix. Asking them to hand-build
`rows` in the checkpoint's exact column order makes the single most damaging
mistake — values landing one column left of where they belong — both easy to
make and impossible to see: the model scores the shifted row confidently and
returns a number that looks fine.

A CSV carries its header WITH the data, so alignment happens by NAME. The
packers already accept `columns` + `rows` and reorder to the declared
contract (refusing anything missing), so expanding a CSV into that pair
removes the positional hazard without any model-side change.

Scope: this is for table-shaped tasks (`input_schema.kind` of `table` /
`table_with_target`). It is deliberately NOT universal — `context_query`
needs two matrices and `rdb` needs several tables plus a join graph, and
flattening either into one CSV would lose the thing that makes it that task.
Those envelopes are passed through untouched.

Every failure names its location. A blank cell is an error, never a zero: in
this domain zero is a real setting (a wave cap, a QC level), so silently
substituting it would turn "I forgot a value" into a plausible wrong answer.
"""

from __future__ import annotations

from typing import Any


class CsvInputError(ValueError):
    """A `data_csv` that cannot be turned into rows without guessing."""


def _coerce(cell: str, *, row_no: int, column: str) -> Any:
    """One cell → int / float / str. Blank raises.

    Numeric-looking values become numbers so a table task gets the tensor it
    expects; anything else stays a string for the packer to interpret (a
    categorical like ``yes`` or a zone name is legitimate in adapter tables).
    """
    text = cell.strip()
    if not text:
        raise CsvInputError(
            f"row {row_no}, column '{column}' is empty. Every cell needs a value — "
            "a blank is not read as 0, because 0 is a real setting here."
        )
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def expand_data_csv(
    inputs: dict[str, Any],
    declared: list[str] | None = None,
) -> dict[str, Any]:
    """Return `inputs` with any `data_csv` expanded into `rows` + `columns`.

    Untouched when there is no `data_csv`, so every existing envelope keeps
    working and this can sit on the shared path.

    `declared` (the task's `input_schema.columns`, when the caller knows them)
    is used only to produce the better error: a CSV missing a column the
    checkpoint requires is named here, at the boundary, instead of surfacing
    later as a width mismatch.
    """
    if not isinstance(inputs, dict) or "data_csv" not in inputs:
        return inputs

    raw = inputs.get("data_csv")
    if not isinstance(raw, str) or not raw.strip():
        raise CsvInputError("'data_csv' is empty — paste a CSV with a header row.")
    if inputs.get("rows"):
        raise CsvInputError(
            "send either 'data_csv' or 'rows', not both — with both present there "
            "is no way to tell which one you meant to score."
        )

    import csv as _csv
    import io

    reader = _csv.reader(io.StringIO(raw.strip()))
    records = [r for r in reader if any(c.strip() for c in r)]
    if len(records) < 2:
        raise CsvInputError(
            "'data_csv' needs a header row naming the columns, then at least one data row."
        )

    header = [c.strip() for c in records[0]]
    if any(not h for h in header):
        raise CsvInputError("every column in the CSV header needs a name.")
    dupes = sorted({h for h in header if header.count(h) > 1})
    if dupes:
        raise CsvInputError(f"CSV header repeats column(s): {', '.join(dupes)}.")

    if declared:
        missing = [c for c in declared if c not in header]
        if missing:
            raise CsvInputError(
                f"CSV is missing column(s) this model scores: {', '.join(missing)}. "
                f"It needs a header naming: {', '.join(declared)}."
            )

    rows: list[list[Any]] = []
    for i, rec in enumerate(records[1:], start=1):
        if len(rec) != len(header):
            raise CsvInputError(
                f"row {i} has {len(rec)} value(s) but the header names {len(header)} column(s)."
            )
        rows.append([_coerce(cell, row_no=i, column=header[j]) for j, cell in enumerate(rec)])

    out = {k: v for k, v in inputs.items() if k != "data_csv"}
    out["rows"] = rows
    out["columns"] = header
    return out
