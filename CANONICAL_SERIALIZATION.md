# Canonical aggregate serialization

EHRDyn uses one writer for generated JSON and JSONL hash surfaces:
`kdd2027_benchmark.canonical`. Static source files retain ordinary byte hashes
and are not rewritten to mimic computed-result portability.

The computed contract is UTF-8, lexicographically sorted object keys, compact
`,` and `:` separators, JSON string escaping with `ensure_ascii=false`, and one
LF after each document. Floating-point values must be finite and are rounded
half-even to 12 decimal places before numeric serialization. Negative zero is
written as `0`. Integers, booleans, strings, arrays, and null retain their JSON
types. Object keys must be strings.

The maximum quantization error is `5e-13`, tighter than the public reporting
precision of `1e-6`. Nonfinite values fail before bytes are written. The same
writer is used by CLI-generated aggregate JSON, credentialed aggregate receipts
and preprocessing contracts, public POMDP mechanism hashes, and the portability
probe. JSONL writes one independently canonicalized object plus LF per row.

Python 3.11, 3.12, and 3.13 must produce identical canonical bytes for the
frozen aggregate probe. Raw pre-quantization floating values are compared
separately and must remain within the frozen absolute tolerance `5e-12`.
