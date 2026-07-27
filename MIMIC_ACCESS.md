# MIMIC-IV access boundary

MIMIC-IV v3.1 is credentialed data and is not redistributed by EHRDyn.
The public package contains the code needed to reconstruct the five EHR task
interfaces locally. It contains no MIMIC rows, derived result tables,
checkpoints, or split membership.

## Local construction

Install the credentialed dependency group, set the authorized source directory
without printing it, and select a new private output directory.

```bash
python -m pip install '.[credentialed]'
export AUTHORIZED_MIMICIV_3_1_ROOT='<authorized MIMIC-IV v3.1 directory>'
ehrdyn-icu construct-ehr --output '<new private output directory>'
```

The command requires the official MIMIC-IV v3.1 flat-file layout. It validates
the required tables and columns before constructing the prespecified sepsis,
respiratory support, shock, AKI, and heart-failure cohorts. Subject assignment
uses each task's frozen deterministic hash rule and is patient-disjoint: 70%
train, 15% validation, and 15% historical evaluation. Preprocessing
statistics are fit on training subjects only.

If both `.csv` and `.csv.gz` encodings of an official table are present, the
wrapper creates a temporary view that selects `.csv.gz` without modifying the
source directory. The output records only aggregate encoding counts. The
scientific constructor still sees exactly one encoding per required table.

The private output contains one restricted modeling archive for each task and
role, plus aggregate construction and resource receipts. Files inside
`private_arrays/` include normalized histories, observation masks, recency,
actions, rewards, termination, valid-step masks, and observed targets. They
must remain within the authorized environment. `--aggregate-only` omits those
archives while retaining the aggregate construction receipt.

Construction reproducibility does not establish clinical validity, causal
action effects, or the value of an unexecuted policy.

Official access information:

- MIMIC-IV v3.1: https://physionet.org/content/mimiciv/3.1/
- Credentialed DUA: https://physionet.org/content/mimiciv/view-dua/3.1/
- Training instructions: https://physionet.org/about/citi-course/
- Credentialing FAQ: https://physionet.org/about/faqs/

Do not upload MIMIC-IV rows or restricted derivatives to public repositories,
third-party APIs, shared prompts, or unapproved online services.
