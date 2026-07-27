# Released schema validation

All released schemas use JSON Schema Draft 2020-12 and are validated by
`jsonschema==4.26.0` before use.

| Released schema | Accepting CLI | Accepting API |
| --- | --- | --- |
| `schemas/leaderboard_submission.schema.json` | `validate-submission` | `validate_submission` |
| `schemas/transition_submission.schema.json` | `validate-transition-submission` | `validate_transition_submission` |
| `schemas/aggregate_metrics.schema.json` | `aggregate-report` | `write_aggregate_report` |
| `schemas/ehr_component_submission.schema.json` | `score-ehr-components` | `score_submission` |
| `schemas/ehr_component_result.schema.json` | scorer output validation | `validate_instance` |

The entire document is schema-validated before semantic evaluation. Schema
rules are authoritative for required properties, types, enums, constants,
patterns, bounds, and additional properties. Semantic checks then bind task
identifiers, task-config hashes, ordered-feature hashes, action/timing views,
allowed tracks, metric identity uniqueness, and observed-count consistency.

Failures use `JSON pointer [failed keyword]: concise message`. Run:

```bash
uv sync --frozen
uv run ehrdyn-icu validate-schemas --schema-dir schemas
uv run ehrdyn-icu validate-submission \
  --submission submission/leaderboard_submission_template.json \
  --config-dir configs/tasks
uv run ehrdyn-icu validate-transition-submission \
  --submission fixtures/transition_submission_small.json \
  --config-dir configs/tasks
uv run ehrdyn-icu score-ehr-components \
  --output /tmp/ehrdyn_ehr_component_score.json
```
