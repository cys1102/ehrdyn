# KDD257 command receipt

Credentialed:

```bash
export AUTHORIZED_MIMICIV_3_1_ROOT='<authorized MIMIC-IV v3.1 directory>'
ehrdyn-icu construct-ehr --output '<new private output directory>'
```

Anonymous:

```bash
python -m pip install 'ehrdyn-icu[credentialed]'
ehrdyn-icu construct-ehr --help
python -m unittest discover -s tests
ehrdyn-icu scan-release --root .
ehrdyn-icu verify-checksums --root .
```
