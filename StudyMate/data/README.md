# Dataset

The full MS MARCO v2.1 dataset is intentionally not stored here because it is about 2 GB.

The notebook downloads a reproducible sample using Hugging Face **streaming mode**:
`microsoft/ms_marco`, configuration `v2.1`, split `train`, streamed with `itertools.islice`.

Full dataset statistics:
- Train: 808,731
- Validation: 101,093
- Test: 101,092
- Total: 1,010,916 queries

For the demo, `StudyMate.ipynb` streams the first 20,000 training records (a small
portion of the first shard), so only a lightweight download is needed.
