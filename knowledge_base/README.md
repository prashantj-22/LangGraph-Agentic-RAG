# Knowledge base — local documents

Drop files here and they get ingested into the FAISS index alongside
`SOURCE_URLS`. The directory is scanned **recursively**.

Supported file types:

| Extension | Loader |
| --- | --- |
| `.pdf` | `PyPDFLoader` (pypdf) |
| `.docx` | `Docx2txtLoader` (docx2txt) |
| `.txt`, `.md` | `TextLoader` |

Legacy `.doc` is **not** supported — convert to `.docx` first.

After adding or changing files:

```bash
python -m src.build_db --rebuild
```

Override this directory with `KB_DIR` in `.env`. Add extra one-off paths with
`python -m src.build_db --source /path/to/file_or_dir`.
