# pdf-query-rag
Read and Query PDF using RAG

Since this is using local ollama embeddings, you need to have ollama installed and running.

Also pull phi4 model.

1. place you pdf file in the data directory
2. run the following command to populate the chroma db
```bash
python rag.py
```
3. run the following command to query the pdf
```bash
python query.py "<query>"
```