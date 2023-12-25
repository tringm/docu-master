# DocuMaster

"DocuMaster" is an expert at answering questions based on PDF documents.

## Getting started

Main components:
- [ChromaDB](https://www.trychroma.com/) to index and retrieve documents.
- [Microsoft Phi 2 LLM](https://huggingface.co/TheBloke/phi-2-GGUF) loaded with [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) to generate answer
- [FastAPI](https://fastapi.tiangolo.com/) for the REST API.


### Run the app

First, download the Phi-2 model:

```shell
make download-model
```

Start the stack (ChromaDB + API) with [docker compose](https://docs.docker.com/compose/:

```shell
docker-compose build
docker-compose up docu-master
```

The API is available at `http://localhost:8080`:
- Check out the [API docs](http://localhost:8080/docs)

## Limitations and basis for future improvements
### Upload Endpoint:
- The application can handle only PDF files and plain text
- The application uses naive PDF parsing (e.g.: only parse text and not images and table) and chunking strategy (recursive splitting into a chunk of specified size).

  While automatic parsing and chunk can be improved, especially if there's a defined set of documents, it might be best the user is guided to an annotation app so that they can verify the parsed document and define semantic correct chunks.
- No document was saved and indexing chunks to the vector store is done in the upload endpoint. This is for demo purpose only.

  For production, this flow can be refactored as :
  - Upload Endpoint returns a signed URL for file upload
  - The user uploads a file to a storage bucket which triggers and event to parse and index the file's content
  - The user can query the upload job status through another API
- The Vector DB uses the default [Sentence Transformers all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) which does not yield the best result. This can be improved with, for example, using LLM for embeddings.


### QA Endpoint:
- For demo purpose, the QA endpoint was not implemented with user identification. This means that there's no separation of documents between users, and an user can access all documents.
  - For future improvements, this issue can be addressed with having a separate [ChromaDB Collection](https://docs.trychroma.com/reference/Collection) for each user, or even a separate ChromaDB instance for each user.


## Development
### Pre-commit

The project uses pre-commit hooks run all the auto-formatters (e.g. `black`, `isort`), linters (e.g. `mypy`, `ruff`), and other quality
 checks to make sure the changeset is in good shape before a commit/push happens.

### Testing

To run unittests:

```shell
pytest
```

To run API integration tests:

```shell
docker compose up integration-tests
```

To run long-running evaluation tests (e.g.: LLM and VectorDB):

```shell
pytest --run-evaluation -k evaluation
```

The evaluation tests use LLM to evaluate the returned answer which is not always reliable.
To address this, evaluation tests use a combination approach of metric evaluation and output human evaluation.

For example, QA tests would write the test case, and the returned answer to the output file.

This enables two additions:
 - The human makes the evaluation trade off between evaluation metrics and eye-tests evaluation (e.g.: comprehensibility of the answer)
 - The changes to the generated outputs is communicated transparently.

To enable comparison mode that compares the generated output after test finishes:

```shell
pytest --output-diff --run-evaluation -k evaluation
```
