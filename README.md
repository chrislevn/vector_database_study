# Vector Database from scratch study
[![Unit Test](https://github.com/chrislevn/vector_database_study/actions/workflows/unit-test.yml/badge.svg)](https://github.com/chrislevn/vector_database_study/actions/workflows/unit-test.yml)
[![HitCount](https://hits.dwyl.com/chrislevn/vector_database_study.svg?style=flat-square)](http://hits.dwyl.com/chrislevn/vector_database_study)

This project studies different index methods in vector databases and how one can create a simple vector database from scratch.
Current supported index methods: FLAT, IVF, HNSW, PQ, SQ. 

Currently data is loaded from SICK2014 dataset. You can change the data source in helper.py.

[Blog post](https://christopherle.com/blog/2024/research-on-vector-db/)

## Takeways: 
- FLAT is good but is slow. 
- HNSW is the fastest but the accuracy is not as good as IVF.
- I don't use wrappers like Langchain. I want to study the concepts not build the tool at production level. The pro of this is full control of the code.

## Current issues: 
- The loading time is long. I handled this with cache in the helper.py file. In returns, the accuracy might be affected. Need to figure out why. 

## Components
- `index.py`: contains the implementation of the vector database indexes.
- `main.py` :contains the implementation of the vector database.
- `app.py` :contains the implementation of the UI.
- `helper.py` :contains the helper functions like preparing data, check the time esplaped, etc.

## How to use 

1. Install the dependencies
```bash
pip install -r requirements.txt
```

2. Run the UI. It will run the `streamlit` UI
```bash
make run
```

3. Optional: Run the tests. The tests also include some time and memory testing. 
```bash
make test
```

<!-- ## DevOps
### Building and running your application

When you're ready, start your application by running:
`docker compose up --build`.

Your application will be available at http://localhost:5002.

### Deploying your application to the cloud

First, build your image, e.g.: `docker build -t myapp .`.
If your cloud uses a different CPU architecture than your development
machine (e.g., you are on a Mac M1 and your cloud provider is amd64),
you'll want to build the image for that platform, e.g.:
`docker build --platform=linux/amd64 -t myapp .`.

Then, push it to your registry, e.g. `docker push myregistry.com/myapp`.

Consult Docker's [getting started](https://docs.docker.com/go/get-started-sharing/)
docs for more detail on building and pushing.

### References
* [Docker's Python guide](https://docs.docker.com/language/python/) -->

## Future development
- Figure out ways to make VectorDB faster and less memory consuming. (although there are tradeoffs but it is work in progress depending on the use case). 
- Implement this in low-level languages like C++ or Rust!.