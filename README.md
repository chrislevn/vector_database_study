# Vector Database from scratch study
This project studies different index methods in vector databases and how one can create a simple vector database from scratch.
Current supported index methods: FLAT, IVF, HNSW, PQ, SQ. 

Currently data is loaded from SICK2014 dataset. You can change the data source in helper.py.

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

2. Run the UI 
```bash
make run
```

3. Optional: Run the tests
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

## Future improvements
1. Loading time for data preprocessing. I'm looking to understand how to improve the loading time for the data preprocessing. I'm considering using a database to store the data and load it faster.