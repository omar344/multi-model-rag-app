# multi-model-rag

This is a implementation of the RAG multi model for question answering.

## Requirements

- Python 3.8 or later

#### Install Python using MiniConda

1) Download and install Miniconda from [here](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)
2) create a new environment using the following command:
```bash
conda create -n multi-model-rag-app python=3.12.9
```
3) Activate the environment:
```bash
conda activate multi-model-rag-app
```

### (Optional) Setup your command line interface for better readability
```bash
export PS1="\[\033[01;32m\]\u@\h:\w\n\[\033[00m\]\$ "
```

## Installation 

### Install the required pakages

```bash
pip install -r requirements.txt
```

### Setup the environment variables

```bash
cp .env.example .env
```

set your environment variables in the `.env` file. like `OPEN_API_KEY` value.

## Run the FastAPI server
```bash
 uvicorn main:app --reload --host 0.0.0.0 --port 5000
```
## Postman Collection

Download the POSTMAN collection from [/assets/multi-model-rag-app.postman_collection.json](/assets\multi-model-rag-app.postman_collection.json)
