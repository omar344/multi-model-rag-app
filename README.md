# Multi-Model RAG Application

A FastAPI-based application implementing Retrieval-Augmented Generation (RAG) with multiple model support. This application provides a flexible architecture for document processing, embedding generation, and question answering using various LLM providers and vector databases.

## Features

- Multiple LLM provider support (OpenAI, Groq, Voyage AI)
- Multiple vector database support
- Document processing and embedding generation
- Authentication system
- RESTful API endpoints for RAG operations
- MongoDB integration for data persistence

## Requirements

- Python 3.10 or later
- MongoDB
- System dependencies for document processing

### System Dependencies

For Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr libmagic-dev
```

For Windows:
1. Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/)
3. Add both to your system PATH

## Installation

1. **Set up Python Environment**

   Using Miniconda (recommended):
   ```bash
   # Install Miniconda from https://docs.conda.io/en/latest/miniconda.html
   
   # Create and activate environment
   conda create -n multi-model-rag-app python=3.10
   conda activate multi-model-rag-app
   ```

2. **Install PyTorch (CPU version)**
   ```bash
   pip install torch==2.2.0+cpu --index-url https://download.pytorch.org/whl/cpu
   pip install torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Install Project Dependencies**
   ```bash
   cd src
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   ```bash
   # Copy example environment file
   cp .env.example .env
   ```
   
   Edit `.env` file and set the following required variables:
   - `MONGODB_URL`: Your MongoDB connection string
   - `GENERATION_BACKEND`: Your chosen LLM provider (e.g., "openai", "groq", "voyage")
   - `GENERATION_MODEL_ID`: Model ID for text generation
   - `EMBEDDING_BACKEND`: Provider for embeddings
   - `EMBEDDING_MODEL_ID`: Model ID for embeddings
   - Required API keys based on your chosen providers

## Running the Application

1. **Start MongoDB**
   Ensure your MongoDB instance is running and accessible

2. **Start the FastAPI Server**
   ```bash
   cd src
   uvicorn main:app --reload --host 0.0.0.0 --port 5000
   ```
   The API will be available at `http://localhost:5000`

## API Documentation

- Swagger UI: `http://localhost:5000/docs`
- ReDoc: `http://localhost:5000/redoc`


## License

This project is licensed under the terms included in the LICENSE file.
