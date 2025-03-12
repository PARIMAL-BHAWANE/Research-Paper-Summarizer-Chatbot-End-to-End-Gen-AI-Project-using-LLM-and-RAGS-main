# Research-Paper-Summarizer-Chatbot-End-to-End-Gen-AI-Project-using-LLM-and-RAGS


A project leveraging large language models (LLMs) to simplify research papers. Users upload papers and ask questions in plain English, receiving clear, layman-friendly explanations of key concepts and equations. Powered by LLMs and Retrieval Augmented Generation (RAG), this chatbot aids researchers and students in understanding complex content.

<p align="center">
  <img width="800" alt="image" src="https://github.com/user-attachments/assets/af2b836d-aea2-49da-955e-e7f777fdae61">
</p>



## Overview

This project provides a Streamlit web application for interacting with PDF documents using a conversational AI model. Users can upload a PDF, ask questions based on the content, and receive contextually relevant answers. The system uses a retrieval-based approach combined with a generative model to handle complex queries.

## Project Structure

- **`chains.py`**: Contains the `Chain` class which handles PDF processing, document retrieval, and generating responses using a conversational model.
- **`main.py`**: Implements a Streamlit app for uploading PDFs, asking questions, and displaying answers.

## Code Explanation

### `chains.py`

- **Chain Class**: Manages the entire workflow including PDF parsing, document embedding, and querying.
  - **PDF Parsing**: Uses `PyPDFLoader` to extract text from PDF files and `RecursiveCharacterTextSplitter` to split the text into chunks.
  - **Embedding Creation**: Utilizes `HuggingFaceEmbeddings` to create embeddings and stores them in a `FAISS` vector store.
  - **Conversational Chain**: Constructs a `ConversationalRetrievalChain` that integrates the language model (`ChatGroq`) with the FAISS-based retriever.
  - **Query Handling**: The `query_chain` method processes user queries and returns answers based on the chat history.

### `main.py`

- **Streamlit Application**: Provides a web interface for users to interact with the chatbot.
  - **File Upload**: Allows users to upload PDFs.
  - **Question Input**: Enables users to ask questions related to the PDF content.
  - **Chat History**: Displays previous questions and answers.

## RAG (Retrieval-Augmented Generation) Usage

In this project, principles of RAG are utilized but not explicitly through an RAG model:

- **Retrieval**:
  - **Component**: FAISS vector store.
  - **Function**: Retrieves relevant text chunks from the PDF based on user queries.

- **Generation**:
  - **Component**: ChatGroq (or any generative model you use).
  - **Function**: Generates responses based on the retrieved text chunks and the query.

While the setup does not use the RAG model architecture directly, it effectively combines retrieval and generation in a manner similar to RAG principles, where retrieval is followed by response generation.

## Set up environment variables

Create a `.env` file in the root directory with the following content:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

## Usage
#### Run the Streamlit app:
```
streamlit run main.py
 ```

## Interact with the app

1. Upload a PDF file.
2. Ask questions related to the content of the uploaded PDF.

<img width="463" alt="image" src="https://github.com/user-attachments/assets/ecefe50c-aee2-4371-92ee-c31a3ac76056">
