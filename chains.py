from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS  # ✅ Correct import
import os
from dotenv import load_dotenv

load_dotenv()


class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
        )

    def pdf_parser(self, file_path):
        """
        Parses a PDF file and splits it into chunks.

        Args:
            file_path (str): The path to the PDF file.

        Returns:
            list: A list of text chunks extracted from the PDF.
        """
        pdf_reader = PyPDFLoader(file_path)
        documents = pdf_reader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        return chunks

    def create_embeddings(self, chunks):
        """
        Creates embeddings from text chunks and initializes a FAISS vector store.

        Args:
            chunks (list): A list of text chunks to be embedded.

        Returns:
            FAISS: The FAISS vector store instance containing the embeddings.
        """
        model_name = "sentence-transformers/all-mpnet-base-v2"
        hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)

        # Create FAISS Vector Store
        db = FAISS.from_documents(documents=chunks, embedding=hf_embeddings)
        return db

    def create_condense_question_prompt(self):
        """
        Creates and returns a PromptTemplate for condensing the follow-up question.

        Returns:
            PromptTemplate: The PromptTemplate instance.
        """
        return PromptTemplate.from_template(
            """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
            Chat History:
            {chat_history}
            Follow Up Input: {question}
            Standalone question:"""
        )

    def create_conversational_retrieval_chain(self, db):
        """
        Creates and returns a ConversationalRetrievalChain with the specified database.

        Args:
            db (FAISS): The FAISS vector store to be used for retrieval.

        Returns:
            ConversationalRetrievalChain: The ConversationalRetrievalChain instance.
        """
        condense_question_prompt = self.create_condense_question_prompt()

        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=db.as_retriever(),
            condense_question_prompt=condense_question_prompt,
            return_source_documents=True,
            verbose=False,
        )

    def query_chain(self, qa, query, chat_history):
        """
        Queries the ConversationalRetrievalChain and returns the answer.

        Args:
            qa (ConversationalRetrievalChain): The ConversationalRetrievalChain instance.
            query (str): The question to ask.
            chat_history (list): The history of the conversation.

        Returns:
            str: The answer from the chain.
        """
        # Ensure chat_history is a list
        if not isinstance(chat_history, list):
            chat_history = []

        result = qa({"question": query, "chat_history": chat_history})
        return result["answer"]


# # Example usage
# # Create an instance of the Chain class
# chain_instance = Chain()

# # Parse the PDF and create embeddings
# chunks = chain_instance.pdf_parser(r"D:\path\to\your\pdf.pdf")
# db = chain_instance.create_embeddings(chunks)

# # Create the ConversationalRetrievalChain
# qa_chain = chain_instance.create_conversational_retrieval_chain(db)

# # Query the chain
# chat_history = []
# query = "What is this paper all about?"
# answer = chain_instance.query_chain(qa_chain, query, chat_history)

# print(answer)
