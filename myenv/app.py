import ollama  # Enables interaction with local large language models (LLMs)
import gradio as gr  # Provides an easy-to-use web interface for the chatbot
import os  # For file management
import re  # For working with regular expressions

from langchain_community.document_loaders import PyMuPDFLoader  # Extracts text from PDF files for processing
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits text into smaller chunks
from langchain.vectorstores import Chroma  # Handles storage and retrieval of vector embeddings
from langchain_ollama import OllamaEmbeddings  # Use the updated OllamaEmbeddings class from langchain-ollama

# Ensure directory for vectorstore exists
if not os.path.exists('./chroma_db'):
    os.makedirs('./chroma_db')

def process_pdf(pdf_files):
    if not pdf_files:
        return None, None, None  # Return None if no PDFs are uploaded
    
    all_chunks = []
    for pdf_bytes in pdf_files:
        loader = PyMuPDFLoader(pdf_bytes) 
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        all_chunks.extend(chunks)
    
    embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
    vectorstore = Chroma.from_documents(documents=all_chunks, embedding=embeddings, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever()

    return text_splitter, vectorstore, retriever

def combine_docs(retrieved_docs):
    return "\n".join([doc.page_content for doc in retrieved_docs])

def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(
        model="deepseek-r1:1.5b",  
        messages=[{'role': 'user', 'content': formatted_prompt}]
    )
    response_content = response['message']['content']
    final_answer = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()
    return final_answer

def rag_chain(question, text_splitter, vectorstore, retriever):
    retrieved_docs = retriever.invoke(question)
    formatted_content = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_content)

def ask_question(pdf_files, question): 
    text_splitter, vectorstore, retriever = process_pdf(pdf_files)
    if text_splitter is None:
        return "No PDF uploaded or PDFs are empty."
    result = rag_chain(question, text_splitter, vectorstore, retriever)
    return result

# Set up Gradio interface
interface = gr.Interface(
    fn=ask_question,  # Function to process user input and generate response
    inputs=[
        gr.File(label="Upload PDF(s) (multiple files allowed)", file_count="multiple"),  # Accept multiple files
        gr.Textbox(label="Ask a question")  # Text input where the user types their question
    ],
    outputs="text",  # The function returns a text response
    title="Ask questions about your PDFs",  # The title displayed on the interface
    description="Use DeepSeek-R1 1.5B to answer your questions about the uploaded PDF documents.",  # Description of the app
)

# Launch the Gradio interface
interface.launch(share=True)