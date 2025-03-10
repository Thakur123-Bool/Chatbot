import ollama
import os
import re
from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ensure directory for vectorstore exists
VECTOR_DB_PATH = "./chroma_db"
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

def process_pdf(pdf_files):
    if not pdf_files:
        return None, None, None
    
    all_chunks = []
    for pdf_file in pdf_files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(pdf_file.filename))
        pdf_file.save(file_path)
        
        loader = PyMuPDFLoader(file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        all_chunks.extend(chunks)
    
    embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")  # Ensure Ollama model is available
    vectorstore = Chroma.from_documents(documents=all_chunks, embedding=embeddings, persist_directory=VECTOR_DB_PATH)
    retriever = vectorstore.as_retriever()
    
    return text_splitter, vectorstore, retriever

def combine_docs(retrieved_docs):
    return "\n".join([doc.page_content for doc in retrieved_docs])

def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(
        model="deepseek-r1:1.5b",  # Use the model you specified
        messages=[{'role': 'user', 'content': formatted_prompt}]
    )
    response_content = response['message']['content']
    final_answer = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()
    return final_answer

def rag_chain(question, text_splitter, vectorstore, retriever):
    retrieved_docs = retriever.invoke(question)
    formatted_content = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_content)

@app.route('/ask', methods=['POST'])
def ask_question():
    if 'pdf_files' not in request.files:
        return jsonify({"error": "No PDF files uploaded"}), 400
    
    pdf_files = request.files.getlist('pdf_files')
    question = request.form.get('question')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    text_splitter, vectorstore, retriever = process_pdf(pdf_files)
    if text_splitter is None:
        return jsonify({"error": "No valid PDFs uploaded"}), 400
    
    result = rag_chain(question, text_splitter, vectorstore, retriever)
    return jsonify({"answer": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
