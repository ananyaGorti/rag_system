import os
import sys
import time
import re
import numpy as np
from typing import List, Dict, Any, Tuple
import argparse
import gradio as gr
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
    
    def process_documents(self, directory_path: str) -> List:
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory {directory_path} not found")
        
        documents = []
        
        pdf_loader = DirectoryLoader(
            directory_path, 
            glob="**/*.pdf", 
            loader_cls=PyPDFLoader
        )
        documents.extend(pdf_loader.load())
        
        text_loader = DirectoryLoader(
            directory_path, 
            glob="**/*.txt", 
            loader_cls=TextLoader
        )
        documents.extend(text_loader.load())
        
        if not documents:
            print(f"No documents found in {directory_path}")
            return []
        
        print(f"Found {len(documents)} documents")
        return self.text_splitter.split_documents(documents)

class VectorStore:
    def __init__(self, embedding_model=None):
        if embedding_model is None:
            self.embedding_model = OpenAIEmbeddings()
        else:
            self.embedding_model = embedding_model
        self.vector_store = None
    
    def create_vector_store(self, documents, persist_directory=None):
        start_time = time.time()
        print("Creating vector store...")
        self.vector_store = FAISS.from_documents(documents, self.embedding_model)
        
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self.vector_store.save_local(persist_directory)
            print(f"Vector store saved to {persist_directory}")
        
        print(f"Vector store created in {time.time() - start_time:.2f} seconds")
        return self.vector_store
    
    def load_vector_store(self, persist_directory):
        if not os.path.exists(persist_directory):
            raise FileNotFoundError(f"Directory {persist_directory} not found")
        
        start_time = time.time()
        print(f"Loading vector store from {persist_directory}...")
        self.vector_store = FAISS.load_local(
            persist_directory, 
            self.embedding_model,
            allow_dangerous_deserialization=True
        )
        print(f"Vector store loaded in {time.time() - start_time:.2f} seconds")
        return self.vector_store
    
    def get_retriever(self, k=4):
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized")
        
        return self.vector_store.as_retriever(search_kwargs={"k": k})

class RAGSystem:
    def __init__(self, llm_model="gpt-4.1", temperature=0.0, 
                 embedding_model=None, vector_store=None):
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        
        if vector_store is None:
            self.vector_store = VectorStore(embedding_model)
        else:
            self.vector_store = vector_store
            
        self.qa_chain = None
    
    def setup_from_documents(self, documents, persist_directory=None):
        vector_store = self.vector_store.create_vector_store(documents, persist_directory)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
        self.qa_chain = self._create_qa_chain(retriever)
        return self
    
    def setup_from_vector_store(self, persist_directory):
        vector_store = self.vector_store.load_vector_store(persist_directory)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
        self.qa_chain = self._create_qa_chain(retriever)
        return self
    
    def _create_qa_chain(self, retriever):
        from langchain.prompts import PromptTemplate
        
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            you are an assistant tasked with answering questions based ONLY on the provided context.
            if the answer cannot be found in the context, say "I don't have enough information to answer this question." 
            do not use your general knowledge to answer questions. answer hello or greetings with "How can I help you?". try your best to provide an answer. 
            
            Context:
            {context}
            
            Question: {question}
            Answer:
            """
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": prompt_template
            },
            return_source_documents=True
        )
    
    def answer_question(self, question):
        if self.qa_chain is None:
            return "The system is not initialized. Please set up the system first."
        
        result = self.qa_chain({"query": question})
        answer = result["result"]
        source_docs = result.get("source_documents", [])
        
        # Format source metadata for display
        sources = []
        for i, doc in enumerate(source_docs):
            source = doc.metadata.get("source", f"Document {i+1}")
            page = doc.metadata.get("page", "")
            if page:
                source = f"{source} (Page {page})"
            sources.append(source)
        
        sources_text = "\n".join([f"- {source}" for source in set(sources)])
        
        return f"{answer}\n\nSources:\n{sources_text}" if sources else answer

def create_ui(rag_system=None):
    with gr.Blocks(title="Document RAG System") as app:
        gr.Markdown("# Document-based RAG System")
        
        with gr.Tab("Setup"):
            with gr.Row():
                docs_dir = gr.Textbox(label="Documents Directory", placeholder="Path to documents directory")
                vector_store_dir = gr.Textbox(label="Vector Store Directory", placeholder="Path to save/load vector store")
            
            with gr.Row():
                llm_model = gr.Dropdown(
                    ["gpt-3.5-turbo", "gpt-4.1", "gpt-4o"],
                    label="LLM Model",
                    value="gpt-4.1"
                )
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="Temperature")
            
            with gr.Row():
                process_btn = gr.Button("Process Documents & Create Vector Store")
                load_btn = gr.Button("Load Existing Vector Store")
            
            status_msg = gr.Textbox(label="Status", interactive=False)
        
        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column():
                    chatbot = gr.Chatbot(height=500)
                    msg = gr.Textbox(placeholder="Ask a question about your documents...")
                    submit_btn = gr.Button("Submit")
        
        # Setup event handlers
        def process_documents(docs_dir, vector_store_dir, llm_model, temperature):
            try:
                nonlocal rag_system
                processor = DocumentProcessor()
                documents = processor.process_documents(docs_dir)
                
                if not documents:
                    return "No documents found or error processing documents."
                
                rag_system = RAGSystem(llm_model=llm_model, temperature=float(temperature))
                rag_system.setup_from_documents(documents, vector_store_dir)
                
                return f"Successfully processed {len(documents)} document chunks and created vector store."
            except Exception as e:
                return f"Error: {str(e)}"
        
        def load_vector_store(vector_store_dir, llm_model, temperature):
            try:
                nonlocal rag_system
                rag_system = RAGSystem(llm_model=llm_model, temperature=float(temperature))
                rag_system.setup_from_vector_store(vector_store_dir)
                
                return f"Successfully loaded vector store from {vector_store_dir}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        def user_query(message, history):
            if rag_system is None:
                history.append((message, "Please set up the system in the Setup tab first."))
                return history, ""
            
            answer = rag_system.answer_question(message)
            history.append((message, answer))
            return history, ""
        
        process_btn.click(process_documents, inputs=[docs_dir, vector_store_dir, llm_model, temperature], outputs=status_msg)
        load_btn.click(load_vector_store, inputs=[vector_store_dir, llm_model, temperature], outputs=status_msg)
        submit_btn.click(user_query, inputs=[msg, chatbot], outputs=[chatbot, msg])
        msg.submit(user_query, inputs=[msg, chatbot], outputs=[chatbot, msg])
        
        return app

def main():
    parser = argparse.ArgumentParser(description="Document-based RAG System")
    parser.add_argument("--docs_dir", type=str, help="Directory containing documents")
    parser.add_argument("--vector_store_dir", type=str, help="Directory to save/load vector store")
    parser.add_argument("--llm_model", type=str, default="gpt-3.5-turbo", help="LLM model to use")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for LLM")
    parser.add_argument("--ui", action="store_true", help="Launch Gradio UI")
    
    args = parser.parse_args()
    
    if args.ui:
        app = create_ui()
        app.launch(share=True)
        return
    
    if not args.docs_dir and not args.vector_store_dir:
        print("Please provide either a documents directory or a vector store directory.")
        parser.print_help()
        return
    
    rag_system = RAGSystem(llm_model=args.llm_model, temperature=args.temperature)
    
    if args.docs_dir:
        processor = DocumentProcessor()
        documents = processor.process_documents(args.docs_dir)
        rag_system.setup_from_documents(documents, args.vector_store_dir)
    elif args.vector_store_dir:
        rag_system.setup_from_vector_store(args.vector_store_dir)
    
    # Interactive console
    print("\nRAG System is ready. Type your questions (or 'exit' to quit):")
    while True:
        question = input("\nQuestion: ")
        if question.lower() in ["exit", "quit", "q"]:
            break
        
        answer = rag_system.answer_question(question)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()