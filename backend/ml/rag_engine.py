import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import json
import shutil
import glob
def ingest_syllabus(file_path, student_id):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    persist_directory = os.path.join(base_dir, 'chroma_db', student_id)
    os.makedirs(persist_directory, exist_ok=True)
    
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    # Chroma persistence is automatic in newer versions
    # if hasattr(vectorstore, 'persist'):
    #     vectorstore.persist()
        
    return True

def delete_syllabus(student_id):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    # 1. Delete Chroma DB folder
    persist_directory = os.path.join(base_dir, 'chroma_db', student_id)
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory, ignore_errors=True)
        
    # 2. Delete associated PDFs in uploads directory
    upload_dir = os.path.join(base_dir, 'uploads')
    pattern = os.path.join(upload_dir, f"{student_id}_*.pdf")
    for file_path in glob.glob(pattern):
        try:
            os.remove(file_path)
        except OSError:
            pass
            
    return True

def delete_single_syllabus(student_id, filename):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    upload_dir = os.path.join(base_dir, 'uploads')
    file_path = os.path.join(upload_dir, filename)
    
    # 1. Delete from ChromaDB
    persist_directory = os.path.join(base_dir, 'chroma_db', student_id)
    if os.path.exists(persist_directory):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            if vectorstore._collection:
                vectorstore._collection.delete(where={"source": file_path})
        except Exception as e:
            print(f"Error removing from vectorstore: {e}")
            
    # 2. Delete the physical file
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass
            
    return True

def generate_quiz_from_rag(topic, student_id, selected_files=None, quantity=5):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    persist_directory = os.path.join(base_dir, 'chroma_db', student_id)
    
    if not os.path.exists(persist_directory):
        raise ValueError("No uploaded syllabus found. Please upload a PDF first.")
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    # Setup filter if selected_files are provided
    search_kwargs = {"k": 3}
    if selected_files and len(selected_files) > 0:
        # Reconstruct absolute paths
        upload_dir = os.path.join(base_dir, 'uploads')
        full_paths = [os.path.join(upload_dir, f) for f in selected_files]
        if len(full_paths) == 1:
            search_kwargs["filter"] = {"source": full_paths[0]}
        else:
            search_kwargs["filter"] = {"source": {"$in": full_paths}}
            
    import random
    if topic.lower() == "all topics":
        # Pull documents filtered by source if provided
        f = search_kwargs.get("filter")
        try:
            if not f:
                res = vectorstore.get(include=['documents'])
            else:
                res = vectorstore.get(where=f, include=['documents'])
                
            all_contents = res.get('documents', [])
            if not all_contents:
                raise ValueError("No documents found in the selected collection.")
            # Sample 5 random chunks for a broader spread of "all topics"
            selected_contents = random.sample(all_contents, min(5, len(all_contents)))
            context = "\n\n".join(selected_contents)
        except Exception as e:
            print(f"Error fetching all topics: {e}")
            raise ValueError(f"Could not retrieve all topics: {str(e)}")
    else:
        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
        docs = retriever.invoke(topic)
        
        if not docs:
            raise ValueError("No relevant context found in your documents for this topic.")
            
        context = "\n\n".join([doc.page_content for doc in docs])
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    
    prompt = PromptTemplate.from_template(
        "You are an academic examiner. Based ONLY on the following context, generate {quantity} Multiple Choice Questions on the topic: {topic}.\n\n"
        "Context:\n{context}\n\n"
        "Format the output STRICTLY as a JSON list of objects. Each object must have:\n"
        '- "question": The question text\n'
        '- "options": A list of 4 possible answers\n'
        '- "answer": The exact text of the correct option\n'
        '- "explanation": A brief "Neural Explanation" of why the answer is correct based on the context\n\n'
        "Output ONLY valid JSON. Do not include markdown code blocks like ```json."
    )
    
    chain = prompt | llm
    
    response = chain.invoke({"topic": topic, "context": context, "quantity": quantity})
    
    import re
    raw_text = response.content.strip()
    # Find the JSON array pattern in the response
    match = re.search(r'\[\s*\{.*\}\s*\]', raw_text, re.DOTALL)
    if match:
        raw_text = match.group(0)
    else:
        # Fallback to original cleaning if regex fails
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        
    return json.loads(raw_text.strip())
