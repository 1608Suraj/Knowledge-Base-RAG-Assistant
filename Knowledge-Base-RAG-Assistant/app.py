"""
Simple RAG Document Assistant
A beginner-friendly document chat system using AI
"""

import os
import json
import faiss
import PyPDF2
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from groq import Groq

# =============================================================================
# CONFIGURATION - Easy to understand settings
# =============================================================================

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# File paths
UPLOAD_FOLDER = "uploads"
INDEX_FILE = "search_index.bin"
METADATA_FILE = "documents.json"

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =============================================================================
# INITIALIZATION - Load AI models and data
# =============================================================================

app = Flask(__name__)

# Load AI model for understanding documents
print("Loading AI model...")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load Groq AI for answering questions
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("✅ AI system ready!")
except:
    groq_client = None
    print("⚠️ AI system not available - using simple responses")

# Load existing documents
documents = []
search_index = None

def load_documents():
    """Load saved documents from file"""
    global documents, search_index
    
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            documents = json.load(f)
        print(f"Loaded {len(documents)} documents")
    
    if os.path.exists(INDEX_FILE):
        search_index = faiss.read_index(INDEX_FILE)
        print("Search index loaded")

def save_documents():
    """Save documents to file"""
    with open(METADATA_FILE, 'w') as f:
        json.dump(documents, f, indent=2)

# Load on startup
load_documents()

# =============================================================================
# DOCUMENT PROCESSING - Handle PDF uploads
# =============================================================================

def extract_pdf_text(file_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text.strip()

def extract_txt_text(file_path):
    """Extract text from TXT file"""
    text = ""
    try:
        # Try different encodings to handle various text files
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    text = file.read()
                break
            except UnicodeDecodeError:
                continue
        if not text:
            # Fallback to binary read and decode
            with open(file_path, 'rb') as file:
                content = file.read()
                text = content.decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"Error reading TXT file: {e}")
    return text.strip()

def add_document(filename, text):
    """Add a new document to the system"""
    global documents, search_index
    
    # Create document entry
    doc_id = len(documents)
    document = {
        "id": doc_id,
        "filename": filename,
        "text": text,
        "title": filename
    }
    
    # Add to documents list
    documents.append(document)
    
    # Create search index if this is the first document
    if search_index is None:
        # Create new index
        embeddings = embedding_model.encode([text])
        search_index = faiss.IndexFlatL2(embeddings.shape[1])
        search_index.add(embeddings)
    else:
        # Add to existing index
        embeddings = embedding_model.encode([text])
        search_index.add(embeddings)
    
    # Save everything
    save_documents()
    faiss.write_index(search_index, INDEX_FILE)
    
    print(f"✅ Added document: {filename}")

def delete_document(doc_id):
    """Delete a document from the system"""
    global documents, search_index
    
    try:
        doc_id = int(doc_id)
        
        # Find document
        doc_to_delete = None
        for doc in documents:
            if doc['id'] == doc_id:
                doc_to_delete = doc
                break
        
        if not doc_to_delete:
            return False, "Document not found"
        
        # Remove from documents list
        documents.remove(doc_to_delete)
        
        # Rebuild search index (FAISS doesn't support deletion)
        if len(documents) == 0:
            search_index = None
            # Remove index file
            if os.path.exists(INDEX_FILE):
                os.remove(INDEX_FILE)
        else:
            # Rebuild index with remaining documents
            texts = [doc['text'] for doc in documents]
            embeddings = embedding_model.encode(texts)
            search_index = faiss.IndexFlatL2(embeddings.shape[1])
            search_index.add(embeddings)
            faiss.write_index(search_index, INDEX_FILE)
        
        # Update document IDs to be sequential
        for i, doc in enumerate(documents):
            doc['id'] = i
        
        # Save documents
        save_documents()
        
        # Remove physical file
        file_path = os.path.join(UPLOAD_FOLDER, doc_to_delete['filename'])
        if os.path.exists(file_path):
            os.remove(file_path)
        
        print(f"✅ Deleted document: {doc_to_delete['filename']}")
        return True, f"Successfully deleted {doc_to_delete['filename']}"
        
    except Exception as e:
        return False, f"Error deleting document: {str(e)}"

# =============================================================================
# SEARCH FUNCTION - Find relevant documents
# =============================================================================

def search_documents(query, max_results=3):
    """Search for relevant documents"""
    if search_index is None or len(documents) == 0:
        return []
    
    # Get query embedding
    query_embedding = embedding_model.encode([query])
    
    # Search for similar documents
    scores, indices = search_index.search(query_embedding, max_results)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(documents):
            results.append({
                "document": documents[idx],
                "score": float(score)
            })
    
    return results

# =============================================================================
# AI RESPONSE - Generate answers using Groq
# =============================================================================

def get_ai_response(query, relevant_docs):
    """Get AI response using Groq"""
    
    # If no Groq available, use simple response
    if groq_client is None:
        return create_simple_response(query, relevant_docs)
    
    # Build context from relevant documents
    context = ""
    for result in relevant_docs:
        doc = result["document"]
        context += f"Document: {doc['title']}\nContent: {doc['text'][:500]}...\n\n"
    
    # Create prompt for AI with expertise assignment
    prompt = f"""You are an expert knowledge assistant designed for employees to quickly retrieve accurate information from internal documentation, manuals, and FAQs. Your role is to:

Provide precise, context-aware answers by leveraging semantic search over company knowledge bases.

Help employees with onboarding by explaining processes, policies, and technical details in simple terms.

Retrieve and summarize relevant documents or sections, highlighting only what's most important.

Maintain a professional, concise, and supportive tone at all times.

If the requested information is not found, guide the employee on where or how they can access it.

Your primary goal is to reduce time spent searching documents and ensure reliable, real-time assistance for employees.

FORMATTING REQUIREMENTS:
- Use clear headings and bullet points when appropriate
- Keep responses concise but comprehensive
- Use professional language
- If referencing specific documents, mention the document name
- End with "Is there anything else I can help you with?" if the answer is complete

Based on these documents, answer the question: {query}

Documents:
{context}

Please provide a well-structured, professional response:"""
    
    try:
        # Get response from Groq
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"AI error: {e}")
        return create_simple_response(query, relevant_docs)

def create_simple_response(query, relevant_docs):
    """Create simple response without AI"""
    if not relevant_docs:
        return "I don't have any relevant documents to answer your question. Please upload some PDF or TXT files first, then ask your question again."
    
    response = f"## Answer to: {query}\n\n"
    response += "Based on your uploaded documents, here's what I found:\n\n"
    
    for i, result in enumerate(relevant_docs[:2], 1):
        doc = result["document"]
        response += f"### {i}. From {doc['title']}\n"
        response += f"{doc['text'][:300]}...\n\n"
    
    response += "**Note:** This is a basic response. For more detailed answers, please ensure the AI system is properly configured.\n\n"
    response += "Is there anything else I can help you with?"
    
    return response

# =============================================================================
# WEB INTERFACE - Flask routes
# =============================================================================

@app.route('/')
def home():
    """Main page"""
    return render_template('simple_index.html', 
                         document_count=len(documents),
                         has_ai=groq_client is not None)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload - supports multiple files"""
    files = request.files.getlist('file')
    
    if not files or all(file.filename == '' for file in files):
        return jsonify({"error": "No files selected"}), 400
    
    uploaded_files = []
    errors = []
    
    for file in files:
        if file.filename == '':
            continue
            
        # Check file extension
        filename_lower = file.filename.lower()
        if not (filename_lower.endswith('.pdf') or filename_lower.endswith('.txt')):
            errors.append(f"{file.filename}: Only PDF and TXT files are supported")
            continue
        
        try:
            # Save file
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            
            # Extract text based on file type
            if filename_lower.endswith('.pdf'):
                text = extract_pdf_text(file_path)
                if not text:
                    errors.append(f"{file.filename}: Could not extract text from PDF. The file might be corrupted or password-protected.")
                    # Clean up the file
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    continue
            else:  # TXT file
                text = extract_txt_text(file_path)
                if not text:
                    errors.append(f"{file.filename}: Could not read TXT file. The file might be empty or corrupted.")
                    # Clean up the file
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    continue
            
            # Add to system
            add_document(file.filename, text)
            uploaded_files.append(file.filename)
            
        except Exception as e:
            errors.append(f"{file.filename}: Upload failed - {str(e)}")
            # Clean up the file if it was saved but processing failed
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass
    
    # Prepare response
    if uploaded_files and not errors:
        return jsonify({"message": f"Successfully uploaded {len(uploaded_files)} file(s): {', '.join(uploaded_files)}"})
    elif uploaded_files and errors:
        return jsonify({
            "message": f"Partially successful: uploaded {len(uploaded_files)} file(s), {len(errors)} failed",
            "uploaded": uploaded_files,
            "errors": errors
        })
    else:
        return jsonify({"error": "Upload failed", "details": errors}), 400

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "No question provided"}), 400
    
    # Search for relevant documents
    relevant_docs = search_documents(query)
    
    # Get AI response
    answer = get_ai_response(query, relevant_docs)
    
    return jsonify({
        "answer": answer,
        "documents_found": len(relevant_docs)
    })

@app.route('/documents')
def list_documents():
    """List all documents"""
    return jsonify(documents)

@app.route('/delete/<int:doc_id>', methods=['DELETE'])
def delete_document_route(doc_id):
    """Delete a document"""
    success, message = delete_document(doc_id)
    
    if success:
        return jsonify({"message": message})
    else:
        return jsonify({"error": message}), 400

# =============================================================================
# START APPLICATION
# =============================================================================

if __name__ == '__main__':
    print("🚀 Starting Simple RAG Assistant...")
    print("📁 Upload PDF files to add them to your knowledge base")
    print("💬 Ask questions about your documents")
    print("🌐 Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
