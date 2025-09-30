# Simple RAG Document Assistant 📚🤖

A beginner-friendly AI-powered document chat system that allows you to upload PDF and TXT files, then ask questions about them using artificial intelligence.

## 🌟 Features

- 📄 **Multi-format Support**: Upload both PDF and TXT files
- 🔍 **AI-Powered Search**: Find relevant information using semantic search
- 💬 **Intelligent Chat**: Ask questions and get AI-generated answers
- 🗑️ **Document Management**: View and delete uploaded documents
- 🎨 **Simple Interface**: Clean, drag-and-drop web interface
- 🚀 **Easy Setup**: One-command installation and setup

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Libraries Used](#libraries-used)
3. [Code Explanation](#code-explanation)
4. [How It Works](#how-it-works)
5. [File Structure](#file-structure)
6. [Usage Examples](#usage-examples)
7. [Troubleshooting](#troubleshooting)
8. [Customization](#customization)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r simple_requirements.txt
```

### 2. Run the Application
```bash
python simple_app.py
```

### 3. Open Your Browser
Go to: http://localhost:5000

### 4. Start Using
1. Upload PDF or TXT files by dragging them to the upload area
2. Ask questions about your documents in the chat box
3. Get AI-powered answers based on your content

## 📚 Libraries Used

### Core Libraries

#### 1. **Flask** (`flask`)
```python
from flask import Flask, request, jsonify, render_template
```
**What it does**: Web framework for creating the web application
**Why we use it**: 
- Provides HTTP routing (handles web requests)
- Renders HTML templates
- Manages file uploads
- Returns JSON responses for API calls

**How it works**: Creates a web server that listens for requests and responds with web pages or data.

#### 2. **PyPDF2** (`PyPDF2`)
```python
import PyPDF2
```
**What it does**: Extracts text from PDF files
**Why we use it**: 
- PDFs are binary files that need special parsing
- Converts PDF content into readable text
- Handles different PDF formats and structures

**How it works**: Opens PDF files in binary mode and extracts text from each page.

#### 3. **SentenceTransformers** (`sentence_transformers`)
```python
from sentence_transformers import SentenceTransformer
```
**What it does**: Converts text into numerical vectors (embeddings)
**Why we use it**: 
- Enables semantic search (finding meaning, not just keywords)
- Converts text to numbers that AI can understand
- Allows similarity comparison between documents

**How it works**: Uses pre-trained neural networks to convert sentences into 384-dimensional vectors that represent meaning.

#### 4. **FAISS** (`faiss`)
```python
import faiss
```
**What it does**: Fast similarity search and clustering of dense vectors
**Why we use it**: 
- Extremely fast search through thousands of documents
- Efficiently finds similar content
- Optimized for AI/ML applications

**How it works**: Creates an index of vectors and uses advanced algorithms to quickly find the most similar vectors to a query.

#### 5. **Groq** (`groq`)
```python
from groq import Groq
```
**What it does**: Provides access to large language models (LLMs) via API
**Why we use it**: 
- Generates human-like responses
- Understands context and questions
- Provides intelligent answers based on documents

**How it works**: Sends text prompts to powerful AI models and receives generated responses.

### Supporting Libraries

#### 6. **JSON** (`json`)
```python
import json
```
**What it does**: Handles data serialization and storage
**Why we use it**: 
- Saves document metadata in readable format
- Easy to debug and modify
- Standard format for data exchange

#### 7. **OS** (`os`)
```python
import os
```
**What it does**: File system operations
**Why we use it**: 
- Creates directories
- Checks if files exist
- Manages file paths
- Handles file operations

## 🔧 Code Explanation

### Application Structure

The application is organized into clear sections:

```python
# =============================================================================
# CONFIGURATION - Easy to understand settings
# =============================================================================
```

#### 1. **Configuration Section**
```python
GROQ_API_KEY = "your-api-key-here"
GROQ_MODEL = "llama-3.3-70b-versatile"
UPLOAD_FOLDER = "uploads"
INDEX_FILE = "search_index.bin"
METADATA_FILE = "documents.json"
```

**What**: Stores all configuration settings
**Why**: Centralized location for easy modification
**How**: Simple variables that control the application behavior

#### 2. **Initialization Section**
```python
app = Flask(__name__)
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
groq_client = Groq(api_key=GROQ_API_KEY)
```

**What**: Sets up the core components
**Why**: 
- Flask creates the web server
- SentenceTransformer loads the AI model for text understanding
- Groq client connects to the AI service

**How**: Each component is initialized once at startup for efficiency.

#### 3. **Document Processing Section**

##### PDF Text Extraction
```python
def extract_pdf_text(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text.strip()
```

**What**: Converts PDF files into plain text
**Why**: PDFs are binary files that need special processing
**How**: 
1. Opens file in binary mode (`'rb'`)
2. Creates a PDF reader object
3. Iterates through each page
4. Extracts text from each page
5. Combines all text into one string

##### TXT Text Extraction
```python
def extract_txt_text(file_path):
    text = ""
    try:
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    text = file.read()
                break
            except UnicodeDecodeError:
                continue
    except Exception as e:
        print(f"Error reading TXT file: {e}")
    return text.strip()
```

**What**: Handles various text file encodings
**Why**: Different text files use different character encodings
**How**: 
1. Tries multiple common encodings
2. Uses the first one that works
3. Handles encoding errors gracefully

##### Document Addition
```python
def add_document(filename, text):
    global documents, search_index
    
    doc_id = len(documents)
    document = {
        "id": doc_id,
        "filename": filename,
        "text": text,
        "title": filename
    }
    
    documents.append(document)
    
    if search_index is None:
        embeddings = embedding_model.encode([text])
        search_index = faiss.IndexFlatL2(embeddings.shape[1])
        search_index.add(embeddings)
    else:
        embeddings = embedding_model.encode([text])
        search_index.add(embeddings)
    
    save_documents()
    faiss.write_index(search_index, INDEX_FILE)
```

**What**: Adds a new document to the system
**Why**: Each document needs to be processed and indexed for search
**How**: 
1. Creates a document object with metadata
2. Converts text to embeddings using AI model
3. Creates or updates the search index
4. Saves everything to disk

#### 4. **Search Function**
```python
def search_documents(query, max_results=3):
    if search_index is None or len(documents) == 0:
        return []
    
    query_embedding = embedding_model.encode([query])
    scores, indices = search_index.search(query_embedding, max_results)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(documents):
            results.append({
                "document": documents[idx],
                "score": float(score)
            })
    
    return results
```

**What**: Finds documents similar to a query
**Why**: Enables semantic search (finding meaning, not just keywords)
**How**: 
1. Converts query to embedding vector
2. Searches the FAISS index for similar vectors
3. Returns documents with similarity scores
4. Lower scores = more similar content

#### 5. **AI Response Generation**
```python
def get_ai_response(query, relevant_docs):
    if groq_client is None:
        return create_simple_response(query, relevant_docs)
    
    context = ""
    for result in relevant_docs:
        doc = result["document"]
        context += f"Document: {doc['title']}\nContent: {doc['text'][:500]}...\n\n"
    
    prompt = f"""Based on these documents, answer the question: {query}

Documents:
{context}

Answer:"""
    
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message.content
```

**What**: Generates AI responses using Groq
**Why**: Provides intelligent, contextual answers
**How**: 
1. Builds context from relevant documents
2. Creates a prompt for the AI model
3. Sends prompt to Groq API
4. Returns the generated response

#### 6. **Web Routes**

##### File Upload Route
```python
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file selected"}), 400
    
    file = request.files['file']
    filename_lower = file.filename.lower()
    
    if not (filename_lower.endswith('.pdf') or filename_lower.endswith('.txt')):
        return jsonify({"error": "Only PDF and TXT files are supported"}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    if filename_lower.endswith('.pdf'):
        text = extract_pdf_text(file_path)
    else:
        text = extract_txt_text(file_path)
    
    add_document(file.filename, text)
    return jsonify({"message": f"Successfully uploaded {file.filename}"})
```

**What**: Handles file uploads from the web interface
**Why**: Provides a web API for uploading documents
**How**: 
1. Validates file presence and type
2. Saves file to uploads folder
3. Extracts text based on file type
4. Adds document to the system
5. Returns success/error message

##### Chat Route
```python
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query', '')
    
    relevant_docs = search_documents(query)
    answer = get_ai_response(query, relevant_docs)
    
    return jsonify({
        "answer": answer,
        "documents_found": len(relevant_docs)
    })
```

**What**: Handles chat messages and returns AI responses
**Why**: Provides the main chat functionality
**How**: 
1. Receives question from frontend
2. Searches for relevant documents
3. Generates AI response
4. Returns answer and metadata

##### Delete Route
```python
@app.route('/delete/<int:doc_id>', methods=['DELETE'])
def delete_document_route(doc_id):
    success, message = delete_document(doc_id)
    
    if success:
        return jsonify({"message": message})
    else:
        return jsonify({"error": message}), 400
```

**What**: Handles document deletion
**Why**: Allows users to remove unwanted documents
**How**: 
1. Receives document ID from frontend
2. Calls delete function
3. Returns success/error message

## 🔄 How It Works

### 1. **Document Upload Process**
```
User uploads file → File saved to disk → Text extracted → 
Text converted to embeddings → Added to search index → 
Metadata saved to JSON file
```

### 2. **Question Answering Process**
```
User asks question → Question converted to embedding → 
Search index finds similar documents → Relevant text sent to AI → 
AI generates answer → Answer returned to user
```

### 3. **Search Index Management**
```
First document → Create new FAISS index
Additional documents → Add to existing index
Document deletion → Rebuild entire index
```

## 📁 File Structure

```
rag/
├── simple_app.py              # Main application (275 lines)
├── templates/
│   └── simple_index.html      # Web interface (401 lines)
├── uploads/                   # Uploaded documents
│   ├── Albin Xavier AI.pdf
│   ├── Mobile_Operation_Manual.txt
│   └── Xjdj (1).pdf
├── simple_requirements.txt    # Python dependencies
├── SIMPLE_README.md          # User documentation
├── setup.py                  # Automated setup script
├── documents.json            # Document metadata
├── search_index.bin          # FAISS search index
└── venv/                     # Python virtual environment
```

## 💡 Usage Examples

### Example 1: Uploading Documents
```javascript
// Frontend JavaScript
const formData = new FormData();
formData.append('file', pdfFile);

fetch('/upload', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data.message));
```

### Example 2: Asking Questions
```javascript
// Frontend JavaScript
fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query: "What is this document about?"})
})
.then(response => response.json())
.then(data => console.log(data.answer));
```

### Example 3: Deleting Documents
```javascript
// Frontend JavaScript
fetch('/delete/0', {
    method: 'DELETE'
})
.then(response => response.json())
.then(data => console.log(data.message));
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. **"AI not available"**
- **Cause**: No internet connection or invalid API key
- **Solution**: Check internet connection and API key

#### 2. **"Could not extract text from PDF"**
- **Cause**: Corrupted PDF or password-protected file
- **Solution**: Try a different PDF file

#### 3. **"Only PDF and TXT files are supported"**
- **Cause**: Uploading unsupported file type
- **Solution**: Convert file to PDF or TXT format

#### 4. **"No relevant documents found"**
- **Cause**: No documents uploaded or query too specific
- **Solution**: Upload more documents or try broader questions

### Debug Mode
```python
# Enable debug mode in simple_app.py
app.run(debug=True, host='0.0.0.0', port=5000)
```

## 🎨 Customization

### 1. **Change AI Model**
```python
# In simple_app.py, line 20
GROQ_MODEL = "llama-3.3-70b-versatile"  # Change this
```

### 2. **Modify Upload Folder**
```python
# In simple_app.py, line 23
UPLOAD_FOLDER = "uploads"  # Change this
```

### 3. **Adjust Search Results**
```python
# In search_documents function
def search_documents(query, max_results=3):  # Change max_results
```

### 4. **Customize AI Responses**
```python
# In get_ai_response function
prompt = f"""Your custom prompt here: {query}
Documents: {context}
Answer:"""
```

### 5. **Change Embedding Model**
```python
# In simple_app.py, line 38
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# Change to: 'sentence-transformers/all-mpnet-base-v2' for better quality
```

## 🔒 Security Considerations

### 1. **API Key Protection**
- Never commit API keys to version control
- Use environment variables in production
- Rotate keys regularly

### 2. **File Upload Security**
- Validate file types on server side
- Limit file sizes
- Scan uploaded files for malware

### 3. **Input Validation**
- Sanitize user inputs
- Validate JSON data
- Handle errors gracefully

## 📈 Performance Optimization

### 1. **Index Optimization**
- Use FAISS GPU index for large datasets
- Implement index compression
- Cache embeddings

### 2. **Memory Management**
- Process large files in chunks
- Implement document pagination
- Use streaming responses

### 3. **Caching**
- Cache AI responses
- Store embeddings in database
- Implement Redis caching

## 🚀 Deployment

### 1. **Local Development**
```bash
python simple_app.py
```

### 2. **Production Deployment**
```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 simple_app:app

# Using Docker
docker build -t rag-app .
docker run -p 5000:5000 rag-app
```

### 3. **Environment Variables**
```bash
export GROQ_API_KEY="your-api-key"
export FLASK_ENV="production"
export UPLOAD_FOLDER="/app/uploads"
```

## 📚 Learning Resources

### 1. **Flask Documentation**
- [Flask Official Docs](https://flask.palletsprojects.com/)
- [Flask Tutorial](https://flask.palletsprojects.com/tutorial/)

### 2. **AI/ML Libraries**
- [SentenceTransformers](https://www.sbert.net/)
- [FAISS Documentation](https://faiss.ai/)
- [Groq API Docs](https://console.groq.com/docs)

### 3. **Web Development**
- [HTML/CSS/JavaScript](https://developer.mozilla.org/)
- [Fetch API](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Flask** - Web framework
- **SentenceTransformers** - Text embeddings
- **FAISS** - Vector search
- **Groq** - AI language models
- **PyPDF2** - PDF processing

---

**Happy Document Chatting!** 🎉

For more help, check the `SIMPLE_README.md` file or run `python setup.py` for automated setup.

#   R A G - R e t r i e v a l - A u g m e n t e d - G e n e r a t i o n -  
 