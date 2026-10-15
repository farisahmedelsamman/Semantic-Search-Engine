import os
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# =====================================================================
# Task 1: Load Documents
# قراءة ملفات الـ PDF واستخراج النصوص منها مع حفظ اسم الملف لكل نص.
# =====================================================================
print("--- Starting Task 1: Load Documents ---")

def load_documents_from_folder(folder_path):
    documents = {} # { "filename.pdf": "extracted text..." }
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            documents[filename] = text
            print(f"Loaded {filename} successfully.")
            
    return documents

# تحديد المسار لمجلد البيانات الموضح في الصورة[cite: 4]
data_folder = "C:/Users/HP/Desktop/Fares/Andalusia/Assignment 1/data"
raw_documents = load_documents_from_folder(data_folder)


# =====================================================================
# Task 2: Implement Text Chunking
# تقسيم النصوص إلى مقاطع أصغر (Chunks) لتسهيل عمليات البحث والتمثيل المتجهي.
# =====================================================================
print("\n--- Starting Task 2: Implement Text Chunking ---")

def chunk_text(text, chunk_size=300, overlap=50):
    """
    تقسيم النص إلى مقاطع بناءً على عدد الكلمات مع وجود تداخل (overlap) 
    لضمان عدم ضياع المعنى بين الأجزاء.
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        
    return chunks

# تطبيق دالة الـ Chunking وتخزين كل Chunk مع اسم الملف المصدر
all_chunks = []      # القائمة التي ستخزن النصوص المقسمة
chunk_metadata = []  # القائمة التي ستخزن اسم الملف المقابل لكل Chunk

for doc_name, doc_text in raw_documents.items():
    chunks = chunk_text(doc_text, chunk_size=200, overlap=30)
    for chunk in chunks:
        all_chunks.append(chunk)
        chunk_metadata.append(doc_name)

print(f"Total chunks created across all documents: {len(all_chunks)}")


# =====================================================================
# Task 3: Generate Text Embeddings
# تحويل مقاطع النصوص إلى المتجهات (Vectors) باستخدام SentenceTransformer[cite: 1].
# =====================================================================
print("\n--- Starting Task 3: Generate Text Embeddings ---")

# تحميل نموذج Sentence Transformer المناسب للبحث الدلالي[cite: 1]
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# تحويل جميع الـ Chunks إلى Embeddings[cite: 1]
chunk_embeddings = embedding_model.encode(all_chunks, show_progress_bar=True)
chunk_embeddings = np.array(chunk_embeddings, dtype=np.float32)


# =====================================================================
# Task 4: Build FAISS Vector Database
# تطبيع المتجهات (Normalization) وإنشاء فهرس FAISS لتخزين المتجهات.
# =====================================================================
print("\n--- Starting Task 4: Build FAISS Vector Database ---")

# 1. تطبيع المتجهات (Normalization) لتمكين Cosine Similarity
faiss.normalize_L2(chunk_embeddings)

# 2. تحديد أبعاد المتجهات وإنشاء الفهرس (FAISS Index)[cite: 2]
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner Product بعد الـ Normalization يعطي Cosine Similarity

# 3. إضافة المتجهات إلى الفهرس[cite: 2]
index.add(chunk_embeddings)
print(f"Stored {index.ntotal} vectors in FAISS index.")


# =====================================================================
# Task 5: Implement Search Function
# دالة semantic_search تحول الاستعلام لمتجه وتبحث في FAISS عن أفضل التطابقات[cite: 2].
# =====================================================================
print("\n--- Starting Task 5: Implement Search Function ---")

def semantic_search(query, top_k=2):
    # 1. تحويل الاستعلام إلى Embedding[cite: 2]
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)
    
    # 2. تطبيع متجه الاستعلام[cite: 2]
    faiss.normalize_L2(query_embedding)
    
    # 3. البحث في FAISS[cite: 2]
    scores, indices = index.search(query_embedding, top_k)
    
    # 4. إرجاع النتائج[cite: 2]
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "pdf_name": chunk_metadata[idx],
            "chunk_text": all_chunks[idx],
            "score": float(score)
        })
    return results


# =====================================================================
# Task 6: Test the System
# تشغيل حالات الاختبار المذكورة في الاسايمنت للتأكد من صحة النتائج.
# =====================================================================
print("\n--- Starting Task 6: Test the System ---")

test_cases = [
    "How many vacation days do employees receive?",
    "What is the refund policy?",
    "How is customer satisfaction improved?",
    "What metrics evaluate classification models?",
    "Explain transformer architecture."
]

for i, query in enumerate(test_cases, 1):
    print(f"\n--- Test Case {i} ---")
    print(f"Query: {query}")
    search_results = semantic_search(query, top_k=1)
    
    top_match = search_results[0]
    print(f"Matched PDF : {top_match['pdf_name']}")
    print(f"Similarity Score : {top_match['score']:.4f}")
    print(f"Relevant Chunk   : {top_match['chunk_text'][:150]}...")