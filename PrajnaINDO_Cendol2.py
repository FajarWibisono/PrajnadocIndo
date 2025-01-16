import streamlit as st
import os
import tempfile
import PyPDF2
import docx
import csv
from io import StringIO

# Pastikan environment sudah terpasang di terminal/CLI (BUKAN di dalam Python):
#   pip install streamlit sentence-transformers faiss-cpu transformers PyPDF2 python-docx

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ----
# 1. SETUP MODEL EMBEDDINGS (INDOBERT LITE) DAN MODEL GENERASI (INDOT5)
# ----
# Embedding model yang fokus pada Bahasa Indonesia
EMBEDDING_MODEL_NAME = "indobenchmark/indobert-lite-base-p2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Inisialisasi FAISS index untuk menampung embedding dokumen
dimension = embedding_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatIP(dimension)   # IP = inner product
texts_data = []      # Menyimpan potongan (chunk) teks
embedding_data = None  # Menyimpan embedding chunk dalam bentuk numpy array

# Model generatif Bahasa Indonesia berbasis T5 (seq2seq)
# Pastikan model ini tersedia di Hugging Face (public)
GENERATOR_MODEL_NAME = "indobenchmark/indot5-base-generator"

# Load tokenizer & model
indo_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
indo_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL_NAME)

# Pipeline "text2text-generation" untuk seq2seq T5
indo_pipeline = pipeline(
    "text2text-generation",
    model=indo_model,
    tokenizer=indo_tokenizer
)

# ----
# 2. FUNGSI EKSTRAK TEKS DARI FILE
# ----

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    lines = [para.text for para in doc.paragraphs]
    return "\n".join(lines)

def extract_text_from_csv(file):
    content = file.read()
    file.seek(0)
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="ignore")
    data = []
    reader = csv.reader(StringIO(content), delimiter=",")
    for row in reader:
        data.append(" ".join(row))
    return "\n".join(data)

def extract_text_from_txt(file):
    content = file.read()
    file.seek(0)
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="ignore")
    return content

def chunk_text(text, chunk_size=500):
    """
    Membagi teks menjadi potongan (chunk) berukuran ~500 kata.
    Sesuaikan chunk_size dengan kebutuhan.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

# ----
# 3. FUNGSI UNTUK MEMPROSES DAN INDEX FILE
# ----

def process_and_index_file(uploaded_file):
    global embedding_data

    # Simpan sementara
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_filepath = tmp_file.name

    # Ekstraksi teks sesuai tipe file
    if uploaded_file.name.lower().endswith(".pdf"):
        extracted_text = extract_text_from_pdf(tmp_filepath)
    elif uploaded_file.name.lower().endswith(".docx"):
        extracted_text = extract_text_from_docx(tmp_filepath)
    elif uploaded_file.name.lower().endswith(".csv"):
        extracted_text = extract_text_from_csv(uploaded_file)
    elif uploaded_file.name.lower().endswith(".txt"):
        extracted_text = extract_text_from_txt(uploaded_file)
    else:
        st.warning("Format file tidak dikenal. Hanya PDF, DOCX, CSV, dan TXT.")
        os.remove(tmp_filepath)
        return

    os.remove(tmp_filepath)

    # Membagi teks menjadi chunk
    chunks = chunk_text(extracted_text)

    # Embedding chunk
    chunk_embeddings = embedding_model.encode(chunks)

    # Update data di global scope
    global texts_data
    start_idx = len(texts_data)
    texts_data.extend(chunks)

    if embedding_data is None:
        embedding_data = chunk_embeddings
    else:
        embedding_data = np.vstack((embedding_data, chunk_embeddings))

    # Tambahkan embeddings ke index FAISS
    index.add(chunk_embeddings)

# ----
# 4. RETRIEVAL & GENERATE ANSWER
# ----

def search_relevant_chunks(query, top_k=3):
    """
    Mencari top_k chunk paling relevan dari koleksi dokumen
    menggunakan kesamaan vektor (FAISS).
    """
    query_vec = embedding_model.encode([query])
    distances, indices = index.search(query_vec, top_k)
    results = []
    for idx_row in indices:
        for idx in idx_row:
            if idx < len(texts_data):
                results.append(texts_data[idx])
    return results

def generate_answer(query):
    """
    1. Lakukan retrieval chunk.
    2. Feed chunk + query ke model generatif (T5) untuk jawaban final.
    """
    relevant_chunks = search_relevant_chunks(query, top_k=3)
    if not relevant_chunks:
        return "Maaf, tidak ada data relevan yang ditemukan."

    context_text = "\n\n".join(relevant_chunks)
    # Prompt/format sederhana, dapat dioptimalkan
    prompt = (
        f"Dokumen referensi:\n{context_text}\n\n"
        f"Pertanyaan: {query}\n\n"
        f"Jawablah secara singkat dan relevan dalam Bahasa Indonesia."
    )

    # Gunakan pipeline text2text-generation
    result = indo_pipeline(prompt, max_length=512, do_sample=False)
    final_answer = result[0]["generated_text"]
    return final_answer

# ----
# 5. STREAMLIT APP
# ----

def main():
    st.set_page_config(page_title="PrajnaDoc Indo", layout="wide")

    st.title("PrajnaDoc Indo")
    st.markdown(
        "<b>Selamat datang di PrajnaDoc Indo!</b><br><br>"
        "Aplikasi ini adalah chatbot RAG (Retrieval-Augmented Generation) "
        "untuk dokumen-dokumen berbahasa Indonesia. "
        "Silakan upload dokumen Anda (PDF, DOCX, CSV, TXT) di sidebar, "
        "lalu ajukan pertanyaan di kolom tersedia. "
        "Tekan tombol 'Cari/Chat' untuk mendapatkan jawaban dari model T5. "
        "<br><br>Terima kasih telah mencoba!",
        unsafe_allow_html=True,
    )

    # Sidebar Upload
    st.sidebar.header("Upload Dokumen Anda")
    uploaded_files = st.sidebar.file_uploader(
        "Pilih File (PDF, DOCX, CSV, TXT):",
        type=["pdf", "docx", "csv", "txt"],
        accept_multiple_files=True
    )
    if uploaded_files:
        for file in uploaded_files:
            process_and_index_file(file)
        st.sidebar.success("Berhasil memproses file!")

    query = st.text_input("Masukkan pertanyaan atau kata kunci Anda...")
    if st.button("Cari/Chat"):
        if not query.strip():
            st.warning("Mohon masukkan pertanyaan.")
        else:
            if index.ntotal == 0:
                st.warning("Belum ada dokumen yang diunggah untuk diindeks.")
            else:
                with st.spinner("Sedang mencari jawaban..."):
                    response = generate_answer(query)
                    st.success(response)

if __name__ == "__main__":
    main()