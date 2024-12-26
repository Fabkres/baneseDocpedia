from PyPDF2 import PdfReader
import docx
import time  # Import necessário para medir o tempo

def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf.file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_docx_text(docx_files):
    text = ""
    for docx_file in docx_files:
        doc = docx.Document(docx_file.file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

def get_txt_text(txt_files):
    text = ""
    for txt in txt_files:
        text += txt.file.read().decode('utf-8')
    return text

def get_combined_text(files):
    text = ""
    pdf_files = [f for f in files if f.filename.endswith('.pdf')]
    docx_files = [f for f in files if f.filename.endswith('.docx')]
    txt_files = [f for f in files if f.filename.endswith('.txt')]
    
    if pdf_files:
        text += get_pdf_text(pdf_files)
    if docx_files:
        text += get_docx_text(docx_files)
    if txt_files:
        text += get_txt_text(txt_files)
    
    return text

def get_text_chunks(text):
    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def generate_response(query, retriever):
    start_time = time.time()  # Início da medição de tempo
    
    # Obtenha a resposta usando o retriever
    response = retriever.retrieve(query)  # Substitua pelo método correto se necessário
    
    end_time = time.time()  # Fim da medição de tempo
    elapsed_time = end_time - start_time
    
    print(f"Resposta gerada em {elapsed_time:.2f} segundos.")
    
    # Exibir fontes (simulação, ajuste de acordo com o seu modelo real)
    sources = retriever.get_sources(query)  # Simule obtenção de fontes
    for idx, source in enumerate(sources):
        print(f"Fonte {idx + 1}: {source}")

    return response
