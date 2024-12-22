from PyPDF2 import PdfReader
import docx

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
