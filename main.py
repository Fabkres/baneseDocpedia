from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import threading
import uuid
from mangum import Mangum
from file_processing import get_text_chunks, get_vectorstore
from keywords import extract_keywords
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from keywords import extract_keywords, search_youtube_videos
from MindMap import generate_mind_map_structure, export_mind_map_to_json
from feedback_routes import feedback_router  # Importa o feedback router
from docling.document_converter import DocumentConverter  # Importa o DocumentConverter
import os

# Cria a aplicação FastAPI
app = FastAPI()

# Adiciona middleware para CORS (se necessário para front-end)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estrutura de dados para armazenar as sessões
sessions = {}
lock = threading.Lock()

# Modelo para entrada de dados da API
class QuestionRequest(BaseModel):
    session_id: str
    question: str

# Função para criar a cadeia de conversa
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        memory=memory
    )
    return conversation_chain

@app.get("/")
async def root():
    return {"message": "Bem-vindo ao DocpedIA!"}

# Endpoint para fazer upload de arquivos e criar uma nova sessão

@app.post("/upload/")
async def upload_files(files: List[UploadFile]):
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum upload feito.")

    # Instancia o conversor do docling
    converter = DocumentConverter()

    raw_text = ""
    for file in files:
        # Salva o arquivo carregado temporariamente
        file_content = await file.read()  # Lê o conteúdo do arquivo
        temp_filename = f"temp_{file.filename}"

        with open(temp_filename, "wb") as f:
            f.write(file_content)  # Salva o arquivo temporariamente no sistema

        try:
            result = converter.convert(temp_filename)
            raw_text += result.document.export_to_markdown()
        finally:
            os.remove(temp_filename)

    # Processa o texto convertido (se necessário)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vectorstore)
    session_id = str(uuid.uuid4())

    # Armazena a sessão criada
    sessions[session_id] = {"conversation": conversation_chain}
        # return {"session_id": session_id, "formatado":{result.document.export_to_markdown()}}


    return {"session_id": session_id}

# Endpoint para enviar perguntas
@app.post("/ask/")
async def ask_question(request: QuestionRequest):
    session_id = request.session_id
    question = request.question

    with lock:
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")

    conversation_chain = session["conversation"]
    response = conversation_chain({"question": question})

    return {
        "response": response["answer"],
        "chat_history": response["chat_history"]
    }



@app.post("/generate_mind_map/") 
async def generate_mind_map(files: List[UploadFile]):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    
    # Instancia o conversor do docling
    converter = DocumentConverter()
    
    raw_text = ""
    for file in files:
        if file.filename.endswith(('.pdf', '.docx', '.txt')):
            # Converte o arquivo para Markdown usando o DocumentConverter
            file_content = await file.read()  # Lê o conteúdo do arquivo
            with open(f"temp_{file.filename}", "wb") as f:
                f.write(file_content)

            # Converte o documento para Markdown
            result = converter.convert(f"temp_{file.filename}")
            raw_text += result.document.export_to_markdown()

    # Gera o mapa mental
    mind_map = generate_mind_map_structure(raw_text)
    
    return mind_map

# Endpoint para fazer upload de arquivos e processar palavras-chave
@app.post("/upload_with_keywords/")
async def upload_files_with_keywords(files: List[UploadFile]):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    
    documents_keywords = {}
    youtube_results = {}

    # Instancia o conversor do docling
    converter = DocumentConverter()

    for file in files:
        if file.filename.endswith(('.pdf', '.docx', '.txt')):
            # Converte o arquivo para Markdown usando o DocumentConverter
            file_content = await file.read()  # Lê o conteúdo do arquivo
            with open(f"temp_{file.filename}", "wb") as f:
                f.write(file_content)

            # Converte o documento para Markdown
            result = converter.convert(f"temp_{file.filename}")
            markdown_content = result.document.export_to_markdown()

            # Extraí as palavras-chave do conteúdo markdown
            keywords = extract_keywords(markdown_content)
            documents_keywords[file.filename] = keywords

            # Buscar vídeos no YouTube para as palavras-chave
            youtube_results[file.filename] = search_youtube_videos(keywords)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file.filename}")
    
    return {
        "documents_keywords": documents_keywords,
        "youtube_results": youtube_results,
    }


# Endpoint para sair do chat e apagar o contexto
@app.post("/exit/")
async def exit_chat(request: QuestionRequest):
    session_id = request.session_id

    with lock:
        if session_id in sessions:
            del sessions[session_id]
            return {"message": "Session closed and context cleared."}
        else:
            raise HTTPException(status_code=404, detail="Session not found.")

# Inclui o roteador de feedback
app.include_router(feedback_router)

handler = Mangum(app)
