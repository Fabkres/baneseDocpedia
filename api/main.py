from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import threading
import uuid
from mangum import Mangum
from api.file_processing import get_combined_text, get_text_chunks, get_vectorstore
from api.keywords import extract_keywords
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from api.keywords import extract_keywords, search_youtube_videos
from api.MindMap import generate_mind_map_structure, export_mind_map_to_json

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

# Endpoint para fazer upload de arquivos e processar palavras-chave
@app.post("/upload_with_keywords/")
async def upload_files_with_keywords(files: List[UploadFile]):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    
    documents_keywords = {}
    youtube_results = {}

    for file in files:
        if file.filename.endswith(('.pdf', '.docx', '.txt')):
            raw_text = get_combined_text([file])
            keywords = extract_keywords(raw_text)
            documents_keywords[file.filename] = keywords

            # Buscar vídeos no YouTube para as palavras-chave
            youtube_results[file.filename] = search_youtube_videos(keywords)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file.filename}")
    
    return {
        "documents_keywords": documents_keywords,
        "youtube_results": youtube_results,
    }


# Endpoint para fazer upload de arquivos e criar uma nova sessão
@app.post("/upload/")
async def upload_files(files: List[UploadFile]):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    raw_text = get_combined_text(files)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vectorstore)
    session_id = str(uuid.uuid4())
    
    with lock:
        sessions[session_id] = {"conversation": conversation_chain}

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
    
    # Combina texto dos arquivos
    raw_text = get_combined_text(files)
    
    # Gera o mapa mental
    mind_map = generate_mind_map_structure(raw_text)
    
    # Exporta o mapa mental como JSON (opcional)
    # export_mind_map_to_json(mind_map, output_file="mind_map.json")
    
    return mind_map


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


handler = Mangum(app)
