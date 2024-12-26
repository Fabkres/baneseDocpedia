import json
from fastapi import APIRouter, HTTPException
from datetime import datetime
from pydantic import BaseModel
from pathlib import Path

# Modelo para o feedback
class FeedbackRequest(BaseModel):
    document_id: str
    conversation_id: str
    feedback: bool  # True para positivo, False para negativo

# Caminho para o arquivo de feedback
FEEDBACK_FILE = Path("feedback.json")

# Inicializa o roteador
feedback_router = APIRouter()

# Função para salvar feedback no arquivo
def save_feedback(data: list):
    with open(FEEDBACK_FILE, 'w') as file:
        json.dump(data, file, indent=4)

def load_feedback():
    if FEEDBACK_FILE.exists() and FEEDBACK_FILE.stat().st_size > 0:  # Verifica se o arquivo existe e não está vazio
        with open(FEEDBACK_FILE, 'r') as file:
            return json.load(file)
    return []  # Retorna uma lista vazia caso o arquivo não exista ou esteja vazio


# Endpoint para criar um feedback
@feedback_router.post("/feedback/")
async def create_feedback(feedback_request: FeedbackRequest):
    feedback_data = {
        "id": str(len(load_feedback()) + 1),  # Gera um ID único
        "document_id": feedback_request.document_id,
        "conversation_id": feedback_request.conversation_id,
        "feedback": feedback_request.feedback,
        "timestamp": datetime.utcnow().isoformat()
    }

    feedback_list = load_feedback()
    feedback_list.append(feedback_data)

    save_feedback(feedback_list)

    return {"message": "Feedback recebido com sucesso!", "feedback_id": feedback_data["id"]}

# Endpoint para ler todos os feedbacks
@feedback_router.get("/feedbacks/")
async def get_feedbacks():
    feedback_list = load_feedback()
    return {"feedbacks": feedback_list}

# Endpoint para ler um feedback específico por ID
@feedback_router.get("/feedback/{feedback_id}")
async def get_feedback(feedback_id: str):
    feedback_list = load_feedback()
    feedback = next((item for item in feedback_list if item["id"] == feedback_id), None)

    if feedback is None:
        raise HTTPException(status_code=404, detail="Feedback not found")
    
    return {"feedback": feedback}

# Função para atualizar um feedback
@feedback_router.put("/feedback/{conversation_id}")
async def update_feedback(conversation_id: str, feedback_request: FeedbackRequest):
    feedback_list = load_feedback()  # Carrega os feedbacks do arquivo

    # Encontra o feedback com a conversation_id fornecida
    feedback = next((item for item in feedback_list if item["conversation_id"] == conversation_id), None)

    if feedback is None:
        raise HTTPException(status_code=404, detail="Feedback not found")

    # Atualiza os campos do feedback
    feedback["document_id"] = feedback_request.document_id  # Atualiza o ID do documento
    feedback["feedback"] = feedback_request.feedback  # Atualiza o feedback (True ou False)
    feedback["timestamp"] = datetime.utcnow().isoformat()  # Atualiza a data do feedback

    # Salva os feedbacks atualizados no arquivo
    save_feedback(feedback_list)

    return {"message": "Feedback atualizado com sucesso!", "feedback": feedback}


# Endpoint para excluir um feedback
@feedback_router.delete("/feedback/{feedback_id}")
async def delete_feedback(feedback_id: str):
    feedback_list = load_feedback()
    feedback = next((item for item in feedback_list if item["id"] == feedback_id), None)

    if feedback is None:
        raise HTTPException(status_code=404, detail="Feedback not found")

    # Remove o feedback da lista
    feedback_list = [item for item in feedback_list if item["id"] != feedback_id]
    save_feedback(feedback_list)

    return {"message": "Feedback excluído com sucesso!"}
