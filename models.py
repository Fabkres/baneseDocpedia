from pydantic import BaseModel

class FeedbackRequest(BaseModel):
    session_id: str
    document_id: str
    like: bool
