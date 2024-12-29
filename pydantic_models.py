from pydantic import BaseModel

class Query(BaseModel):
    query: str
    user_id: str
    thread_id: str
    state: dict


