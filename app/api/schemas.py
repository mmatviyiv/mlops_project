from pydantic import BaseModel


class RefactorRequest(BaseModel):
    code: str
    

class RefactorResponse(BaseModel):
    refactored_code: str
