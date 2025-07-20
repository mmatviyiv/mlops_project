import logging
from fastapi import FastAPI, HTTPException

from .schemas import RefactorRequest, RefactorResponse
from .model import RefactorModel


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = FastAPI(
    title="Code Refactoring API",
    description="An API to refactor Python code using a local LLM."
)

model_instance = RefactorModel()

@app.post("/refactor", response_model=RefactorResponse)
async def refactor_code(request: RefactorRequest):
    try:
        refactored_text = model_instance.refactor(request.code)
        return RefactorResponse(refactored_code=refactored_text)
    except Exception as e:
        logging.error(f"Error during refactoring: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during model inference.")


@app.get("/health")
async def health_check():
    return {"status": "ok"}
