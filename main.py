from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model.bert import extract_keywords_bert_
from model.yark import extract_keywords_yake_
from model.huggingface import keyword_fetch_

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; specify origins if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)


class KeywordRequest(BaseModel):
    article_text: str

class KeywordRequestHF(BaseModel):
    article_text: str
    model_id: str
    secret_key: str


@app.get("/")
async def root():
    return {"message": "Chale chhe"}


@app.post("/extract_keywords/")
async def extract_keywords(request: KeywordRequestHF):
    try:
        keywords = keyword_fetch_(request.model_id, request.secret_key, request.article_text)
        return {"keywords": keywords}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract_keywords_bert/")
async def extract_keywords_bert(request: KeywordRequest):
   try:
        keywords = extract_keywords_bert_(request.article_text)
        return {"keywords": keywords}
   except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract_keywords_yake/")
async def extract_keywords_yake(request: KeywordRequest):
    print(request.article_text)
    keywords = extract_keywords_yake_(request.article_text)
    return {"keywords": keywords}

