from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tldextract

app = FastAPI()

class PredictRequest(BaseModel):
    title: str
    url: str | None = None
    user: str | None = None

@app.post("/api/predict")
def predict(data: PredictRequest):
    if not data.title:
        raise HTTPException(status_code=400, detail="Title is required")

    user = data.user or "<unk>"
    url = data.url or "<unk>"

    if url != "<unk>":
        ext = tldextract.extract(url)
        domain = f"{ext.domain}.{ext.suffix}" if ext.domain and ext.suffix else "<unk>"
    else:
        domain = "<unk>"

    return {
        "title": data.title,
        "user": user,
        "domain": domain,
        "prediction": 1  # Stub for actual model output
    }

