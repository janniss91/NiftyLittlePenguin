from fastapi import FastAPI
from pydantic import BaseModel

from niftylittlepenguin.qa.config import INFERENCE_MODEL
from niftylittlepenguin.qa.inference import (get_inference_model,
                                             get_inference_tokenizer)
from niftylittlepenguin.qa.inference import infer as qa_infer
from niftylittlepenguin.qa.read_data import SQuADInstance

app = FastAPI()
model = get_inference_model(INFERENCE_MODEL)
tokenizer = get_inference_tokenizer()


class QARequest(BaseModel):
    context: str
    question: str


@app.get("/")
def endpoint_overview():
    overview = "The inference server currently supports the following endpoints: /qa"
    return {"overview": overview}


@app.post("/qa")
def qa_inference(qa_request: QARequest):
    instance = SQuADInstance(
        title="",
        context=qa_request.context,
        question=qa_request.question,
        answer_starts=[],
        is_impossible=False,
    )

    response = qa_infer(model, tokenizer, instance)
    return response
