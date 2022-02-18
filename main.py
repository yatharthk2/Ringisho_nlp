# heroku logs --app profanityringisho --tail
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
from copy import deepcopy
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from enum import Enum
from transformers import AutoTokenizer, T5ForConditionalGeneration

import uvicorn
from profanity_check import predict, predict_prob
import os
import openai

app = FastAPI()

class UserRequestIn(BaseModel):
    text: str
    questionID: str


class profanity(BaseModel):
    profanity: str
    questionID: str

class essay(BaseModel):
    essay: str
    questionID: str

@app.get('/')
async def root():
    return {'message': 'Hello World'}

@app.post('/profanities', response_model=profanity)
def infer_profanity(user_request: UserRequestIn):
    
    text = user_request.text
    questionID = user_request.questionID
    result = predict([text])
    if result[0] == 0:
        output = "No profanity"
    else:
        output = "Profanity"

    return {'profanity':output , 'questionID':questionID}

@app.post('/essay', response_model=essay)
def infer_essay(user_request: UserRequestIn):
    
    text = user_request.text
    questionID = user_request.questionID
    openai.api_key = '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'

    response = openai.Completion.create(
        engine="text-davinci-001",
        prompt=text,
        temperature=0,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
)

    return {'essay':response , 'questionID':questionID}



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5049)









