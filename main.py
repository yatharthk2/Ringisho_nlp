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

from response_model_class import UserRequestIn, profanity, essay , Info
from Unsplash import unsplash




app = FastAPI()



@app.get('/')
async def root():
    return {'welcome to ringisho_AI'}

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

@app.post("/image")
def read_root(info : Info):

    best_photo_id = unsplash().doStuff(info.question)

    photo_image_url_1 = "https://unsplash.com/photos/"
    photo_image_url_2 = "/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjQ0ODE4NTk4&force=true"

    return {
        
        "url" : photo_image_url_1 + best_photo_id[0] + photo_image_url_2,
        'questionID':info.question_id
    }



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5049)









