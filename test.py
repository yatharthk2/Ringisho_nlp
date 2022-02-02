# uvicorn main:app --reload

from copy import deepcopy
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from enum import Enum
from transformers import AutoTokenizer, T5ForConditionalGeneration
app = FastAPI()
import uvicorn



ckpt = 'Narrativa/byt5-base-tweet-hate-detection'

tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = T5ForConditionalGeneration.from_pretrained(ckpt).to("cuda")

def classify_sentence(sentence):
    
    inputs = tokenizer([sentence], padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    input_ids = inputs.input_ids.to('cuda')
    attention_mask = inputs.attention_mask.to('cuda')
    output = model.generate(input_ids, attention_mask=attention_mask)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# output = classify_sentence('This is a test')
# print(output)

class UserRequestIn(BaseModel):
    text: str
    userID: str


class EntityOut(BaseModel):
    start: int
    end: int
    type: str
    text: str



@app.post("/entities", response_model=EntityOut)
def extract_entities(user_request: UserRequestIn):
    
    text = user_request.text
    userID = user_request.userID
    output = classify_sentence(text)

    
    return {"result": output}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5049)






# from typing import Optional

# from fastapi import FastAPI
# from pydantic import BaseModel


# class Item(BaseModel):
#     name: str
#     description: Optional[str] = None
#     price: float
#     tax: Optional[float] = None


# app = FastAPI()


# @app.post("/items/")
# async def create_item(item: Item):
#     return item






