from pydantic import BaseModel


class UserRequestIn(BaseModel):
    text: str
    questionID: str


class Profanity(BaseModel):
    profanity: str
    questionID: str

class Essay(BaseModel):
    essay: str
    questionID: str

class image_unsplash(BaseModel):
    user_id : int
    question_id : int
    question: str


class Info(BaseModel):
    user_id : int
    question_id : int
    question: str