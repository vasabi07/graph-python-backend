from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from prisma import Prisma
from graph import graph
from pydantic_models import Query
app = FastAPI()

origins = [
    'http://localhost:3000'
]

@app.get('/')
def home():
    return {"message":"Welcome to the API"}


@app.post('/create-user')
async def create_user():
    db = Prisma()
    await db.connect()
    response = await db.user.create(data={"name":"John Doe","email":"john@gmail.com","password":"password"})
    return response
#send question to the graph in this endpoint and return the answer back to the user
@app.post('/start-workflow')
async def start_workflow(input_data: Query):
    query = input_data.query
    response =await graph.ainvoke({"question": query})
    print(response["answer"].content)
    return response["answer"].content

    