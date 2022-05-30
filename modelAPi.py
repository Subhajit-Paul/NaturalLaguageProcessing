from ast import While
from fastapi import FastAPI, WebSocket
from numpy import array
import numpy as np
import joblib
from keras.models import model_from_json
import json
import ast


token = joblib.load(r'F:\FastAPI Testing\API TEST\models\tokenizer_of_text_test')
with open(r'F:\FastAPI Testing\API TEST\models\model_arch.txt', 'r') as m:
    modelt = m.read()
model = model_from_json(modelt)
model.load_weights(r'F:\FastAPI Testing\API TEST\models\weightsfortest.h5')

diction = {}

def defineNew(val):
    for i in range(len(val) - 1):
        flag = False
        if val[i] not in diction.keys():
            diction[val[i]] = [[val[i+1]]]
        else:
            d = diction.get(val[i])
            dlist = [j for j in d]
            for vald in dlist:
                if val[i + 1] in vald:
                    flag = True
                    break
            if(not flag): d.append([val[i+1]])

def getPredictions(text, model = model, tokenizer = token):
    in_text = text
    encoded = tokenizer.texts_to_sequences([in_text])[0]
    encoded = array(encoded)
    if encoded[0] == 1:
        return [{'word':'No Results Found From model!!','pred': 0}]
    else :
        Yhat = model.predict(encoded)
        outword = []
        i = 0
        for word, _ in tokenizer.word_index.items():
            temp = []
            if Yhat[0][i+1] > 0.0005 and len(word) >= [1, 2, 3, 4, 5][np.random.randint(0, 4)]:
                temp.append(word)
                temp.append(round(Yhat[0][i+1]*100, 2))
                outword.append(temp)
            i += 1
        outword = np.array(outword)
        listOfWords = sorted(outword, key=lambda row: (row[1]), reverse=True)
        listOfWords = listOfWords[:10]
        return [{'word':i[0]} for i in listOfWords]


def fromDiction(val):
    arr = diction.get(val)
    if arr is None: return []
    else: 
        return [{'word': i[0]} for i in arr] 
    

app = FastAPI()

@app.websocket("/savetovar")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        defineNew(data.split())       


@app.websocket("/rw")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        if data == "read":
            with open(r"diction.txt", "r") as f:
                try:
                    fr = f.read()
                    global diction
                    diction = ast.literal_eval(fr)
                except Exception:
                    pass
        if data == "write":
            if diction:
                with open(r"diction.txt", "w") as f:
                    f.write(f"{diction}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        model1 = getPredictions(data)
        diction1 = fromDiction(data)
               
        result = diction1 + model1
        await websocket.send_text(json.dumps(result))
