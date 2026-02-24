import os
import json
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AIPIPE_API_KEY = os.environ.get("AIPIPE_API_KEY")
AIPIPE_URL = "https://aipipe.org/openai/v1/chat/completions"


class CommentRequest(BaseModel):
    comment: str


@app.post("/comment")
async def analyze_comment(data: CommentRequest):

    if not data.comment.strip():
        return {"sentiment": "neutral", "rating": 3}

    headers = {
        "Authorization": f"Bearer {AIPIPE_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "gpt-4.1-mini",
        "messages": [
            {
                "role": "system",
                "content": """
You are a strict sentiment classifier.

Rules:
positive → clearly satisfied
negative → clearly dissatisfied
neutral → mixed opinion

Rating:
5 = extremely positive
4 = positive
3 = neutral
2 = negative
1 = extremely negative

Respond ONLY in this exact JSON format:
{"sentiment":"positive","rating":5}
"""
            },
            {
                "role": "user",
                "content": data.comment
            }
        ],
        "temperature": 0
    }

    try:
        response = requests.post(
            AIPIPE_URL,
            headers=headers,
            json=body,
            timeout=8
        )

        if response.status_code != 200:
            return {"sentiment": "neutral", "rating": 3}

        result = response.json()

        content = result["choices"][0]["message"]["content"]

        # Extract JSON safely
        try:
            parsed = json.loads(content)
            return {
                "sentiment": parsed["sentiment"],
                "rating": parsed["rating"]
            }
        except:
            return {"sentiment": "neutral", "rating": 3}

    except Exception:
        return {"sentiment": "neutral", "rating": 3}
