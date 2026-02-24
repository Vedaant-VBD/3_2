import os
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
AIPIPE_URL = "https://aipipe.org/openai/v1/responses"


class CommentRequest(BaseModel):
    comment: str


@app.post("/comment")
async def analyze_comment(data: CommentRequest):

    if not data.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    headers = {
        "Authorization": f"Bearer {AIPIPE_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "gpt-4.1-mini",
        "input": [
            {
                "role": "system",
                "content": """
You are a strict sentiment classifier.

Rules:
- positive: clearly satisfied or happy
- negative: clearly dissatisfied or unhappy
- neutral: mixed, average, or balanced

Rating scale:
5 = extremely positive
4 = clearly positive
3 = neutral or mixed
2 = clearly negative
1 = extremely negative

Return ONLY valid JSON matching schema.
"""
            },
            {
                "role": "user",
                "content": data.comment
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "sentiment_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "sentiment": {
                            "type": "string",
                            "enum": ["positive", "negative", "neutral"]
                        },
                        "rating": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5
                        }
                    },
                    "required": ["sentiment", "rating"],
                    "additionalProperties": False
                }
            }
        }
    }

    response = requests.post(
        AIPIPE_URL,
        headers=headers,
        json=body,
        timeout=8
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=response.text)

    result = response.json()

    # STRICT extraction (no neutral fallback)
    try:
        return result["output"][0]["content"][0]["parsed"]
    except Exception:
        raise HTTPException(status_code=500, detail="Invalid AI response")
