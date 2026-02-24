import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AIPIPE_API_KEY = os.environ.get("AIPIPE_API_KEY")
AIPIPE_URL = "https://aipipe.org/openai/v1/responses"  # <-- IMPORTANT


class CommentRequest(BaseModel):
    comment: str


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/comment")
async def analyze_comment(data: CommentRequest):

    if not data.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    if not AIPIPE_API_KEY:
        raise HTTPException(status_code=500, detail="API key missing")

    headers = {
        "Authorization": f"Bearer {AIPIPE_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "gpt-4.1-mini",
        "input": f"Analyze sentiment and rate 1-5. Comment: {data.comment}",
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

    try:
        response = requests.post(
            AIPIPE_URL,
            headers=headers,
            json=body,
            timeout=8   # prevent Vercel timeout
        )

        if response.status_code != 200:
            return {
                "sentiment": "neutral",
                "rating": 3
            }

        result = response.json()

        # SAFE extraction
        try:
            structured_output = result["output"][0]["content"][0]["parsed"]
            return structured_output
        except:
            return {
                "sentiment": "neutral",
                "rating": 3
            }

    except Exception:
        return {
            "sentiment": "neutral",
            "rating": 3
        }
