import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

DEFAULT_FS_MODEL = os.getenv("SMART_MODEL") or os.getenv("DEFAULT_MODEL") or "gpt-5-mini"

def create_vector_store(name: str):
    vs = client.vector_stores.create(name=name)
    return vs


def upload_files_to_vector_store(vector_store_id: str, paths: List[str]) -> List[str]:
    file_ids: List[str] = []
    for path in paths:
        with open(path, "rb") as f:
            vs_file = client.vector_stores.files.upload_and_poll(
                vector_store_id=vector_store_id,
                file=f,
            )
        fid = getattr(vs_file, "id", None) or getattr(vs_file, "file_id", None)
        if fid:
            file_ids.append(fid)
    return file_ids


def ask_with_file_search(vector_store_id: str, question: str, model: str | None = None) -> str:
    model_used = model or DEFAULT_FS_MODEL

    resp = client.responses.create(
        model=model_used,
        tools=[{
            "type": "file_search",
            "vector_store_ids": [vector_store_id],
        }],
        input=question,
    )
    return model_used, (resp.output_text or "")
