from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import json

from .orchestrator import run, run_stream

import os
import tempfile
from typing import List

from .schemas import (
    RunRequest, RunResponse, ToolLog, 
    CreateVectorStoreRequest, CreateVectorStoreResponse,
    UploadToVectorStoreResponse,
    AskFileSearchRequest, AskFileSearchResponse,
)
from .file_search_service import (
    create_vector_store, upload_files_to_vector_store, ask_with_file_search
)


app = FastAPI(title="OpenAI Tool + Model Router Orchestrator")

@app.post("/run", response_model=RunResponse)
def run_endpoint(req: RunRequest):
    model_used, final_text, tool_logs = run(req.task, req.allow_web_search)
    return RunResponse(
        model_used=model_used,
        final_text=final_text,
        tool_logs=[ToolLog(**t) for t in tool_logs],
    )

@app.post("/run/stream")
def run_stream_endpoint(req: RunRequest):
    def sse():
        yield from run_stream(req.task, req.allow_web_search)

    return StreamingResponse(sse(), media_type="text/event-stream")

@app.post("/vs/create", response_model=CreateVectorStoreResponse)
def vs_create(req: CreateVectorStoreRequest):
    vs = create_vector_store(req.name)
    return CreateVectorStoreResponse(vector_store_id=vs.id, name=getattr(vs, "name", req.name))


@app.post("/vs/upload", response_model=UploadToVectorStoreResponse)
async def vs_upload(vector_store_id: str, files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    temp_paths: List[str] = []
    try:
        for uf in files:
            suffix = os.path.splitext(uf.filename or "")[1]
            fd, path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            with open(path, "wb") as out:
                out.write(await uf.read())
            temp_paths.append(path)

        file_ids = upload_files_to_vector_store(vector_store_id, temp_paths)
        return UploadToVectorStoreResponse(vector_store_id=vector_store_id, file_ids=file_ids)

    finally:
        for p in temp_paths:
            try:
                os.remove(p)
            except Exception:
                pass


@app.post("/ask/filesearch", response_model=AskFileSearchResponse)
def ask_filesearch(req: AskFileSearchRequest):
    model_used, answer = ask_with_file_search(req.vector_store_id, req.question, req.model)
    return AskFileSearchResponse(model_used=model_used, answer=answer)

