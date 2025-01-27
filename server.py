from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from adapter import NovelGPTAdapter
import json
from typing import List, Dict, Any, Optional, Union
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from datetime import datetime

load_dotenv()

app = FastAPI()
security = HTTPBearer()
adapter = NovelGPTAdapter()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    print("\n=== 收到新请求 ===")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"方法: {request.method}")
    print(f"URL: {request.url}")
    print(f"Headers: {dict(request.headers)}")
    
    # 尝试打印请求体
    try:
        body = await request.body()
        if body:
            print(f"Body: {body.decode()}")
    except Exception as e:
        print(f"无法读取请求体: {str(e)}")
    
    print("================")
    
    # 继续处理请求
    response = await call_next(request)
    return response


# CORS 设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据模型
class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.35
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 800
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None

class ErrorResponse(BaseModel):
    error: Dict[str, Any]

# API密钥验证
async def verify_api_key(
    authorization: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    api_key = authorization.credentials
    valid_api_key = os.getenv("NOVEL_GPT_API_KEY")
    
    if not api_key or api_key != valid_api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid API key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key"
                }
            }
        )
    return api_key

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key)
) -> Any:
    try:
        response = adapter.chat_completion(
            messages=[msg.dict() for msg in request.messages],
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stream=request.stream
        )
        
        if not request.stream:
            return response
            
        async def generate_stream():
            try:
                for chunk in response:
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                error_response = {
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": "internal_server_error"
                    }
                }
                yield f"data: {json.dumps(error_response)}\n\n"
                
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": "internal_server_error"
                }
            }
        )

@app.get("/v1/models")
async def list_models(
    api_key: str = Depends(verify_api_key)
) -> Dict:
    """返回支持的模型列表"""
    return {
        "object": "list",
        "data": [
            {
                "id": "nalang-xl",
                "object": "model",
                "created": 1677610602,
                "owned_by": "novel",
                "permission": [],
                "root": "nalang-xl",
                "parent": None
            }
        ]
    }

@app.get("/v1/models/{model}")
async def retrieve_model(
    model: str,
    api_key: str = Depends(verify_api_key)
) -> Dict:
    """返回指定模型的信息"""
    if model != "nalang-xl":
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"The model '{model}' does not exist",
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            }
        )
    
    return {
        "id": model,
        "object": "model",
        "created": 1677610602,
        "owned_by": "novel",
        "permission": [],
        "root": "nalang-xl",
        "parent": None
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# 错误处理
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": "server_error",
                "code": "internal_server_error"
            }
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)