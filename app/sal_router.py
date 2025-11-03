import io
from fastapi import APIRouter, HTTPException, UploadFile, File, Request
from fastapi.responses import StreamingResponse


router = APIRouter()
@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):

    service =request.app.state.sal_service

    if not service:
        raise HTTPException(status_code=503, detail="Service not found")
    img_bytes = await file.read()

    try:
        img = service.predict(img_bytes)
        return StreamingResponse(io.BytesIO(img), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
def read_root():
    return {"Hello": "World"}