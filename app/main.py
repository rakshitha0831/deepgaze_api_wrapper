from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn
import os


from .sal_service import SaliencyService
from . import sal_router

CB_PATH = 'D:/dragonfly/codebase/deepgaze_api/models/centerbias_mit1003.npy'
@asynccontextmanager
async def lifespan(app: FastAPI):
    sal_service = SaliencyService(cb_path=CB_PATH)
    app.state.sal_service = sal_service
    yield
    app.state.sal_service = None


app = FastAPI(lifespan=lifespan)

app.include_router(sal_router.router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

