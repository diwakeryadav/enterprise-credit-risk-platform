import os
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse

from src.api.routes import explainability, monitoring
from src.logger import logger

app = FastAPI(
    title="Enterprise Credit Risk Platform Services",
    description="Production services for credit scoring, compliance explainability, and observability.",
    version="1.0.0",
)

# Include Routers
app.include_router(
    explainability.router, prefix="/api/v1/explainability", tags=["Explainability"]
)
app.include_router(
    monitoring.router, prefix="/api/v1/monitoring", tags=["Monitoring & Drift"]
)


@app.get("/", include_in_schema=False)
def read_root():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>UI Dashboard Template Not Found</h1>", status_code=404)



@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "service": "credit-risk-platform"}


if __name__ == "__main__":
    logger.info("Starting FastAPI Server...")
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8001, reload=True)

