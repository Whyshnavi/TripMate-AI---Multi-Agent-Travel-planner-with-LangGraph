
from pathlib import Path
import traceback
from contextlib import asynccontextmanager

import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend import (
    run_travel_agent,
    initialize_travel_graph,
    close_database_connection,
)


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"


# --------------------------------------------------
# FASTAPI LIFESPAN
# Initialize PostgreSQL and LangGraph at startup.
# Close the database pool during application shutdown.
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("\n" + "=" * 50)
    print("Starting TripMate AI...")
    print("=" * 50)

    try:
        initialize_travel_graph()

        print("PostgreSQL connection initialized.")
        print("LangGraph workflow ready.")
        print("TripMate AI started successfully!")

    except Exception:
        print("Failed to initialize TripMate AI.")
        traceback.print_exc()

        close_database_connection()
        raise

    yield

    print("\nShutting down TripMate AI...")

    try:
        close_database_connection()
        print("PostgreSQL connection pool closed.")

    except Exception:
        print("Error while closing database connection.")
        traceback.print_exc()

    print("TripMate AI stopped.")


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="TripMate AI",
    description="LangGraph Multi-Agent Travel Planner with FastAPI Frontend",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# STATIC FILES
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


# --------------------------------------------------
# HTML TEMPLATES
# --------------------------------------------------

templates = Jinja2Templates(
    directory=str(TEMPLATES_DIR)
)


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


# --------------------------------------------------
# TRAVEL PLANNING API
# --------------------------------------------------

@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest):

    try:
        user_message = request_data.message.strip()

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Message cannot be empty.",
                },
            )

        print("\n" + "=" * 50)
        print("New travel planning request received.")
        print("User message:", user_message)
        print("=" * 50)

        result = run_travel_agent(
            user_input=user_message,
            thread_id=request_data.thread_id,
        )

        return JSONResponse(
            content={
                "success": True,
                "thread_id": result["thread_id"],
                "answer": result["answer"],
                "flight_results": result["flight_results"],
                "hotel_results": result["hotel_results"],
                "itinerary": result["itinerary"],
                "llm_calls": result["llm_calls"],
            },
        )

    except Exception as e:

        print("\nTRAVEL PLANNING ERROR:")
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
            },
        )


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
async def health_check():

    return {
        "status": "ok",
        "message": "TripMate AI API is running",
    }


# --------------------------------------------------
# FAVICON
# --------------------------------------------------

@app.get("/favicon.ico")
async def favicon():

    return JSONResponse(content={})


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
