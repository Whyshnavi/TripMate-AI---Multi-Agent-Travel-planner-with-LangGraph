
import os
import uuid
import operator
import traceback

from typing import TypedDict, Annotated
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from dotenv import load_dotenv

# ============================================================
# 1. ENVIRONMENT CONFIGURATION
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Please check your .env file."
    )


# ============================================================
# 2. IMPORT DEPENDENCIES
# ============================================================

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights


# ============================================================
# 3. DATABASE URL CONFIGURATION
# ============================================================

def get_database_url():

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Add your Render External Database URL to .env."
        )

    parts = urlsplit(database_url)

    if parts.scheme not in ("postgresql", "postgres"):
        raise ValueError(
            "Invalid DATABASE_URL. Expected PostgreSQL URL."
        )

    if not parts.hostname:
        raise ValueError(
            "Invalid DATABASE_URL: hostname is missing."
        )

    if not parts.path or parts.path == "/":
        raise ValueError(
            "Invalid DATABASE_URL: database name is missing."
        )

    # Preserve existing query parameters.
    query_params = dict(parse_qsl(parts.query))

    # Render PostgreSQL requires SSL for external connections.
    query_params.setdefault("sslmode", "require")

    return urlunsplit((
        parts.scheme,
        parts.netloc,
        parts.path,
        urlencode(query_params),
        parts.fragment
    ))


# ============================================================
# 4. GROQ LLM CONFIGURATION
# ============================================================

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.3
)


# ============================================================
# 5. TRAVEL STATE
# ============================================================

class TravelState(TypedDict):

    messages: Annotated[list[AnyMessage], operator.add]

    user_query: str

    flight_results: str

    hotel_results: str

    itinerary: str

    llm_calls: int


# ============================================================
# 6. FLIGHT AGENT
# ============================================================

def flight_agent(state: TravelState):

    print("\nFlight Agent: Searching for flights...")

    query = state["user_query"]

    flight_data = search_flights(query)

    return {
        "flight_results": flight_data,

        "messages": [
            AIMessage(
                content="Flight results fetched successfully."
            )
        ],

        "llm_calls": state.get("llm_calls", 0)
    }


# ============================================================
# 7. HOTEL AGENT
# ============================================================

def hotel_agent(state: TravelState):

    print("\nHotel Agent: Searching for hotels...")

    query = f"Best hotels for {state['user_query']}"

    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,

        "messages": [
            AIMessage(
                content="Hotel information fetched successfully."
            )
        ],

        "llm_calls": state.get("llm_calls", 0)
    }


# ============================================================
# 8. ITINERARY AGENT
# ============================================================

def itinerary_agent(state: TravelState):

    print("\nItinerary Agent: Creating travel itinerary...")

    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Flight Results:
{state['flight_results']}

Hotel Results:
{state['hotel_results']}

Instructions:
- Create a practical day-by-day itinerary.
- Consider the user's travel budget.
- Include major attractions.
- Suggest suitable activities.
- Organize activities by day.
- Make the itinerary easy to follow.
- Do not invent flight prices or availability.
"""

    response = llm.invoke([
        SystemMessage(
            content="You are an expert AI travel planner."
        ),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,

        "messages": [response],

        "llm_calls": state.get("llm_calls", 0) + 1
    }


# ============================================================
# 9. FINAL RESPONSE AGENT
# ============================================================

def final_agent(state: TravelState):

    print("\nFinal Agent: Preparing final travel response...")

    final_prompt = f"""
Generate the final travel response for the user.

User Request:
{state['user_query']}

Flights:
{state['flight_results']}

Hotels:
{state['hotel_results']}

Itinerary:
{state['itinerary']}

Format the final answer using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Day-by-Day Itinerary
5. Estimated Budget
6. Final Recommendations

Important:
- Be clear and practical.
- Do not invent flight prices.
- Mention when live flight prices are unavailable.
- Include realistic travel recommendations.
- Keep the response useful for real travel planning.
"""

    response = llm.invoke([
        SystemMessage(
            content="You are a professional AI travel planning assistant."
        ),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],

        "llm_calls": state.get("llm_calls", 0) + 1
    }


# ============================================================
# 10. BUILD LANGGRAPH WORKFLOW
# ============================================================

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# ============================================================
# 11. POSTGRESQL CONNECTION POOL
# ============================================================

_pool = None
_checkpointer = None
travel_graph = None


def initialize_travel_graph():

    global _pool
    global _checkpointer
    global travel_graph

    if travel_graph is not None:
        return travel_graph

    print("\nInitializing TripMate PostgreSQL connection pool...")

    database_url = get_database_url()

    try:

        # Same configuration verified by test_pool.py.
        _pool = ConnectionPool(
            conninfo=database_url,

            min_size=1,
            max_size=1,
            num_workers=1,

            kwargs={
                "autocommit": True,
                "row_factory": dict_row,
                "connect_timeout": 15,
            },

            open=False,
        )

        # Open the pool.
        print("Opening PostgreSQL connection pool...")

        _pool.open(
            wait=True,
            timeout=25,
        )

        print("PostgreSQL connection pool established!")

        # Verify database connectivity.
        print("Verifying PostgreSQL connection...")

        with _pool.connection() as conn:

            with conn.cursor() as cur:

                cur.execute("SELECT 1 AS connection_test")

                result = cur.fetchone()

                print(
                    "Database connection verified:",
                    result["connection_test"]
                )

        # Initialize LangGraph checkpointer.
        print("Initializing PostgreSQL checkpointer...")

        _checkpointer = PostgresSaver(_pool)

        _checkpointer.setup()

        print("PostgreSQL checkpointer initialized!")

        # Compile LangGraph workflow.
        print("Compiling TripMate LangGraph workflow...")

        travel_graph = graph.compile(
            checkpointer=_checkpointer
        )

        print("TripMate LangGraph workflow compiled successfully!")

        return travel_graph

    except Exception:

        print("\nPostgreSQL / LangGraph initialization failed!")

        traceback.print_exc()

        # Clean up partially initialized resources.
        if _pool is not None:

            try:
                _pool.close()

            except Exception:
                pass

        _pool = None
        _checkpointer = None
        travel_graph = None

        raise


# ============================================================
# 12. CLOSE DATABASE CONNECTION
# ============================================================

def close_database_connection():

    global _pool
    global _checkpointer
    global travel_graph

    if _pool is not None:

        print("\nClosing PostgreSQL connection pool...")

        try:

            _pool.close()

            print("PostgreSQL connection pool closed!")

        except Exception as e:

            print(
                "Error closing PostgreSQL pool:",
                str(e)
            )

        finally:

            _pool = None
            _checkpointer = None
            travel_graph = None


# ============================================================
# 13. RUN TRAVEL AGENT
# ============================================================

def run_travel_agent(
    user_input: str,
    thread_id: str | None = None
):

    global travel_graph

    if not user_input or not user_input.strip():

        raise ValueError(
            "Please enter a valid travel request."
        )

    # Initialize graph if it has not been initialized.
    if travel_graph is None:

        initialize_travel_graph()

    # Generate a unique conversation ID.
    if not thread_id:

        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    print("\n" + "=" * 55)
    print("TripMate: Starting travel planning workflow...")
    print("=" * 55)

    try:

        result = travel_graph.invoke(

            {
                "messages": [
                    HumanMessage(content=user_input)
                ],

                "user_query": user_input,

                "flight_results": "",

                "hotel_results": "",

                "itinerary": "",

                "llm_calls": 0
            },

            config=config
        )

        # Get final response.
        final_answer = result["messages"][-1].content

        print("\nTripMate: Travel planning completed successfully!")

        return {

            "thread_id": thread_id,

            "answer": final_answer,

            "flight_results": result.get(
                "flight_results", ""
            ),

            "hotel_results": result.get(
                "hotel_results", ""
            ),

            "itinerary": result.get(
                "itinerary", ""
            ),

            "llm_calls": result.get(
                "llm_calls", 0
            )
        }

    except Exception:

        print("\nTripMate workflow failed!")

        traceback.print_exc()

        raise
