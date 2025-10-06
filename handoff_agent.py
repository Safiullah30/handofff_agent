
import os
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled,RunConfig
from dotenv import load_dotenv

# Load env vars
load_dotenv()
set_tracing_disabled(True)

# OpenAI client
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta"
)

# Model
safimodel = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)


confiq = RunConfig(
    model = safimodel,
    model_provider= client,
    tracing_disabled= True
)
# ------------------- Agents -------------------

bio_agent = Agent(
    name="bio agent",
    instructions="Answer only biology-related questions. For anything outside biology, refuse politely.",
    model=safimodel
)

physics_agent = Agent(
    name="physics agent",
    instructions="Answer only physics-related questions. For anything outside physics, refuse politely.",
    model=safimodel
)

chemistry_agent = Agent(
    name="chemistry agent",
    instructions="Answer only chemistry-related questions. For anything outside chemistry, refuse politely.",
    model=safimodel
)

main_agent = Agent(
    name="main agent",
    instructions=(
        "You are the main agent. Your job is to decide if a user question is about Biology, Physics, or Chemistry. "
        "If it's about Biology, handoff to bio agent. "
        "If it's about Physics, handoff to physics agent. "
        "If it's about Chemistry, handoff to chemistry agent. "
        "If it's none of these, politely refuse."
    ),
    handoffs=[bio_agent, physics_agent, chemistry_agent],
    model=safimodel
)

# Run the agent
result = Runner.run_sync(main_agent, input="what is tissue", run_config=confiq)
print(result.final_output)
