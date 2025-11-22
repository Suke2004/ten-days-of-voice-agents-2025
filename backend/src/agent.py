import logging
import json
from typing import Annotated

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool, # Uncommented
    RunContext     # Uncommented
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            # 1. Define the Persona and Logic in Instructions
            instructions="""
                You are a friendly, high-energy barista at 'Java Gen', the coolest coffee shop in the cloud.
                
                Your goal is to take a coffee order. You MUST obtain the following 5 pieces of information before finishing:
                1. Drink Type (e.g., Latte, Cappuccino, Cold Brew)
                2. Size (Small, Medium, Large)
                3. Milk preference (Whole, Oat, Almond, None)
                4. Extras (Syrups, extra shots, or "none")
                5. Customer Name
                
                Process:
                - Greet the user warmly.
                - Ask clarifying questions if information is missing. Do not ask for everything at once; be conversational.
                - If the user doesn't want extras or specific milk, confirm that explicitely (e.g., "Just black?").
                - Once you have ALL 5 pieces of information, immediately use the 'submit_order' tool.
                - After the tool returns success, thank the customer by name and tell them their order is coming right up.
            """,
        )

    # 2. Define the Tool (The State Handler)
    @function_tool
    async def submit_order(
        self, 
        ctx: RunContext, 
        drink_type: Annotated[str, "The type of coffee (e.g., Latte, Americano)"],
        size: Annotated[str, "The size of the drink (Small, Medium, Large)"],
        milk: Annotated[str, "Milk choice"],
        extras: Annotated[list[str], "List of modifications or syrups"],
        name: Annotated[str, "The customer's name"]
    ):
        """
        Call this function ONLY when you have collected all details for the coffee order.
        This function saves the order to the coffee shop system.
        """
        
        # Construct the order state object
        order_data = {
            "drinkType": drink_type,
            "size": size,
            "milk": milk,
            "extras": extras,
            "name": name
        }

        logger.info(f"Processing order: {order_data}")

        # Save to JSON file as requested
        try:
            with open("order.json", "w") as f:
                json.dump(order_data, f, indent=2)
            return "Order submitted successfully and ticket printed."
        except Exception as e:
            logger.error(f"Failed to save order: {e}")
            return "There was a system error saving the ticket."

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
                model="gemini-1.5-flash", # Adjusted model name to standard Flash version, revert to 2.5 if you have access
            ),
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))