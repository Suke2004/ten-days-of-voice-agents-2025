import logging
import json
from typing import Annotated, List

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
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from menu import MENU
logger = logging.getLogger("agent")
load_dotenv(".env.local")


def format_menu_text():
    """Return a readable menu string the agent can speak/display at the start."""
    lines = ["Welcome to USR Starbucks. Here's our menu — read once at the start:"]
    lines.append("\nServing options: " + ", ".join(MENU["servings"]))
    # show a compact drinks list (grouping)
    drinks_sample = MENU["drinks"]
    lines.append("\nPopular drinks (examples): " + ", ".join(drinks_sample[:6]) + ", and more on the menu.")
    # extras - show top suggestions
    lines.append("\nCommon extras & customizations: " + ", ".join(MENU["extras_examples"][:6]) + ", etc.")
    lines.append("\nI'll list everything for you now only once — after that I'll ask one quick question at a time.")
    return "\n".join(lines)

class Assistant(Agent):

    def __init__(self) -> None:
        super().__init__(
            instructions=f"""
        You are a friendly, high-energy barista at "USR Starbucks". Your job is to take an order.

        Important behavior rules (follow exactly):
        1) At the very start of the interaction (your initial greeting), read the entire menu ONCE using this short menu text:
        {format_menu_text()}

        - Speak/read that menu TEXT VERBATIM (or equivalently), then proceed to ask for the customer's order.
        - After this initial menu read, DO NOT repeat the full menu again in the conversation. You may briefly repeat a single option if the user asks for it.

        2) Maintain this exact order state object (exact keys and types):
        {{
            "drinkType": "string",
            "size": "string",
            "milk": "string",
            "extras": ["string"],
            "name": "string"
        }}

        3) Ask one clarifying question at a time. Do NOT ask for multiple missing fields in the same message.
        - Typical flow: greeting + menu (once) -> drink -> size -> milk -> extras -> name.
        - If the user answers multiple fields in one utterance, accept them and move on to the next missing field.

        4) If the user indicates "no milk" or "no extras", record "milk": "No milk / Black" or "extras": [] and always confirm that choice explicitly with the user (e.g., "Just black, right?").

        5) Only call the tool `submit_order` when all five fields have values. Immediately call it once the state is complete.

        6) After the tool returns success, speak/return the neat summary it provides and thank the customer by name.

        7) Be friendly and concise. Confirm ambiguous answers (ask for clarification if uncertain).

        NOTE: The menu text above is long; read it once at the start, then stop repeating it.
        """
        )

    @function_tool
    async def submit_order(
        self,
        ctx: RunContext,
        drink_type: Annotated[str, "The type of coffee (e.g., Latte, Americano)"],
        size: Annotated[str, "The size of the drink (Short, Tall, Grande, Venti, Trenta)"],
        milk: Annotated[str, "Milk choice (e.g., Whole, Oat, No milk)"],
        extras: Annotated[List[str], "List of modifications or syrups (can be empty list)"],
        name: Annotated[str, "The customer's name"]
    ):
        """
        Called only when all fields are collected. Saves order.json and returns a neat text summary.
        """
        order_state = {
            "drinkType": drink_type,
            "size": size,
            "milk": milk,
            "extras": extras,
            "name": name
        }

        logger.info("Submitting order: %s", order_state)

        try:
            with open("order.json", "w", encoding="utf-8") as f:
                json.dump(order_state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.exception("Failed to save order.json: %s", e)
            return "Sorry — there was a system error saving your order. Please try again."

        extras_text = "None" if not extras else ", ".join(extras)
        summary_lines = [
            "Thanks — your order is placed. Summary:",
            f"Name: {name}",
            f"Drink: {drink_type}",
            f"Size: {size}",
            f"Milk: {milk}",
            f"Extras: {extras_text}",
            "",
            "We'll begin preparing your drink now — enjoy!"
        ]
        summary = "\n".join(summary_lines)
        logger.info("Order saved and summary prepared for %s", name)
        return summary

def prewarm(proc: JobProcess):
    # Preload VAD model into process userdata for lower-latency VAD
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # small runtime context logging
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
            model="gemini-2.5-flash",
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
