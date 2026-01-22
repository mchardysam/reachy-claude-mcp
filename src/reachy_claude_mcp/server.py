"""MCP server for Reachy Mini robot integration with Claude Code."""

import os

from mcp.server.fastmcp import FastMCP

from .robot_controller import RobotController, EMOTION_MAPPING, DANCE_MAPPING
from .tts import PiperTTS
from .memory import MemoryManager
from .action_queue import ActionQueue, get_action_queue

# Initialize the MCP server
mcp = FastMCP("reachy-claude")

# Global instances (lazily initialized)
_controller: RobotController | None = None
_tts: PiperTTS | None = None
_memory: MemoryManager | None = None
_queue_started = False


def get_controller() -> RobotController:
    """Get or create the robot controller."""
    global _controller
    if _controller is None:
        _controller = RobotController(connection_mode="localhost_only")
    return _controller


def get_tts() -> PiperTTS:
    """Get or create the TTS instance."""
    global _tts
    if _tts is None:
        _tts = PiperTTS()
    return _tts


def get_memory() -> MemoryManager:
    """Get or create the memory manager."""
    global _memory
    if _memory is None:
        _memory = MemoryManager()
    return _memory


def get_queue() -> ActionQueue:
    """Get the action queue, initializing it with controller/tts if needed."""
    global _queue_started
    queue = get_action_queue()

    if not _queue_started:
        queue.set_controller(get_controller())
        queue.set_tts(get_tts())
        queue.start()
        _queue_started = True

    return queue


def _queue_status() -> str:
    """Return a brief queue status string."""
    queue = get_queue()
    size = queue.queue_size
    if size == 0:
        return "playing now"
    return f"queued ({size} ahead)"


@mcp.tool()
async def robot_respond(
    summary: str,
    emotion: str = "neutral"
) -> str:
    """Make Reachy Mini speak a summary and show emotion.

    Use this to deliver a concise 1-2 sentence summary of your response
    in a fun, interactive way through the robot.

    Args:
        summary: 1-2 sentence summary for the robot to speak (keep it short and conversational)
        emotion: Emotion to display. Options: happy, excited, proud, grateful, success,
                 celebrate, thinking, curious, confused, uncertain, sad, tired, surprised,
                 amazed, neutral, yes, no, laugh, helpful, welcoming, oops
    """
    queue = get_queue()
    memory = get_memory()

    # Enqueue the action (returns immediately)
    queue.enqueue_emotion_and_speak(emotion, summary)
    status = _queue_status()

    # Record this interaction to SQLite and Qdrant
    memory.record_interaction_full(
        output=summary,
        sentiment=emotion,
        reachy_response=summary,
        project_path=os.getcwd(),
    )

    return f"Robot responding with emotion '{emotion}': {summary} [{status}]"


@mcp.tool()
async def robot_emotion(emotion: str) -> str:
    """Play an emotion animation without speaking.

    Use this to show a quick reaction or expression.

    Args:
        emotion: Emotion name. Common options: happy, excited, proud, thinking,
                 curious, confused, sad, surprised, amazed, neutral, yes, no
    """
    queue = get_queue()
    queue.enqueue_emotion(emotion)
    status = _queue_status()
    return f"Playing emotion: {emotion} [{status}]"


@mcp.tool()
async def robot_celebrate(message: str = "Yes! Nailed it!") -> str:
    """Celebrate a success with excited animation and speech.

    Use this when a task is completed successfully, tests pass,
    or something good happens.

    Args:
        message: Short celebration message (default: "Yes! Nailed it!")
    """
    queue = get_queue()
    queue.enqueue_emotion_and_speak("celebrate", message)
    status = _queue_status()
    return f"Celebrating: {message} [{status}]"


@mcp.tool()
async def robot_thinking() -> str:
    """Show a thinking/processing animation.

    Use this when starting to work on something that will take time.
    """
    queue = get_queue()
    queue.enqueue_emotion("thinking")
    status = _queue_status()
    return f"Showing thinking animation [{status}]"


@mcp.tool()
async def robot_wake_up(greeting: str = "Good morning! Ready to code!") -> str:
    """Morning greeting animation with customizable message.

    Use this at the start of a session to greet the user.

    Args:
        greeting: Custom greeting message (default: "Good morning! Ready to code!")
    """
    queue = get_queue()
    queue.enqueue_emotion_and_speak("welcoming", greeting)
    status = _queue_status()
    return f"Robot greeting: {greeting} [{status}]"


@mcp.tool()
async def robot_sleep(message: str = "See you later! Time to rest.") -> str:
    """End session animation with goodbye message.

    Use this when ending a coding session.

    Args:
        message: Goodbye message (default: "See you later! Time to rest.")
    """
    queue = get_queue()
    # Speech first, then sleep animation
    queue.enqueue_emotion_and_speak("sleep", message)
    status = _queue_status()
    return f"Robot saying goodbye: {message} [{status}]"


@mcp.tool()
async def robot_oops(message: str = "Oops! Let me fix that.") -> str:
    """Show an 'oops' reaction for mistakes or errors.

    Use this when an error occurs or something goes wrong.

    Args:
        message: Error acknowledgment message
    """
    queue = get_queue()
    queue.enqueue_emotion_and_speak("oops", message)
    status = _queue_status()
    return f"Robot acknowledging error: {message} [{status}]"


@mcp.tool()
def list_robot_emotions() -> str:
    """List all available emotion keywords for the robot.

    Returns a list of emotion keywords that can be used with robot_respond
    and robot_emotion tools.
    """
    emotions = sorted(EMOTION_MAPPING.keys())
    return "Available emotions:\n" + "\n".join(f"  - {e}" for e in emotions)


# Dance tools

@mcp.tool()
async def robot_dance(dance: str) -> str:
    """Make Reachy perform a dance move.

    Use this for celebratory moments, acknowledgments, or just for fun!

    Args:
        dance: Dance to perform. Options:
               Celebrations: celebrate, victory, playful, party
               Acknowledgments: nod, agree, listening, acknowledge
               Reactions: mind_blown, recovered, fixed_it, whoa
               Subtle: idle, processing, waiting, thinking_dance
               Expressive: peek, glance, sharp, funky, smooth, spiral
    """
    queue = get_queue()
    queue.enqueue_dance(dance)
    status = _queue_status()
    return f"Performing dance: {dance} [{status}]"


@mcp.tool()
async def robot_dance_respond(
    message: str,
    dance: str = "celebrate"
) -> str:
    """Make Reachy perform a dance while speaking.

    Use this for big celebrations, major accomplishments, or fun moments.

    Args:
        message: What Reachy should say while dancing
        dance: Dance to perform (default: celebrate). Options:
               celebrate, victory, playful, party, mind_blown, funky
    """
    queue = get_queue()
    queue.enqueue_dance_and_speak(dance, message)
    status = _queue_status()
    return f"Reachy dancing '{dance}' and saying: {message} [{status}]"


@mcp.tool()
async def robot_big_celebration(message: str = "We did it! Time to celebrate!") -> str:
    """Trigger a big celebration with dance and speech.

    Use this for major milestones: all tests passing, feature complete,
    bug finally squashed, or any big win worth celebrating!

    Args:
        message: Celebration message (default: "We did it! Time to celebrate!")
    """
    queue = get_queue()
    memory = get_memory()

    # Enqueue victory dance with speech
    queue.enqueue_dance_and_speak("victory", message)
    status = _queue_status()

    # Record as a major success
    memory.record_interaction_full(
        output=message,
        sentiment="success",
        reachy_response=message,
        project_path=os.getcwd(),
    )

    return f"Big celebration queued! Reachy dancing and saying: {message} [{status}]"


@mcp.tool()
async def robot_acknowledge() -> str:
    """Quick acknowledgment nod without speaking.

    Use this to show Reachy is listening or to confirm understanding.
    """
    queue = get_queue()
    queue.enqueue_dance("nod")
    status = _queue_status()
    return f"Reachy nodding in acknowledgment [{status}]"


@mcp.tool()
async def robot_recovered(message: str = "Phew! Fixed it!") -> str:
    """Show a 'recovered from a stumble' animation with speech.

    Use this after fixing a tricky bug or recovering from errors.

    Args:
        message: Recovery message (default: "Phew! Fixed it!")
    """
    queue = get_queue()
    queue.enqueue_dance_and_speak("recovered", message)
    status = _queue_status()
    return f"Reachy recovering and saying: {message} [{status}]"


@mcp.tool()
def list_robot_dances() -> str:
    """List all available dance keywords for the robot.

    Returns a list of dance keywords that can be used with robot_dance
    and robot_dance_respond tools.
    """
    dances = sorted(DANCE_MAPPING.keys())
    return "Available dances:\n" + "\n".join(f"  - {d}" for d in dances)


@mcp.tool()
async def process_response(output: str, project: str | None = None) -> str:
    """Process your response through Reachy - he'll analyze it and react appropriately.

    Use this tool to send your output to Reachy. He will:
    - Analyze the sentiment (success, error, thinking, etc.)
    - Decide if he should speak based on context
    - Choose an appropriate emotion and response
    - Trigger dances for big success streaks!
    - Build memory of your coding sessions

    This is the recommended way to interact with Reachy - just send your
    output and let him decide how to react.

    Args:
        output: Your response text to process
        project: Optional project identifier for context tracking
    """
    memory = get_memory()
    queue = get_queue()

    # Get project from environment if not provided
    if project is None:
        project = os.getcwd()

    # Analyze sentiment
    sentiment = memory.classify_sentiment(output)
    emotion = memory.get_emotion_for_sentiment(sentiment)

    # Decide if we should speak
    if not memory.should_speak(sentiment):
        # Still record the interaction even if not speaking
        memory.record_interaction_full(
            output=output,
            sentiment=sentiment,
            reachy_response=None,
            project_path=project,
        )
        return f"Reachy noticed: {sentiment} (staying quiet this time)"

    # Generate a contextual summary
    summary = memory.generate_summary(output, sentiment)

    # Check for streak-based reactions
    stats = memory.get_stats()
    success_streak = stats["current_success_streak"]
    error_streak = stats["current_error_streak"]
    use_dance = False
    dance_name = None

    # Escalate to dance for success streaks
    if sentiment == "success":
        if success_streak >= 5:
            # Big celebration for 5+ in a row!
            use_dance = True
            dance_name = "victory"
            summary = "Five in a row! I'm on fire!"
        elif success_streak >= 3:
            # Medium celebration
            use_dance = True
            dance_name = "celebrate"
            summary = "We're on a roll!"

    # Special reaction after recovering from errors
    if sentiment == "success" and error_streak == 0 and stats.get("last_was_error", False):
        use_dance = True
        dance_name = "recovered"
        summary = "Phew! Back on track!"

    # Enqueue the reaction (returns immediately)
    if use_dance and dance_name:
        queue.enqueue_dance_and_speak(dance_name, summary)
        reaction_type = f"dance:{dance_name}"
    else:
        queue.enqueue_emotion_and_speak(emotion, summary)
        reaction_type = f"emotion:{emotion}"

    status = _queue_status()

    # Record the interaction to SQLite and Qdrant
    memory.record_interaction_full(
        output=output,
        sentiment=sentiment,
        reachy_response=summary,
        project_path=project,
    )

    stats = memory.get_stats()
    return (
        f"Reachy reacting with '{reaction_type}' and saying: \"{summary}\" [{status}]\n"
        f"Session stats: {stats['session_interactions']} interactions, "
        f"success streak: {stats['current_success_streak']}, "
        f"error streak: {stats['current_error_streak']}"
    )


@mcp.tool()
def get_robot_stats() -> str:
    """Get Reachy's memory statistics across all sessions.

    Returns stats about total interactions, success rate, projects worked on, etc.
    """
    memory = get_memory()
    stats = memory.get_stats()

    return (
        f"Reachy's Memory Stats:\n"
        f"  Total interactions: {stats['total_interactions']}\n"
        f"  Total successes: {stats['total_successes']}\n"
        f"  Total errors: {stats['total_errors']}\n"
        f"  Success rate: {stats['success_rate']:.1%}\n"
        f"  Projects seen: {stats['projects_seen']}\n"
        f"\nCurrent Session:\n"
        f"  Interactions: {stats['session_interactions']}\n"
        f"  Success streak: {stats['current_success_streak']}\n"
        f"  Error streak: {stats['current_error_streak']}"
    )


# Project-aware tools

@mcp.tool()
def get_project_greeting(project_path: str | None = None) -> str:
    """Get a personalized greeting for a project based on history.

    Call this when starting work on a project to get a context-aware greeting.
    Reachy remembers past sessions, success rates, and when you last worked on it.

    Args:
        project_path: Path to the project (defaults to current working directory)
    """
    memory = get_memory()

    if project_path is None:
        project_path = os.getcwd()

    greeting = memory.get_project_greeting(project_path)
    memory.set_project(project_path)

    return greeting


@mcp.tool()
def find_similar_problem(problem: str, search_all_projects: bool = True) -> str:
    """Search for similar problems you've encountered before.

    Reachy remembers problems and solutions from all your projects.
    Use this when you hit an error to see if you've solved something similar.

    Args:
        problem: Description of the problem or error message
        search_all_projects: If True, search all projects. If False, only current project.
    """
    memory = get_memory()

    # Search for similar problems
    results = memory.find_similar_problems(
        query=problem,
        current_project_only=not search_all_projects,
        limit=5,
    )

    if not results:
        # Try searching across projects for related content
        results = memory.find_related_across_projects(problem, limit=3)

    if not results:
        return "No similar problems found in memory. This might be a new challenge!"

    # Format results
    output = "Found similar issues from the past:\n\n"
    for i, mem in enumerate(results, 1):
        score_pct = f"{(mem.score or 0) * 100:.0f}%" if mem.score else "N/A"
        output += f"{i}. [Similarity: {score_pct}]\n"
        output += f"   {mem.content[:200]}...\n"
        if mem.metadata and mem.metadata.get("solution"):
            output += f"   Solution: {mem.metadata['solution'][:100]}...\n"
        output += "\n"

    return output


@mcp.tool()
def store_solution(problem: str, solution: str, tags: str | None = None) -> str:
    """Store a problem-solution pair for future reference.

    Call this after solving a tricky problem so Reachy can remember it.
    Next time you hit a similar issue, he'll remind you of the solution!

    Args:
        problem: Description of the problem
        solution: How you solved it
        tags: Optional comma-separated tags (e.g., "python,auth,database")
    """
    memory = get_memory()

    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    memory.store_problem_solution(
        problem=problem,
        solution=solution,
        tags=tag_list,
    )

    return f"Stored solution for future reference! Tags: {tag_list or 'none'}"


@mcp.tool()
def link_projects(
    other_project_path: str,
    link_type: str = "related",
    description: str | None = None,
) -> str:
    """Link the current project to another project.

    Use this to mark relationships between projects (dependencies, forks, related work).
    Reachy uses these links to provide better cross-project suggestions.

    Args:
        other_project_path: Path to the other project
        link_type: Type of relationship: "dependency", "related", "fork", or "shared_code"
        description: Optional description of the relationship
    """
    memory = get_memory()
    current_path = os.getcwd()

    memory.link_projects(
        project_a_path=current_path,
        project_b_path=other_project_path,
        link_type=link_type,
        description=description,
    )

    return f"Linked {current_path} -> {other_project_path} ({link_type})"


@mcp.tool()
def list_projects() -> str:
    """List all projects Reachy remembers.

    Shows recent projects with their stats and relationships.
    """
    memory = get_memory()

    projects = memory.get_recent_projects(limit=20)

    if not projects:
        return "No projects in memory yet. Start working and I'll remember!"

    output = "Projects I remember:\n\n"
    for p in projects:
        output += f"- {p.name}\n"
        output += f"  Path: {p.path}\n"
        output += f"  Sessions: {p.total_sessions}, "
        output += f"Success: {p.total_successes}, Errors: {p.total_errors}\n"

        # Get linked projects
        linked = memory.get_linked_projects(p.path)
        if linked:
            links = [f"{lp.name} ({lt})" for lp, lt, _ in linked]
            output += f"  Linked to: {', '.join(links)}\n"
        output += "\n"

    return output


@mcp.tool()
def end_session(summary: str | None = None) -> str:
    """End the current coding session.

    Call this when wrapping up work on a project. Reachy will remember
    what happened for next time.

    Args:
        summary: Optional summary of what was accomplished
    """
    memory = get_memory()
    memory.end_session(summary=summary)

    return f"Session ended. {'Summary saved: ' + summary if summary else 'See you next time!'}"


def main():
    """Run the MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
