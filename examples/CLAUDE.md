# Reachy Integration

> **How to use:** Copy this file to your project root or `~/projects/CLAUDE.md` for global use.
> Claude Code will read these instructions and use Reachy throughout your coding sessions.

Keep Reachy engaged throughout your work! Use these tools to share progress, reactions, and updates:

## Quick Status Updates (use frequently!)

Use `robot_respond(summary, emotion)` for quick updates as you work:

```python
# Starting a task
robot_respond("Alright, let me dig into this codebase", "thinking")

# Found something interesting
robot_respond("Oh interesting, there's already a helper for this", "curious")

# Making progress
robot_respond("Found the bug, it's a missing null check", "attentive")

# Something unexpected
robot_respond("Hmm, this dependency is way out of date", "confused")

# Quick wins
robot_respond("Fixed that typo", "happy")

# About to do something significant
robot_respond("Running the test suite now", "thinking")
```

## When to Update Reachy

**Update frequently during work:**
- Starting to investigate something
- Finding something unexpected or interesting
- Making progress on a problem
- Before running tests/builds
- After completing sub-tasks
- When hitting minor issues

**Use `process_response()` for significant moments** (lets Reachy analyze and react):
- Major errors or failures
- Tests passing/failing
- Task completion
- Complex situations where sentiment isn't obvious

## Emotion Quick Reference

- **thinking** - investigating, reading code, working on something
- **curious** - found something interesting
- **confused** - something unexpected, needs investigation
- **happy** - small wins, quick fixes
- **excited/proud** - bigger accomplishments
- **oops** - made a mistake, need to fix something

## Dance Moves (for celebrations!)

Use dances for bigger moments and celebrations:

```python
# Big wins - all tests pass, feature complete
robot_big_celebration("All tests passing! Ship it!")

# Dance with custom message
robot_dance_respond("That was a tricky one!", "victory")

# Quick acknowledgment nod
robot_acknowledge()

# After fixing a tough bug
robot_recovered("Finally squashed that bug!")

# Just dance (no speech)
robot_dance("celebrate")
```

**Dance Keywords:**
- **Celebrations:** celebrate, victory, playful, party
- **Acknowledgments:** nod, agree, listening, acknowledge
- **Reactions:** mind_blown, recovered, fixed_it, whoa
- **Subtle:** idle, processing, waiting, thinking_dance

**Auto-celebration:** `process_response()` will automatically trigger dances for success streaks (3+ or 5+ in a row)!

## Project Memory

Reachy remembers across projects. Use these tools when relevant:

- `get_project_greeting()` - Call at session start for context-aware greeting
- `find_similar_problem("error description")` - Search past solutions across all projects
- `store_solution(problem, solution, tags)` - Save solutions for future reference
- `link_projects(other_path, "related")` - Mark project relationships
