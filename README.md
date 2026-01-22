# Reachy Claude MCP

MCP server that brings [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) to life as your coding companion in [Claude Code](https://claude.ai/claude-code).

Reachy reacts to your coding sessions with emotions, speech, and celebratory dances - making coding more interactive and fun!

## Features

| Feature | Basic | + LLM | + Memory |
|---------|-------|-------|----------|
| Robot emotions & animations | ✅ | ✅ | ✅ |
| Text-to-speech (Piper TTS) | ✅ | ✅ | ✅ |
| Session tracking (SQLite) | ✅ | ✅ | ✅ |
| Smart sentiment analysis | ❌ | ✅ | ✅ |
| AI-generated responses | ❌ | ✅ | ✅ |
| Semantic problem search | ❌ | ❌ | ✅ |
| Cross-project memory | ❌ | ❌ | ✅ |

## Requirements

- Python 3.10+
- [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) robot **or** the simulation (see below)
- Audio output (speakers/headphones)

### Platform Support

| Platform | Basic | LLM (MLX) | LLM (Ollama) | Memory |
|----------|-------|-----------|--------------|--------|
| macOS Apple Silicon | ✅ | ✅ | ✅ | ✅ |
| macOS Intel | ✅ | ❌ | ✅ | ✅ |
| Linux | ✅ | ❌ | ✅ | ✅ |
| Windows | ⚠️ Experimental | ❌ | ✅ | ✅ |

## Quick Start

1. **Install the package:**
   ```bash
   pip install reachy-claude-mcp
   ```

2. **Start Reachy Mini simulation** (if you don't have the physical robot):
   ```bash
   # On macOS with Apple Silicon
   mjpython -m reachy_mini.daemon.app.main --sim --scene minimal

   # On other platforms
   python -m reachy_mini.daemon.app.main --sim --scene minimal
   ```

3. **Add to Claude Code** (`~/.mcp.json`):
   ```json
   {
     "mcpServers": {
       "reachy-claude": {
         "command": "reachy-claude"
       }
     }
   }
   ```

4. **Start Claude Code** and Reachy will react to your coding!

## Installation Options

### Basic (robot + TTS only)

```bash
pip install reachy-claude-mcp
```

Without LLM features, Reachy uses keyword matching for sentiment - still works great!

### With LLM (Smart Responses)

**Option A: MLX (Apple Silicon only - fastest)**
```bash
pip install "reachy-claude-mcp[llm]"
```

**Option B: Ollama (cross-platform)**
```bash
# Install Ollama from https://ollama.ai
ollama pull qwen2.5:1.5b

# Then just use the basic install - Ollama is auto-detected
pip install reachy-claude-mcp
```

The system automatically picks the best available backend: MLX → Ollama → keyword fallback.

### Full Features (requires Qdrant)

```bash
pip install "reachy-claude-mcp[all]"

# Start Qdrant vector database
docker compose up -d
```

### Development Install

```bash
git clone https://github.com/mchardysam/reachy-claude-mcp.git
cd reachy-claude-mcp

# Install with all features
pip install -e ".[all]"

# Or specific features
pip install -e ".[llm]"     # MLX sentiment analysis (Apple Silicon)
pip install -e ".[memory]"  # Qdrant vector store
```

## Running Reachy Mini

### No Robot? Use the Simulation!

You don't need a physical Reachy Mini to use this. The simulation works great:

```bash
# On macOS with Apple Silicon, use mjpython for the MuJoCo GUI
mjpython -m reachy_mini.daemon.app.main --sim --scene minimal

# On Linux/Windows/Intel Mac
python -m reachy_mini.daemon.app.main --sim --scene minimal
```

The simulation dashboard will be available at `http://localhost:8000`.

### Physical Robot

Follow the [Reachy Mini setup guide](https://docs.pollen-robotics.com/reachy-mini/) to connect to your physical robot.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REACHY_CLAUDE_HOME` | `~/.reachy-claude` | Data directory for database, memory, voice models |
| **LLM Settings** | | |
| `REACHY_LLM_MODEL` | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | MLX model (Apple Silicon) |
| `REACHY_OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `REACHY_OLLAMA_MODEL` | `qwen2.5:1.5b` | Ollama model name |
| **Memory Settings** | | |
| `REACHY_QDRANT_HOST` | `localhost` | Qdrant server host |
| `REACHY_QDRANT_PORT` | `6333` | Qdrant server port |
| **Voice Settings** | | |
| `REACHY_VOICE_MODEL` | *(auto-download)* | Path to custom Piper voice model |

## MCP Tools

### Basic Interactions

| Tool | Description |
|------|-------------|
| `robot_respond` | Speak a summary (1-2 sentences) + play emotion |
| `robot_emotion` | Play emotion animation only |
| `robot_celebrate` | Success animation + excited speech |
| `robot_thinking` | Thinking/processing animation |
| `robot_wake_up` | Start-of-session greeting |
| `robot_sleep` | End-of-session goodbye |
| `robot_oops` | Error acknowledgment |
| `robot_acknowledge` | Quick nod without speaking |

### Dance Moves

| Tool | Description |
|------|-------------|
| `robot_dance` | Perform a dance move |
| `robot_dance_respond` | Dance while speaking |
| `robot_big_celebration` | Major milestone celebration |
| `robot_recovered` | After fixing a tricky bug |

### Smart Features

| Tool | Description |
|------|-------------|
| `process_response` | Auto-analyze output and react appropriately |
| `get_project_greeting` | Context-aware greeting based on history |
| `find_similar_problem` | Search past solutions across projects |
| `store_solution` | Save problem-solution pairs for future |
| `link_projects` | Mark relationships between projects |

### Utilities

| Tool | Description |
|------|-------------|
| `list_robot_emotions` | List available emotions |
| `list_robot_dances` | List available dance moves |
| `get_robot_stats` | Memory statistics across sessions |
| `list_projects` | All projects Reachy remembers |

## Available Emotions

```
amazed, angry, anxious, attentive, bored, calm, celebrate, come, confused,
curious, default, disgusted, done, excited, exhausted, frustrated, go_away,
grateful, happy, helpful, inquiring, irritated, laugh, lonely, lost, loving,
neutral, no, oops, proud, relieved, sad, scared, serene, shy, sleep, success,
surprised, thinking, tired, uncertain, understanding, wake_up, welcoming, yes
```

## Available Dances

**Celebrations:** celebrate, victory, playful, party
**Acknowledgments:** nod, agree, listening, acknowledge
**Reactions:** mind_blown, recovered, fixed_it, whoa
**Subtle:** idle, processing, waiting, thinking_dance
**Expressive:** peek, glance, sharp, funky, smooth, spiral

## Usage Examples

Claude can call these tools during coding sessions:

```python
# After completing a task
robot_respond(summary="Done! Fixed the type error.", emotion="happy")

# When celebrating a win
robot_celebrate(message="Tests are passing!")

# Big milestone
robot_big_celebration(message="All tests passing! Ship it!")

# When starting to think
robot_thinking()

# Session start
robot_wake_up(greeting="Good morning! Let's write some code!")

# Session end
robot_sleep(message="Great session! See you tomorrow.")
```

## Architecture

```
src/reachy_claude_mcp/
├── server.py           # MCP server with tools
├── config.py           # Centralized configuration
├── robot_controller.py # Reachy Mini control
├── tts.py              # Piper TTS (cross-platform)
├── memory.py           # Session memory manager
├── database.py         # SQLite project tracking
├── vector_store.py     # Qdrant semantic search
├── llm_backends.py     # LLM backend abstraction (MLX, Ollama)
└── llm_analyzer.py     # Sentiment analysis and summarization
```

## Troubleshooting

### Voice model not found

The voice model auto-downloads on first use. If you have issues:

```bash
# Manual download
mkdir -p ~/.reachy-claude/voices
curl -L -o ~/.reachy-claude/voices/en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -L -o ~/.reachy-claude/voices/en_US-lessac-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

### No audio on Linux

Install PulseAudio or ALSA utilities:

```bash
# Ubuntu/Debian
sudo apt install pulseaudio-utils

# Fedora
sudo dnf install pulseaudio-utils
```

### LLM not working

Check which backend is available:
- **MLX**: Only works on Apple Silicon Macs. Install with `pip install "reachy-claude-mcp[llm]"`
- **Ollama**: Make sure Ollama is running (`ollama serve`) and you've pulled a model (`ollama pull qwen2.5:1.5b`)

If neither is available, the system falls back to keyword-based sentiment detection (still works, just less smart).

### Qdrant connection failed

Make sure Qdrant is running:

```bash
docker compose up -d
```

Or point to a remote Qdrant instance:

```bash
export REACHY_QDRANT_HOST=your-qdrant-server.com
```

### Simulation won't start

If `mjpython` isn't found, you may need to install MuJoCo separately or use regular Python:

```bash
# Try without mjpython
python -m reachy_mini.daemon.app.main --sim --scene minimal
```

On Linux, you may need to set `MUJOCO_GL=egl` or `MUJOCO_GL=osmesa` for headless rendering.

## License

MIT
