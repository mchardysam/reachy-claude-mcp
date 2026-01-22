# Reachy Claude MCP

This is the MCP server that integrates Reachy Mini robot with Claude Code.

## Development

```bash
# Install with all features (LLM + memory)
pip install -e ".[all]"

# Or specific features
pip install -e ".[llm]"     # MLX sentiment analysis
pip install -e ".[memory]"  # Qdrant vector store

# Run the MCP server
reachy-claude

# Or run directly
python -m reachy_claude_mcp.server
```

## Architecture

- `server.py` - MCP server with tools
- `robot_controller.py` - Reachy Mini control (emotions, animations)
- `tts.py` - Piper TTS for speech synthesis
- `memory.py` - Persistent memory across sessions (SQLite + Qdrant)
- `llm_analyzer.py` - MLX-based sentiment analysis and summary generation
- `database.py` - SQLite schema for projects/sessions
- `vector_store.py` - Qdrant integration for semantic search

## Dependencies

- **Qdrant** - Vector store on port 6333 (run via Docker)
- **Piper TTS** - Text-to-speech synthesis
- **MLX** - Local LLM for sentiment analysis (Qwen2.5)
