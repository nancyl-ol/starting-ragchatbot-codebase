# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
```bash
# Install dependencies using uv
uv sync

# Set up environment variables (required)
echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
```

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Code Quality Tools
```bash
# Format code (black + isort)
./format.sh

# Run linting checks (flake8 + mypy + format verification)
./lint.sh

# Run all quality checks and tests
./quality-check.sh

# Manual commands
uv run black backend/ main.py          # Format with black
uv run isort backend/ main.py          # Sort imports
uv run flake8 backend/ main.py         # Lint with flake8
uv run mypy backend/ main.py           # Type checking
```

### Development Access Points
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- API Base: http://localhost:8000/api

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) chatbot system** for querying course materials. The architecture follows a tool-based RAG approach where Claude dynamically searches course content and generates contextually aware responses.

### Core System Flow
1. **Frontend** (vanilla HTML/JS) sends queries to FastAPI backend
2. **RAGSystem** orchestrates the query processing pipeline
3. **SessionManager** maintains conversation context across requests
4. **AIGenerator** calls Claude API with structured tool definitions
5. **CourseSearchTool** performs semantic search on vector-stored course content
6. **VectorStore** manages ChromaDB collections for course metadata and content chunks

### Key Components

**Backend Components** (`backend/`):
- `app.py`: FastAPI application with `/api/query` and `/api/courses` endpoints
- `rag_system.py`: Main orchestrator - coordinates all components for query processing
- `ai_generator.py`: Claude API wrapper with tool-calling capability
- `search_tools.py`: Defines CourseSearchTool for semantic search operations
- `vector_store.py`: ChromaDB interface with dual collections (catalog + content)
- `document_processor.py`: Parses course documents into structured chunks
- `session_manager.py`: Handles conversation history and context
- `models.py`: Pydantic models (Course, Lesson, CourseChunk)
- `config.py`: Centralized configuration using environment variables

**Frontend** (`frontend/`):
- `index.html`: Single-page chat interface
- `script.js`: Handles user input, API calls, and response rendering
- `style.css`: UI styling

### Data Models and Processing

**Course Document Format**: Documents follow a specific structure:
```
Course Title: [title]
Course Link: [url]  
Course Instructor: [instructor]

Lesson 1: [lesson title]
Lesson Link: [lesson url]
[lesson content...]

Lesson 2: [lesson title]
[lesson content...]
```

**Vector Storage Strategy**:
- **course_catalog collection**: Stores course metadata for semantic course name resolution
- **course_content collection**: Stores text chunks with course/lesson context for content search
- Uses sentence-transformers (`all-MiniLM-L6-v2`) for embeddings
- Chunks are 800 characters with 100-character overlap

**Tool-Based Search**: The system uses Anthropic's tool calling where Claude can:
- Search course content with optional course name and lesson number filters
- Receive formatted results with source attribution
- Generate responses based on retrieved context

### Configuration and Environment

**Required Environment Variables**:
- `ANTHROPIC_API_KEY`: Claude API access (required)

**Key Configuration** (`config.py`):
- Model: `claude-sonnet-4-20250514`
- Chunk size: 800 characters, 100 overlap
- Max search results: 5
- Conversation history: 2 exchanges
- ChromaDB path: `./chroma_db`

### Session and Context Management

The system maintains conversation state through:
- Session IDs generated for each conversation
- Conversation history limited to recent exchanges
- Context passed to Claude for coherent multi-turn conversations
- Source tracking for response attribution

### Development Notes

**Document Loading**: Course documents are automatically loaded from `docs/` folder on application startup. The system checks for existing courses to avoid duplicates.

**Error Handling**: The system gracefully handles missing courses, failed searches, and API errors with user-friendly messages.

**Frontend-Backend Communication**: Uses JSON API with structured request/response models for type safety and clear interfaces.

**Code Quality Standards**:
- **Black**: Automatic code formatting (88 character line length)
- **isort**: Import organization (black-compatible profile)
- **flake8**: Code linting and style checking
- **mypy**: Static type checking
- All configuration in `pyproject.toml` and `.flake8`
- Run `./format.sh` before committing to ensure consistent formatting
- Run `./lint.sh` to verify code quality standards

**Development Workflow**:
- always use uv to run the server do not use pip directly
- make sure to use uv to manage all dependencies
- use uv to run Python files
- don't use ./run.sh to start the server, I will do it myself