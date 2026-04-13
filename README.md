# Traveligent — C# / ASP.NET Backend

Production-grade multi-agent travel AI system rewritten in C# with ASP.NET Core 9.

## Architecture improvements over the Node.js version

### Type safety
Every agent output, blackboard fact, SSE event payload, and API request/response is a
typed C# record or class. The compiler catches mismatches that would only appear at
runtime in JavaScript.

### CancellationToken propagation
Cancellation tokens flow from the HTTP request down through Hangfire, through every
agent turn, through every Anthropic API call, and through every database query.
Cancellation is structural — no Redis flag polling between turns.

### System.Threading.Channels for SSE
Replaces the JS `Map<tripId, Set<Response>>` pattern. Bounded channels with
`BoundedChannelFullMode.DropOldest` give backpressure handling and are safe for
concurrent multi-agent writes.

### Hangfire replaces BullMQ
Jobs survive server restarts, run on a pool of 3 workers (matching the original
BullMQ concurrency), and are visible in the Hangfire dashboard at `/hangfire`.
Recurring jobs (price monitoring, budget reset) use Hangfire's recurring job manager.

### IHostedService for background work
`MonitoringScheduler` registers Hangfire recurring jobs on startup with proper
lifecycle management and graceful shutdown.

### EF Core with typed migrations
Replaces raw `pg.query()` string calls. HNSW vector indexes, array columns, and
JSONB columns are configured via Fluent API. Migrations are auto-applied on startup.

### Microsoft.Extensions.Http.Resilience
Replaces the hand-rolled circuit breaker. Each external HTTP client (Amadeus,
Open-Meteo, ExchangeRate, RestCountries) gets a standard resilience pipeline with
retries, circuit breaker, and timeout built in via `AddStandardResilienceHandler()`.

## Project structure

```
src/
├── Program.cs                          # App bootstrap, DI registration
├── appsettings.json
├── Traveligent.csproj
├── Api/
│   └── Controllers/
│       ├── TripsController.cs          # POST/GET trips, SSE stream, approve, cancel, rate
│       ├── AuthController.cs           # Signup, signin, refresh
│       ├── GroupsController.cs         # Travel group CRUD
│       ├── MonitoringController.cs     # Price watches, notifications, timeouts
│       ├── SubAgentsController.cs      # Registry, hierarchy viewer, cost breakdown
│       └── DevController.cs            # Evals, audit trace, preferences, health
├── Agents/
│   ├── AgentRunner.cs                  # Core executor — streaming, timeout, retry
│   ├── PipelineOrchestrator.cs         # 5-phase pipeline — all agents in parallel
│   ├── SubAgentRunner.cs               # Hierarchical sub-agent coordinator
│   ├── HierarchicalAgents.cs           # Itinerary (3 sub) + Logistics (2 sub) parents
│   └── Prompts/
│       └── AgentPrompts.cs             # All system prompts as typed constants
├── Data/
│   ├── TraveligentDbContext.cs         # EF Core context, Fluent API, seed data
│   └── Entities/
│       └── Entities.cs                 # All domain entities as typed C# records/classes
├── Services/
│   ├── SseService.cs                   # System.Threading.Channels SSE manager
│   ├── VectorMemoryService.cs          # pgvector RAG
│   ├── EpisodicMemoryService.cs        # Structured episode extraction + retrieval
│   ├── PreferenceLearningService.cs    # Rating signal extraction + preference synthesis
│   ├── ReactiveBlackboardService.cs    # Redis pub/sub reactive subscriptions
│   ├── CircuitBreakerService.cs        # Circuit breaker state (backed by Polly)
│   ├── ExternalApiService.cs           # Amadeus, Open-Meteo, ExchangeRate, RestCountries
│   ├── CostService.cs                  # Token budget tracking + deduction
│   ├── RateLimiterService.cs           # Per-user Redis sliding window rate limits
│   ├── MultiTurnService.cs             # Conversational clarification before pipeline
│   ├── GroupTravelService.cs           # Group creation, profile merge
│   ├── OrchestratorParseService.cs     # Guardrail + intent parsing
│   ├── CancellationService.cs          # Pipeline cancellation via CancellationToken
│   ├── ConfidenceService.cs            # Agent self-evaluation + escalation
│   └── AuditService.cs                 # Structured event logging
└── Workers/
    └── MonitoringScheduler.cs          # Hangfire job registration + price monitor
```

## Setup

### Prerequisites
- .NET 9 SDK
- PostgreSQL 15+ with pgvector extension
- Redis 7+
- Anthropic API key
- OpenAI API key (embeddings)
- Export or download the required local model artifacts before running the app

### Model artifacts

This project expects local model files to be present before startup. Export or place the required models into the `models/` directory before running the API. If the application depends on ONNX artifacts generated from source checkpoints, run the export scripts first and verify the files exist locally.

Example expected workflow:

```bash
# From the repo root
python scripts/export_hand_detector_onnx.py
python scripts/export_hand_landmarks_onnx.py
python scripts/export_yolo_detector_onnx.py
python scripts/export_clip_vision_to_onnx.py
```

Example expected output files:

```text
models/hand_det.onnx
models/hand_landmarks.onnx
models/yolo_nano.onnx
models/clip_image.onnx
```

Notes:
- Do not assume large ONNX artifacts are checked into Git.
- If an exported model creates a companion `.onnx.data` file, keep that file alongside the `.onnx` file.
- Add model binaries to `.gitignore` or store them with Git LFS / external artifact storage if they exceed standard GitHub size limits.

### Database

```bash
# Install pgvector (Ubuntu)
sudo apt install postgresql-15-pgvector

# Install pgvector (macOS)
brew install pgvector
```

### Run

```bash
cd src

# Set secrets (development)
dotnet user-secrets set "Anthropic:ApiKey" "sk-ant-..."
dotnet user-secrets set "OpenAI:ApiKey" "sk-..."
dotnet user-secrets set "Supabase:ServiceKey" "eyJ..."
dotnet user-secrets set "ConnectionStrings:Postgres" "Host=localhost;Database=traveligent;..."
dotnet user-secrets set "Redis:ConnectionString" "localhost:6379"

# Run (migrations apply automatically on startup)
dotnet run
```

### Publish

```bash
dotnet publish -c Release -o ./publish
# Deploy publish/ to Railway, Fly.io, Azure App Service, etc.
```

## API surface

Identical to the Node.js version — all routes, SSE events, and response shapes are preserved.
See the Node.js README for the full API reference. The frontend `index.html` works unchanged.

## Cost estimate

Same as Node.js: ~$0.17 per pipeline run with Sonnet pricing.

## What's not yet ported

The following are noted for future implementation (see "genuinely missing patterns"):
- Goal-directed planning with backtracking (Semantic Kernel integration point)
- Agent negotiation protocol
- Persistent stateful agents
- Eval harness (DevController stub exists, full runner pending)
- Prompt versioning live read from DB (AgentPrompts currently hardcoded constants)
- Destination memory aggregation background job

## Key C# → Node.js mapping

| Node.js | C# |
|---|---|
| `BullMQ` | `Hangfire` |
| `EventSource` + `res.write()` | `System.Threading.Channels` + `StreamToResponse` |
| `Promise.allSettled()` | `Task.WhenAll()` |
| `AbortController` | `CancellationTokenSource.CreateLinkedTokenSource()` |
| `ioredis` pub/sub | `StackExchange.Redis` pub/sub |
| `pg.Pool` | `Npgsql.EntityFrameworkCore` |
| `nodemon` | `dotnet watch` |
| `process.on('SIGTERM')` | `IHostApplicationLifetime` |
| Redis rate limiter | Same Redis approach via `StackExchange.Redis` |
