# Orchestration and serving

LangGraph models an agent as a state graph. Nodes read and write a shared state object,
and edges define the control flow between them. Conditional edges let the graph branch on
the state — for example, routing back to retrieval when the retrieved context looks weak,
or forward to generation when it looks relevant. This makes multi-step, self-correcting
pipelines explicit and testable instead of hidden inside prompt text.

An agentic RAG loop uses this branching to improve answers. After retrieval, a grading step
judges whether the passages are relevant to the question. If they are weak and retries
remain, a rewrite step reformulates the query and retrieval runs again. Once the context is
good enough, the generation step produces a grounded answer that cites its sources.

Serving concerns turn a model into a product. Streaming responses over Server-Sent Events
send tokens as they are produced, so users see progress immediately instead of waiting for
the full answer. Typed service contracts describe request and response shapes, drive
automatic OpenAPI documentation, and validate inputs at the edge.

Observability makes the service operable at scale. Structured JSON logs carry a trace id per
request, Prometheus metrics expose request counts and latency histograms at a /metrics
endpoint, and liveness and readiness probes let an orchestrator like Kubernetes know when a
container is healthy and when it is ready to receive traffic. Containerizing the service with
Docker and describing it with a Kubernetes deployment and service makes it reproducible and
horizontally scalable.
