## Why This Architecture?

Most traditional RAG systems rely on fixed-size chunking and often introduce additional LLM layers for query classification, retrieval validation, or routing.

In this project, I chose a different approach:

- Semantic Chunking instead of fixed-size chunking
- Retrieval-first architecture
- Agent-based reasoning with LangGraph
- Tool fallback only when necessary

The improved retrieval quality achieved through semantic chunking removes the need for an additional classification LLM before retrieval.

This provides several advantages:

- Lower inference cost
- Reduced latency
- Fewer LLM calls per request
- Simpler system architecture
- Better scalability in production environments
- Reduced hallucinations through retrieval-grounded responses

By focusing on retrieval quality rather than adding extra orchestration layers, the system remains efficient while maintaining strong response accuracy.


## Design Philosophy

Instead of adding multiple LLM layers for query classification and retrieval validation, this system focuses on improving retrieval quality itself through semantic chunking and vector search.

This approach:

- Reduces operational cost
- Minimizes latency
- Simplifies architecture
- Improves scalability
- Keeps responses grounded in retrieved knowledge

Better retrieval often eliminates the need for additional LLM-based routing or classification steps.