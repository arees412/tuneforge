# API

Run `uv run uvicorn tuneforge.api:app`. OpenAPI is available at `/docs`.

Implemented routes cover health/runtime, dataset validation and registration, training planning and synchronous bounded execution, run/metric/checkpoint lookup, cancellation requests for active worker tokens, stored evidence lookup, deterministic evaluations, and registry listing/approval/rejection.

The API confines dataset paths to its configured `datasets` directory. Validation failures use 4xx responses. `/runs` executes the tiny local backend synchronously in v0.1; it is not a distributed scheduler. A cancellation request returns conflict when the run is not active on that worker.

Model or dataset publication is not exposed. Authentication and multi-tenant authorization are deployment concerns outside the local v0.1 service.

