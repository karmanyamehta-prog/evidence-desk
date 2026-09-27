---
title: Evidence Desk
emoji: 📚
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
---
# Prepared Space configuration (not deployed)

Copy this YAML header to the repository-root README only when explicitly deploying a Space.
Use `deploy/spaces/Dockerfile` as the root Dockerfile in that deployment branch.

Current Hugging Face rules require a paid plan to create Docker compute Spaces;
CPU Basic's zero hourly charge does not remove that requirement. Recheck
https://huggingface.co/docs/hub/spaces-overview before deployment.

This profile uses **extractive evidence mode**, not an LLM. A 7B model is a poor
CPU-only interactive hosting choice; the report does not claim this fallback has
the same answer quality. Embeddings still run locally on the Space. Data is ephemeral.
Set MAX_UPLOAD_MB=5 and MAX_PAGES=60. This single-user app is not hardened for public
uploads; the prepared demo disables uploads and loads only the fixed corpus.
No paid API key or personal laptop endpoint is used.
