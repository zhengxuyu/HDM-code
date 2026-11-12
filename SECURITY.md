# Security

This preview is intended for local research, not unauthenticated public hosting.
The server listens on `127.0.0.1`; do not expose it directly to the internet.

Before remote deployment, provide authentication, TLS, body-size limits, rate limits,
timeouts, and a bounded queue at a reverse proxy. Inference is serialized, but waiting
requests can exhaust resources. A request can trigger a large model download/load.
Model probabilities are not an authorization boundary or proof of action safety.

Image requests accept base64 PNG/JPEG only (4 MB, 1,048,576 pixels); the server never
fetches client image URLs or file paths. These limits do not replace reverse-proxy
body/queue limits. Game traces include screenshots and diagnostics: review before
sharing. Film playback renders labels as text and only accepts inline PNG/JPEG frames.

Requests and provenance may contain sensitive state and prompt text. Review logging
and retention. Use trusted model sources and pin revisions. Model and dependency
licenses must be reviewed separately.

Do not post secrets or exploit details in public issues. Once published, use the
repository's private vulnerability reporting channel if enabled; otherwise request
a private maintainer contact without disclosing the vulnerability. A public reporting
address has not yet been configured.
