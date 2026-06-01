# Recall as diffusion

The payoff direction for the substrate: instead of compressing a need into a query string and
getting a ranked list, your current activation *is* the query, continuously. Focus on one node
and adjacent nodes light at distance-decayed weight; moving to a neighbouring need is a step,
not a restart.

Diffusion *reads* the shared durable graph and *writes* transient, per-mind activation — which
is exactly the layer that should be partitioned per mind.
