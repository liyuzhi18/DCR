# DCR Agent Collaboration Rules

- Primary goal: for prescribed `u(x)` and `T(x)`, return a fully audited steady DCR solution.
- Do not change physical equations without explicit user approval.
- Never overwrite an accepted checkpoint from a diagnostic run.
- Change one numerical hypothesis at a time.
- Every experiment must record its baseline, changed variable, prediction, result, and stopping condition.
- Never infer physical branch behavior from unaudited transient states.
- Failed prototypes remain isolated and default OFF.
- Production checkpoints require the complete transient-free audit.
- After each experiment, report verified facts and stop for ChatGPT analysis rather than autonomously starting a new solver architecture.
