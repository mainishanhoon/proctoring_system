# STAR-format Behavioral Answers (5 examples)

## 1) Fixing MediaPipe / protobuf startup failure
- Situation: Users reported the app crashed with "module mediapipe has no attribute solutions" on startup.
- Task: Identify root cause and make the environment reproducible.
- Action: Reproduced locally, discovered protobuf incompatibility. Pinned `protobuf==3.20.3` and documented the fix in README. Added a clear troubleshooting note in docs.
- Result: Users could start the server reliably; reduced support tickets and added a reproducible startup checklist.

## 2) Reducing false positive deepfake alerts
- Situation: Invigilators saw many false positive deepfake flags during certain webcams.
- Task: Improve detection precision without losing recall for real attacks.
- Action: Logged flags from FFT/noise/blend modules, gathered sample frames, tuned weighted thresholds, and only raised high-severity alerts when multiple sub-checks triggered.
- Result: False positives dropped significantly; invigilator trust increased and fewer manual reviews needed.

## 3) Implementing face registration flow
- Situation: Initial demo lacked robust registration handling for poor selfies.
- Task: Make registration resilient and provide actionable feedback to users.
- Action: Implemented `extract_encoding` checks; returned 400 with friendly messages when no face detected; updated the registration UI to show guidance (lighting, framing).
- Result: Registration success rate improved; fewer support requests during exams.

## 4) Improving liveness reliability
- Situation: Some students failed liveness challenges due to sensitive thresholds.
- Task: Stabilize blink detection and challenge timing.
- Action: Adjusted EAR threshold from noisy telemetry, increased challenge windows, and implemented per-session state to avoid repeated immediate challenges.
- Result: Challenge pass rates rose and fewer legitimate students were flagged.

## 5) Handling WebSocket scaling discussion
- Situation: Prototype used in-memory WebSocket manager, unsuitable for horizontal scaling.
- Task: Propose production-ready architecture for real-time alerts.
- Action: Designed an approach using Redis pub/sub for event routing, a small connection registry service, and sticky session guidance for load balancers; documented migration steps.
- Result: Clear upgrade path for production; readiness plan for distributed deployment.
