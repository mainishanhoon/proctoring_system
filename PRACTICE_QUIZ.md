# Practice Quiz — Proctoring System

1) What endpoint registers a student's face?
- A: `POST /api/students/{student_id}/register-face`

2) Which library provides face mesh landmarks used for gaze and EAR?
- A: MediaPipe

3) What does EAR stand for and what is it used for?
- A: Eye Aspect Ratio — used to detect blinks for liveness checks.

4) True/False: Face encodings are stored as raw image files in the DB.
- A: False — embeddings are stored as JSON text (HOG descriptor).

5) Multiple choice: Which of the following is NOT used by the deepfake detector?
- A) FFT frequency analysis
- B) Noise / Laplacian statistics
- C) Face landmark-based emotion classification
- D) DCT block analysis
- Correct answer: C

6) What severity leads to session.status being set to `flagged`?
- A: `high` severity violations

7) Short answer: Name two ways to scale WebSocket alerting across multiple app instances.
- A: Use Redis pub/sub with a distributed connection manager; use sticky sessions with a load balancer or a dedicated WS broker (socket.io cluster/managed service).

8) True/False: The code initializes MediaPipe models per-frame to ensure fresh state.
- A: False — models are initialised at import to avoid per-frame reinitialisation.

9) What is a recommended fix when MediaPipe raises an attribute error about `solutions`?
- A: Pin or reinstall compatible `protobuf` (for example `pip install protobuf==3.20.3 --force-reinstall`) and ensure `mediapipe==0.10.14` if needed.

10) Short answer: Where are live violation alerts sent for human review?
- A: Invigilator UI via WebSocket `/ws/invigilator/{session_id}`

--- Answers (compact)
1: POST /api/students/{id}/register-face
2: MediaPipe
3: Eye Aspect Ratio — blink detection
4: False
5: C
6: high
7: Redis pub/sub; sticky sessions or WS broker
8: False
9: Pin protobuf (3.20.3)
10: Invigilator WebSocket
