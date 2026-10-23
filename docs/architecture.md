# Architecture

```mermaid
flowchart LR
  subgraph Field["Field"]
    CT[Camera trap: video clip<br/>or burst of stills]
  end
  subgraph AWS["AWS: EC2 on Graviton (t4g.medium, arm64), Docker"]
    API[FastAPI<br/>POST /v1/triage]
    subgraph Agent["TriageAgent: perception, decision, action"]
      M[detect_motion<br/>OpenCV 5: MOG2 for video,<br/>median-background difference for bursts]
      Q[assess_quality<br/>brightness, Laplacian sharpness,<br/>infrared test]
      E[enhance<br/>adaptive gamma, CLAHE, unsharp mask]
      D[detect_animals<br/>MegaDetector v5a]
      C[classify<br/>SpeciesNet v4.0.3a]
      P[decide<br/>policy, watchlist, look-alikes]
    end
    T[(trace.json)]
  end
  subgraph People["People"]
    R[Human review]
    A[Alert approval]
  end
  CT --> API --> M
  M -- no motion twice --> D
  M -- events --> Q
  D -- box --> Q
  Q -- poor --> E --> Q
  Q -- usable --> C
  C -- unsure --> Q
  C -- blank but motion --> D
  C --> P
  P -- log --> T
  P -- discard --> T
  P -- review --> R
  P -- alert_pending_approval --> A
```

**The loop is the point.** OpenCV 5 measurements decide which tool is called next:

| Visual evidence | Next action |
| --- | --- |
| No motion found | Retry motion detection with a lower threshold |
| Still no motion | Call MegaDetector (slower, finds animals that stood still) |
| Frame dark or blurry | Enhance, then re-measure; if still poor, move to the next sharpest frame |
| Classifier unsure | Re-classify the enhanced crop, then try another frame (up to 3 attempts) |
| Motion found but classifier says "blank" | Ask MegaDetector; re-crop around its box, or discard if it agrees |
| Top answer is a look-alike of a watch-listed species | Send to human review |
| Watch-listed species | Raise an alert that waits for a person to approve it |

Every step is recorded as `{tool, inputs, observation, decision}` in the trace returned by the API.
