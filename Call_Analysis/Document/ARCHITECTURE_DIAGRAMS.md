## FYP Architecture Diagrams

This document Mermaid diagrams for Call Analysis FYP. 

### 1) System Block Diagram (End-to-End Architecture)
```mermaid
flowchart LR
  %% Styles
  classDef box fill:#111827,stroke:#8b5cf6,stroke-width:1px,color:#e5e7eb,rx:6,ry:6
  classDef store fill:#0b1220,stroke:#06b6d4,color:#a7f3d0,rx:6,ry:6
  classDef ext fill:#0b1325,stroke:#f59e0b,color:#fde68a,rx:6,ry:6
  classDef proc fill:#101828,stroke:#22c55e,color:#d1fae5,rx:6,ry:6
  classDef io fill:#0f172a,stroke:#e11d48,color:#fecdd3,rx:6,ry:6

  %% Users
  U["Agent / Analyst"]:::ext

  %% Frontend
  subgraph FE["Frontend - Next.js App Router + Tailwind + Recharts"]
    direction TB
    FE_UI["UI: Upload, Dashboard, History, About"]:::box
    FE_Charts["Visualizations: Sentiment, Emotions, Gauge, Key Phrases"]:::box
    FE_Services["API Client Axios, Toasts, Router"]:::box
    FE_Exports["Client triggers: Export PDF CSV JSON"]:::box
  end

  %% Backend
  subgraph BE["Backend - Flask API"]
    direction TB

    API_Root["/ /health /api/* /"]:::io

    subgraph API["REST Endpoints"]
      direction LR
      UP["POST /api/upload/"]:::io
      AN["POST /api/analyze/"]:::io
      ST["GET /api/status/:id/"]:::io
      RS["GET /api/results/:id/"]:::io
      HI["GET /api/history/"]:::io
      PDF["GET /api/export/:id/pdf/"]:::io
      CSV["GET /api/export/:id/csv/"]:::io
    end

    subgraph Services["Core Services"]
      direction TB

      PRE["Preprocessing<br/>- Audio validation convert<br/>- Whisper ASR<br/>- Pyannote Diarization<br/>- PII Masking spaCy<br/>- WER jiwer"]:::proc

      TXT["Text Processing<br/>- PII Masking spaCy regex<br/>- Normalization"]:::proc

      FEAT["Feature Extraction<br/>- Acoustic Text features"]:::proc

      ML["Models<br/>- Sentiment: DistilBERT<br/>- Emotion: CNN+LSTM<br/>- Sale Probability: XGBoost"]:::proc

      DASH["Dashboard Builder<br/>- Sentiment over time<br/>- Emotion distribution<br/>- Key phrases<br/>- Gauges KPIs"]:::proc

      EXP["Reports<br/>- PDF ReportLab<br/>- CSV Python csv<br/>- JSON"]:::proc
    end
  end

  %% Storage
  subgraph STG["Storage"]
    direction TB
    DB[("MongoDB Atlas<br/>- calls collection<br/>- results metadata")]:::store
    UPF[("Uploads Folder<br/>- Raw audio<br/>- Temp artifacts")]:::store
    LOGS[("Logs")]:::store
  end

  %% External
  HF["Hugging Face Hub<br/>- Pyannote pipelines<br/>- HF_TOKEN"]:::ext

  %% Flows
  U --> FE_UI
  FE_UI --> FE_Services
  FE_Services -->|"Upload audio"| UP
  FE_Services -->|"Run analysis"| AN
  FE_Services -->|"Poll status"| ST
  FE_Services -->|"Fetch results"| RS
  FE_Services -->|"History"| HI
  FE_Services -->|"Export PDF CSV"| PDF
  FE_Services -->|"Export PDF CSV"| CSV
  FE_Services --> FE_Charts
  FE_Services --> FE_Exports

  %% Backend flows
  UP -->|"Save file"| UPF
  UP -->|"Create call record"| DB

  AN --> PRE
  PRE --> TXT
  TXT --> FEAT
  FEAT --> ML
  ML --> DASH
  DASH -->|"Persist results"| DB

  RS -->|"Read results"| DB
  HI -->|"List calls"| DB

  PDF -->|"Read results"| DB
  CSV -->|"Read results"| DB

  PRE -->|"Auth models"| HF

  %% Logs
  Services --> LOGS
```

### 2) Processing Pipeline (Detailed)
```mermaid
flowchart LR
  classDef step fill:#0f172a,stroke:#22c55e,color:#d1fae5,rx:6,ry:6
  classDef model fill:#0b1325,stroke:#8b5cf6,color:#ddd6fe,rx:6,ry:6
  classDef data fill:#0b1220,stroke:#06b6d4,color:#a7f3d0,rx:6,ry:6

  A["Audio File wav mp3 m4a"]:::data --> B["Validate Convert"]:::step
  B --> C["ASR - Whisper Transcription"]:::model
  C --> D["Speaker Diarization - Pyannote"]:::model
  D --> E["PII Masking - spaCy Regex"]:::step
  E --> F["Text Acoustic Feature Extraction"]:::step
  F --> G["Sentiment - DistilBERT"]:::model
  F --> H["Emotion - CNN+LSTM"]:::model
  F --> I["Sale Probability - XGBoost"]:::model
  G --> J["Dashboard Assembly: Sentiment over time"]:::step
  H --> J
  I --> J
  J --> K[("Store Results in MongoDB")]:::data
```

### 3) Request Lifecycle (Upload → Analyze → Results)
```mermaid
sequenceDiagram
  autonumber
  participant UI as Frontend (Next.js)
  participant API as Flask API
  participant PRE as Preprocessing/ML
  participant DB as MongoDB
  participant FS as Uploads Folder

  UI->>API: POST /api/upload (multipart audio)
  API->>FS: Save raw audio
  API->>DB: Create call record (status=pending)
  API-->>UI: { call_id, filename, size }

  UI->>API: POST /api/analyze { call_id }
  API->>DB: Update status=processing
  API->>PRE: Start pipeline (ASR, diarization, masking, features, models)
  PRE->>DB: Save results (sentiment/emotions/sale prob, key phrases)
  PRE->>DB: Update status=completed
  API-->>UI: { started: true }

  loop Poll until completed
    UI->>API: GET /api/status/:call_id
    API-->>UI: { status, progress }
  end

  UI->>API: GET /api/results/:call_id
  API->>DB: Read results
  API-->>UI: { results payload }

  alt Export
    UI->>API: GET /api/export/:call_id/pdf
    API-->>UI: application/pdf
    UI->>API: GET /api/export/:call_id/csv
    API-->>UI: text/csv
  end
```


