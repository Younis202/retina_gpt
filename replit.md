# RetinaGPT — AI Ophthalmology Platform

## Project Overview

A production-grade clinical AI platform for retinal fundus image analysis. Consists of:

1. **Python FastAPI Backend** (`/api`, `/models`, `/inference`, etc.) — AI pipeline for retinal analysis
2. **Next.js Frontend** (`/frontend`) — Full clinical dashboard UI

## Architecture

### Backend (Python/FastAPI)
- FastAPI REST API with 10+ endpoints
- SQLite database for case storage (`database/retina_cases.db`)
- Foundation model for multi-task retinal analysis
- Grad-CAM explainability, SAM segmentation
- FAISS vector search for similar cases
- PDF report generation with reportlab

### Frontend (Next.js 14)
- App Router with TypeScript
- TailwindCSS + custom dark theme (slate-950 based)
- TanStack React Query for data fetching
- Zustand for client state (upload/analysis flow)
- Framer Motion for animations
- Recharts for data visualization
- next-themes for dark/light mode

## Pages
1. `/` — Landing page (hero, capabilities, stats)
2. `/dashboard` — Doctor dashboard with real-time stats
3. `/upload` — Drag-and-drop retinal scan upload
4. `/results` — Full AI analysis results with Grad-CAM viewer
5. `/cases` — Case database with filtering/pagination
6. `/cases/[id]` — Individual case detail
7. `/analytics` — Charts and statistics
8. `/settings` — Profile and system status

## Running the Project

### Frontend (port 5000)
```bash
cd frontend && npm run dev
```

### Backend (port 8000)
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The frontend proxies `/api/*` requests to `http://localhost:8000/*` via Next.js rewrites.

## Key Files
- `frontend/app/` — All Next.js pages
- `frontend/components/retina/` — Medical UI components
- `frontend/components/layout/` — App shell (Sidebar, Header)
- `frontend/lib/api.ts` — API client (axios)
- `frontend/store/index.ts` — Zustand store
- `api/main.py` — FastAPI backend
- `inference/pipeline.py` — AI inference pipeline
- `db/cases_db.py` — SQLite database operations

## Environment
- Python 3.12
- Node.js 20
- No GPU required (runs in demo mode without checkpoint)
- Backend runs in DEMO mode without `RETINA_CHECKPOINT` env var

## Deployment
- Target: autoscale
- Build: `cd frontend && npm run build`
- Run: `cd frontend && npm run start`
