# Financial Pragmatic AI

Full-stack transcript analysis system for earnings calls.

## Architecture

React UI  
-> FastAPI backend  
-> EarningsCallAnalyzer  
-> Financial NLP pipeline

- UI: http://localhost:3000
- API: http://127.0.0.1:8000

## Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn api.server:app --reload
```

## Frontend

```bash
cd frontend
npm install
npm start
```

## API

`POST /analyze`

Request body:

```json
{
  "transcript": "CEO: ..."
}
```

Response body:

```json
{
  "segments": [],
  "signal": "risk",
  "insight": "Conversation suggests potential margin or profitability risk."
}
```
