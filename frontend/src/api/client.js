import axios from "axios";

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
});

export async function analyzeTranscript(transcript) {
  const response = await api.post("/analyze", { transcript });
  return response.data;
}

export async function compareTranscripts(transcript1, transcript2) {
  const response = await api.post("/compare", {
    transcript_1: transcript1,
    transcript_2: transcript2,
  });
  return response.data;
}

export async function uploadTranscript(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await api.post("/upload", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
}
