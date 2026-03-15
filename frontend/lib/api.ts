import axios from "axios";
import type {
  FullAnalysisResponse,
  HealthResponse,
  CasesStatsResponse,
  CasesResponse,
  Case,
  SearchResponse,
} from "@/types";

const api = axios.create({
  baseURL: "/api",
  timeout: 120000,
});

export const retinaApi = {
  health: async (): Promise<HealthResponse> => {
    const { data } = await api.get("/health");
    return data;
  },

  modelInfo: async () => {
    const { data } = await api.get("/model/info");
    return data;
  },

  analyze: async (
    file: File,
    options: { explain?: boolean; segment?: boolean; imageId?: string } = {}
  ): Promise<FullAnalysisResponse> => {
    const form = new FormData();
    form.append("file", file);
    form.append("explain", String(options.explain ?? true));
    form.append("segment", String(options.segment ?? false));
    if (options.imageId) form.append("image_id", options.imageId);

    const { data } = await api.post("/analyze", form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return data;
  },

  searchSimilar: async (file: File, k = 5): Promise<SearchResponse> => {
    const form = new FormData();
    form.append("file", file);
    form.append("k", String(k));

    const { data } = await api.post("/search", form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return data;
  },

  getCasesStats: async (): Promise<CasesStatsResponse> => {
    const { data } = await api.get("/cases/stats");
    return data;
  },

  getCases: async (params?: {
    limit?: number;
    offset?: number;
    patient_id?: string;
    dr_grade?: number;
    refer_only?: boolean;
  }): Promise<CasesResponse> => {
    const { data } = await api.get("/cases", { params });
    return data;
  },

  getCase: async (id: string): Promise<Case> => {
    const { data } = await api.get(`/cases/${id}`);
    return data;
  },

  generatePDF: async (
    file: File,
    patientInfo: { id: string; name?: string; age?: string; sex?: string }
  ): Promise<Blob> => {
    const form = new FormData();
    form.append("file", file);
    form.append("patient_id", patientInfo.id);
    if (patientInfo.name) form.append("patient_name", patientInfo.name);
    if (patientInfo.age) form.append("patient_age", patientInfo.age);
    if (patientInfo.sex) form.append("patient_sex", patientInfo.sex);

    const { data } = await api.post("/report/pdf", form, {
      headers: { "Content-Type": "multipart/form-data" },
      responseType: "blob",
    });
    return data;
  },
};

export default retinaApi;
