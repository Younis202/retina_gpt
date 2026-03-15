import { create } from "zustand";
import type { FullAnalysisResponse } from "@/types";

interface AnalysisStore {
  currentFile: File | null;
  currentResult: FullAnalysisResponse | null;
  isAnalyzing: boolean;
  uploadProgress: number;
  setFile: (file: File | null) => void;
  setResult: (result: FullAnalysisResponse | null) => void;
  setIsAnalyzing: (v: boolean) => void;
  setUploadProgress: (v: number) => void;
  reset: () => void;
}

export const useAnalysisStore = create<AnalysisStore>((set) => ({
  currentFile: null,
  currentResult: null,
  isAnalyzing: false,
  uploadProgress: 0,
  setFile: (file) => set({ currentFile: file }),
  setResult: (result) => set({ currentResult: result }),
  setIsAnalyzing: (v) => set({ isAnalyzing: v }),
  setUploadProgress: (v) => set({ uploadProgress: v }),
  reset: () =>
    set({
      currentFile: null,
      currentResult: null,
      isAnalyzing: false,
      uploadProgress: 0,
    }),
}));
