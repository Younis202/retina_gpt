"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { AppLayout } from "@/components/layout/AppLayout";
import { UploadDropzone } from "@/components/retina/UploadDropzone";
import { useAnalysisStore } from "@/store";
import retinaApi from "@/lib/api";
import { generateDemoResult } from "@/lib/demo";
import toast from "react-hot-toast";
import {
  Zap,
  Brain,
  Eye,
  Shield,
  Layers,
  AlertCircle,
  CheckCircle,
  Loader2,
} from "lucide-react";

const FEATURES = [
  { icon: Brain, label: "DR Grading (0-4)", color: "text-sky-400" },
  { icon: Eye, label: "AMD Staging", color: "text-violet-400" },
  { icon: Shield, label: "Glaucoma Detection", color: "text-emerald-400" },
  { icon: Layers, label: "Lesion Mapping", color: "text-orange-400" },
  { icon: Zap, label: "Grad-CAM Explainability", color: "text-pink-400" },
];

const STEPS = [
  { label: "Validating image quality", duration: 1500 },
  { label: "Running foundation model", duration: 2500 },
  { label: "Generating Grad-CAM maps", duration: 1500 },
  { label: "Building clinical report", duration: 1000 },
  { label: "Finalizing results", duration: 500 },
];

export default function UploadPage() {
  const router = useRouter();
  const { setFile, setResult, setIsAnalyzing, currentFile } = useAnalysisStore();
  const [localFile, setLocalFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState(-1);
  const [options, setOptions] = useState({ explain: true, segment: false });

  const handleFileSelect = (file: File) => {
    setLocalFile(file);
    setFile(file);
  };

  const handleClear = () => {
    setLocalFile(null);
    setFile(null);
  };

  const handleAnalyze = async () => {
    if (!localFile) {
      toast.error("Please select a retinal image first");
      return;
    }

    setIsLoading(true);
    setIsAnalyzing(true);
    setProgress(0);
    setCurrentStep(0);

    const totalDuration = STEPS.reduce((acc, s) => acc + s.duration, 0);
    let elapsed = 0;

    const runSteps = async () => {
      for (let i = 0; i < STEPS.length; i++) {
        setCurrentStep(i);
        await new Promise((r) => setTimeout(r, STEPS[i].duration));
        elapsed += STEPS[i].duration;
        setProgress(Math.round((elapsed / totalDuration) * 90));
      }
    };

    try {
      const stepsPromise = runSteps();
      let apiResult = null;
      let isDemo = false;

      try {
        apiResult = await retinaApi.analyze(localFile, options);
      } catch {
        isDemo = true;
      }

      await stepsPromise;

      if (isDemo) {
        apiResult = await generateDemoResult(localFile);
        toast("Running in Demo Mode — backend not detected", {
          icon: "🧪",
          style: {
            background: "#1e293b",
            color: "#94a3b8",
            border: "1px solid #334155",
            fontSize: "13px",
          },
        });
      }

      setProgress(100);
      setResult(apiResult!);
      setTimeout(() => router.push("/results"), 400);
    } catch (err: unknown) {
      const msg =
        err && typeof err === "object" && "response" in err
          ? (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
          : "Something went wrong. Please try again.";
      toast.error(msg || "Analysis failed");
      setIsLoading(false);
      setIsAnalyzing(false);
      setCurrentStep(-1);
      setProgress(0);
    }
  };

  return (
    <AppLayout title="Analyze Scan">
      <div className="max-w-4xl mx-auto space-y-6">
        <div>
          <h2 className="text-2xl font-bold text-white mb-1">Upload Retinal Scan</h2>
          <p className="text-slate-400 text-sm">
            Upload a fundus image to run the full AI diagnostic pipeline
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-5">
            <UploadDropzone
              onFileSelect={handleFileSelect}
              selectedFile={localFile}
              onClear={handleClear}
            />

            <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-4 space-y-3">
              <h3 className="text-sm font-semibold text-white">Analysis Options</h3>
              <div className="grid grid-cols-2 gap-3">
                <label className="flex items-start gap-3 p-3 bg-slate-800/40 rounded-xl border border-slate-700/40 cursor-pointer hover:border-sky-500/30 transition-all">
                  <input
                    type="checkbox"
                    checked={options.explain}
                    onChange={(e) => setOptions((o) => ({ ...o, explain: e.target.checked }))}
                    className="mt-0.5 accent-sky-500"
                  />
                  <div>
                    <p className="text-sm font-medium text-slate-200">Grad-CAM XAI</p>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Generate heatmap explaining AI decisions
                    </p>
                  </div>
                </label>
                <label className="flex items-start gap-3 p-3 bg-slate-800/40 rounded-xl border border-slate-700/40 cursor-pointer hover:border-sky-500/30 transition-all">
                  <input
                    type="checkbox"
                    checked={options.segment}
                    onChange={(e) => setOptions((o) => ({ ...o, segment: e.target.checked }))}
                    className="mt-0.5 accent-sky-500"
                  />
                  <div>
                    <p className="text-sm font-medium text-slate-200">Segmentation</p>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Vessel & optic disc masks (slower)
                    </p>
                  </div>
                </label>
              </div>
            </div>

            <AnimatePresence>
              {isLoading && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="bg-slate-900/80 border border-sky-500/20 rounded-2xl p-5 space-y-4"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full border-2 border-sky-500 border-t-transparent animate-spin" />
                    <div>
                      <p className="text-sm font-medium text-white">AI Analysis in Progress</p>
                      <p className="text-xs text-slate-400">
                        {currentStep >= 0 ? STEPS[currentStep]?.label : "Initializing..."}
                      </p>
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-slate-500">Progress</span>
                      <span className="text-sky-400 font-bold">{progress}%</span>
                    </div>
                    <div className="h-1.5 bg-slate-700/60 rounded-full overflow-hidden">
                      <motion.div
                        animate={{ width: `${progress}%` }}
                        transition={{ duration: 0.3 }}
                        className="h-full bg-gradient-to-r from-sky-500 to-blue-500 rounded-full"
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-1 gap-1.5">
                    {STEPS.map((step, i) => (
                      <div key={i} className="flex items-center gap-2 text-xs">
                        {i < currentStep ? (
                          <CheckCircle className="w-3 h-3 text-emerald-400 flex-shrink-0" />
                        ) : i === currentStep ? (
                          <Loader2 className="w-3 h-3 text-sky-400 animate-spin flex-shrink-0" />
                        ) : (
                          <div className="w-3 h-3 rounded-full border border-slate-600 flex-shrink-0" />
                        )}
                        <span
                          className={
                            i <= currentStep ? "text-slate-300" : "text-slate-600"
                          }
                        >
                          {step.label}
                        </span>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <motion.button
              whileHover={{ scale: localFile && !isLoading ? 1.02 : 1 }}
              whileTap={{ scale: localFile && !isLoading ? 0.98 : 1 }}
              onClick={handleAnalyze}
              disabled={!localFile || isLoading}
              className="w-full py-4 bg-gradient-to-r from-sky-500 to-blue-600 disabled:from-slate-700 disabled:to-slate-700 text-white disabled:text-slate-500 rounded-2xl font-bold text-lg shadow-xl shadow-sky-500/20 disabled:shadow-none transition-all flex items-center justify-center gap-3"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Zap className="w-5 h-5" />
                  Run AI Analysis
                </>
              )}
            </motion.button>
          </div>

          <div className="space-y-4">
            <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-5">
              <h3 className="text-sm font-semibold text-white mb-4">AI Capabilities</h3>
              <div className="space-y-3">
                {FEATURES.map(({ icon: Icon, label, color }) => (
                  <div key={label} className="flex items-center gap-3">
                    <div className="w-7 h-7 rounded-lg bg-slate-800/60 flex items-center justify-center">
                      <Icon className={`w-3.5 h-3.5 ${color}`} />
                    </div>
                    <span className="text-sm text-slate-300">{label}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-5">
              <div className="flex items-start gap-3">
                <AlertCircle className="w-4 h-4 text-amber-400 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-xs font-semibold text-amber-400 mb-1">Clinical Note</p>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    RetinaGPT is a clinical decision support tool. Results should be
                    reviewed by a qualified ophthalmologist before clinical action.
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-5">
              <h3 className="text-xs font-semibold text-slate-400 mb-3">Supported Formats</h3>
              <div className="grid grid-cols-3 gap-1.5">
                {["JPG", "PNG", "BMP", "TIFF", "DICOM", "RAW"].map((fmt) => (
                  <span
                    key={fmt}
                    className="text-center py-1 text-xs font-mono bg-slate-800/60 border border-slate-700/40 rounded-lg text-slate-400"
                  >
                    {fmt}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </AppLayout>
  );
}
