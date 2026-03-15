"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { useQuery } from "@tanstack/react-query";
import { AppLayout } from "@/components/layout/AppLayout";
import { RetinaImageViewer } from "@/components/retina/RetinaImageViewer";
import { DiagnosisBadge } from "@/components/retina/DiagnosisBadge";
import { ConfidenceMeter } from "@/components/retina/ConfidenceMeter";
import { SimilarCaseCard } from "@/components/retina/SimilarCaseCard";
import { useAnalysisStore } from "@/store";
import retinaApi from "@/lib/api";
import { base64ToDataUrl } from "@/lib/utils";
import toast from "react-hot-toast";
import {
  Brain,
  Eye,
  Shield,
  FileText,
  Download,
  AlertTriangle,
  CheckCircle,
  Activity,
  Microscope,
  Clock,
} from "lucide-react";
import Link from "next/link";

export default function ResultsPage() {
  const router = useRouter();
  const { currentResult, currentFile } = useAnalysisStore();
  const [imageUrl, setImageUrl] = useState<string | undefined>();

  useEffect(() => {
    if (!currentResult) {
      router.replace("/upload");
      return;
    }
    if (currentFile) {
      const url = URL.createObjectURL(currentFile);
      setImageUrl(url);
      return () => URL.revokeObjectURL(url);
    }
  }, [currentResult, currentFile, router]);

  const { data: similarCases } = useQuery({
    queryKey: ["similar-cases", currentFile?.name],
    queryFn: () => (currentFile ? retinaApi.searchSimilar(currentFile, 6) : null),
    enabled: !!currentFile,
    retry: false,
  });

  if (!currentResult) return null;

  const r = currentResult;
  const lesionEntries = Object.entries(r.lesions ?? {});

  const handleDownloadPDF = async () => {
    if (!currentFile) {
      toast.error("Original file not found");
      return;
    }
    try {
      const blob = await retinaApi.generatePDF(currentFile, { id: r.image_id });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `retina_report_${r.image_id}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success("PDF downloaded!");
    } catch {
      toast.error("PDF generation failed. Install reportlab on backend.");
    }
  };

  return (
    <AppLayout title="Analysis Results">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-white">AI Diagnosis Results</h2>
            <p className="text-slate-400 text-sm mt-0.5">
              Image ID:{" "}
              <span className="font-mono text-sky-400">{r.image_id}</span>
              {" • "}
              <span className="text-slate-500">{r.inference_time_ms.toFixed(0)}ms inference</span>
              {" • "}
              <span className="text-slate-500">{r.model_version}</span>
            </p>
          </div>

          <div className="flex items-center gap-3">
            <Link href="/upload">
              <button className="px-4 py-2 bg-slate-800/60 border border-slate-700/60 text-slate-300 hover:text-white rounded-xl text-sm font-medium transition-colors">
                New Scan
              </button>
            </Link>
            <motion.button
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              onClick={handleDownloadPDF}
              className="flex items-center gap-2 px-4 py-2 bg-sky-500 hover:bg-sky-400 text-white rounded-xl text-sm font-medium shadow-lg shadow-sky-500/20 transition-all"
            >
              <Download className="w-4 h-4" />
              Download PDF Report
            </motion.button>
          </div>
        </div>

        {r.dr_grading.refer && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center gap-4 p-4 bg-rose-500/10 border border-rose-500/30 rounded-2xl"
          >
            <AlertTriangle className="w-5 h-5 text-rose-400 flex-shrink-0" />
            <div>
              <p className="text-rose-300 font-semibold text-sm">Ophthalmology Referral Recommended</p>
              <p className="text-rose-400/70 text-xs mt-0.5">{r.report.recommendation}</p>
            </div>
          </motion.div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="space-y-4">
            <RetinaImageViewer
              originalSrc={imageUrl}
              gradcamB64={r.explainability.gradcam_image}
              attentionB64={r.explainability.attention_image}
              vesselMaskB64={r.segmentation.vessel_mask}
              opticDiscB64={r.segmentation.optic_disc_mask}
            />

            <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-4 space-y-2.5">
              <div className="flex items-center gap-2 text-xs text-slate-400 font-medium mb-1">
                <Activity className="w-3.5 h-3.5" />
                Image Quality
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-300">Quality Score</span>
                <div className="flex items-center gap-1.5">
                  {r.quality.adequate ? (
                    <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                  ) : (
                    <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                  )}
                  <span className="text-sm font-bold text-white">
                    {(r.quality.score * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              <ConfidenceMeter value={r.quality.score} />
              {!r.quality.adequate && (
                <p className="text-xs text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-lg p-2">
                  Image quality may affect accuracy. Consider retaking.
                </p>
              )}
            </div>
          </div>

          <div className="lg:col-span-2 space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {[
                {
                  icon: Brain,
                  title: "Diabetic Retinopathy",
                  color: "text-sky-400",
                  bg: "bg-sky-500/10",
                  content: (
                    <div className="space-y-3">
                      <DiagnosisBadge
                        grade={r.dr_grading.grade}
                        label={r.dr_grading.label}
                        refer={r.dr_grading.refer}
                        size="sm"
                      />
                      <div>
                        <div className="text-xs text-slate-500 mb-1.5">
                          Grade {r.dr_grading.grade} — Confidence
                        </div>
                        <ConfidenceMeter value={r.dr_grading.confidence} />
                      </div>
                      {r.dr_grading.probabilities.length > 0 && (
                        <div className="space-y-1">
                          <p className="text-xs text-slate-500">Grade probabilities</p>
                          {r.dr_grading.probabilities.map((p, i) => (
                            <div key={i} className="flex items-center gap-2 text-xs">
                              <span className="text-slate-600 w-12">Grade {i}</span>
                              <div className="flex-1 h-1 bg-slate-700/60 rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-sky-500/60 rounded-full"
                                  style={{ width: `${p * 100}%` }}
                                />
                              </div>
                              <span className="text-slate-400 w-10 text-right">
                                {(p * 100).toFixed(0)}%
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ),
                },
                {
                  icon: Eye,
                  title: "AMD",
                  color: "text-violet-400",
                  bg: "bg-violet-500/10",
                  content: (
                    <div className="space-y-3">
                      <div className="text-sm font-semibold text-white">{r.amd.label}</div>
                      <div className="text-xs text-slate-500">Stage {r.amd.stage}</div>
                      <ConfidenceMeter value={r.amd.confidence} label="Confidence" />
                    </div>
                  ),
                },
                {
                  icon: Shield,
                  title: "Glaucoma",
                  color: "text-emerald-400",
                  bg: "bg-emerald-500/10",
                  content: (
                    <div className="space-y-3">
                      <div
                        className={`text-sm font-semibold ${
                          r.glaucoma.suspect ? "text-amber-400" : "text-emerald-400"
                        }`}
                      >
                        {r.glaucoma.suspect ? "Suspect" : "No Suspect"}
                      </div>
                      <div className="text-xs text-slate-500">
                        C/D Ratio:{" "}
                        <span className="text-white font-bold">
                          {r.glaucoma.cup_disc_ratio.toFixed(2)}
                        </span>
                      </div>
                      <ConfidenceMeter value={r.glaucoma.confidence} label="Confidence" />
                    </div>
                  ),
                },
              ].map(({ icon: Icon, title, color, bg, content }) => (
                <div
                  key={title}
                  className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-4 space-y-3"
                >
                  <div className="flex items-center gap-2">
                    <div className={`w-7 h-7 rounded-lg ${bg} flex items-center justify-center`}>
                      <Icon className={`w-3.5 h-3.5 ${color}`} />
                    </div>
                    <h3 className="text-xs font-semibold text-slate-300">{title}</h3>
                  </div>
                  {content}
                </div>
              ))}
            </div>

            {lesionEntries.length > 0 && (
              <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-4">
                <div className="flex items-center gap-2 mb-4">
                  <Microscope className="w-4 h-4 text-slate-400" />
                  <h3 className="text-sm font-semibold text-white">Lesion Detection</h3>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {lesionEntries.map(([name, info]) => (
                    <div
                      key={name}
                      className={`p-3 rounded-xl border transition-all ${
                        info.present
                          ? "bg-rose-500/10 border-rose-500/30"
                          : "bg-slate-800/40 border-slate-700/40"
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="text-xs font-medium text-slate-300 capitalize">
                          {name.replace(/_/g, " ")}
                        </span>
                        {info.present ? (
                          <span className="w-2 h-2 rounded-full bg-rose-400" />
                        ) : (
                          <span className="w-2 h-2 rounded-full bg-slate-600" />
                        )}
                      </div>
                      <div className="text-xs text-slate-500">
                        {(info.probability * 100).toFixed(0)}% probability
                      </div>
                      <div className="mt-1.5 h-1 bg-slate-700/60 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${info.present ? "bg-rose-500" : "bg-slate-600"}`}
                          style={{ width: `${info.probability * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {r.report.full_text && (
              <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-4">
                <div className="flex items-center gap-2 mb-3">
                  <FileText className="w-4 h-4 text-slate-400" />
                  <h3 className="text-sm font-semibold text-white">Clinical Report</h3>
                </div>
                <div className="bg-slate-950/60 rounded-xl p-4 text-sm text-slate-300 leading-relaxed whitespace-pre-wrap font-mono text-xs border border-slate-800/60">
                  {r.report.full_text}
                </div>
                {r.report.recommendation && (
                  <div className="mt-3 p-3 bg-sky-500/10 border border-sky-500/20 rounded-xl">
                    <p className="text-xs text-sky-300 font-medium">Recommendation</p>
                    <p className="text-sm text-sky-100 mt-1">{r.report.recommendation}</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {similarCases && similarCases.results.length > 0 && (
          <div>
            <div className="flex items-center gap-2 mb-4">
              <Eye className="w-4 h-4 text-slate-400" />
              <h3 className="text-lg font-bold text-white">Similar Cases</h3>
              <span className="px-2 py-0.5 bg-slate-800 border border-slate-700 text-slate-400 text-xs rounded-full">
                {similarCases.num_results} found
              </span>
              <span className="text-slate-600 text-xs ml-auto">
                <Clock className="w-3 h-3 inline mr-1" />
                {similarCases.search_time_ms.toFixed(0)}ms search
              </span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
              {similarCases.results.map((result, i) => (
                <SimilarCaseCard key={result.image_id} result={result} index={i} />
              ))}
            </div>
          </div>
        )}
      </div>
    </AppLayout>
  );
}
