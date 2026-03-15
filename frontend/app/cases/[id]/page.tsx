"use client";

import { useParams, useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { AppLayout } from "@/components/layout/AppLayout";
import { DiagnosisBadge } from "@/components/retina/DiagnosisBadge";
import { ConfidenceMeter } from "@/components/retina/ConfidenceMeter";
import { RetinaImageViewer } from "@/components/retina/RetinaImageViewer";
import retinaApi from "@/lib/api";
import { formatDate, getDRGradeLabel } from "@/lib/utils";
import type { FullAnalysisResponse } from "@/types";
import {
  ArrowLeft,
  Brain,
  Eye,
  Shield,
  FileText,
  Activity,
  AlertTriangle,
  Microscope,
  Clock,
  User,
} from "lucide-react";
import Link from "next/link";

export default function CaseDetailPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;

  const { data: caseData, isLoading } = useQuery({
    queryKey: ["case", id],
    queryFn: () => retinaApi.getCase(id),
    retry: 1,
  });

  if (isLoading) {
    return (
      <AppLayout title="Case Detail">
        <div className="flex items-center justify-center h-64">
          <div className="w-8 h-8 border-2 border-sky-500 border-t-transparent rounded-full animate-spin" />
        </div>
      </AppLayout>
    );
  }

  if (!caseData) {
    return (
      <AppLayout title="Case Not Found">
        <div className="text-center py-16">
          <Eye className="w-12 h-12 text-slate-700 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-white mb-2">Case Not Found</h2>
          <p className="text-slate-400 mb-6">Case ID: {id}</p>
          <Link href="/cases">
            <button className="px-4 py-2 bg-sky-500 text-white rounded-xl text-sm font-medium">
              Back to Cases
            </button>
          </Link>
        </div>
      </AppLayout>
    );
  }

  let result: FullAnalysisResponse | null = null;
  try {
    result = typeof caseData.full_result === "string"
      ? JSON.parse(caseData.full_result)
      : caseData.full_result as unknown as FullAnalysisResponse;
  } catch {
    result = null;
  }

  return (
    <AppLayout title="Case Detail">
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="flex items-center gap-4">
          <button
            onClick={() => router.back()}
            className="w-9 h-9 rounded-xl bg-slate-800/60 border border-slate-700/60 flex items-center justify-center text-slate-400 hover:text-white transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
          </button>
          <div>
            <h2 className="text-2xl font-bold text-white">Case Detail</h2>
            <p className="text-slate-400 text-sm font-mono">{id}</p>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { icon: User, label: "Patient ID", value: caseData.patient_id, color: "text-sky-400", bg: "bg-sky-500/10" },
            { icon: Brain, label: "Diagnosis", value: getDRGradeLabel(caseData.dr_grade), color: "text-violet-400", bg: "bg-violet-500/10" },
            { icon: Clock, label: "Analyzed", value: formatDate(caseData.created_at), color: "text-emerald-400", bg: "bg-emerald-500/10" },
            { icon: Activity, label: "Status", value: caseData.status, color: "text-amber-400", bg: "bg-amber-500/10" },
          ].map(({ icon: Icon, label, value, color, bg }, i) => (
            <motion.div
              key={label}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
              className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-4"
            >
              <div className={`w-8 h-8 rounded-lg ${bg} flex items-center justify-center mb-3`}>
                <Icon className={`w-4 h-4 ${color}`} />
              </div>
              <p className="text-xs text-slate-500 mb-1">{label}</p>
              <p className="text-sm font-semibold text-white truncate">{value}</p>
            </motion.div>
          ))}
        </div>

        {caseData.dr_refer === 1 && (
          <div className="flex items-center gap-3 p-4 bg-rose-500/10 border border-rose-500/30 rounded-2xl">
            <AlertTriangle className="w-5 h-5 text-rose-400 flex-shrink-0" />
            <p className="text-rose-300 font-medium text-sm">
              Ophthalmology referral recommended for this case
            </p>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="space-y-4">
            {result && (
              <RetinaImageViewer
                gradcamB64={result.explainability?.gradcam_image}
                attentionB64={result.explainability?.attention_image}
                vesselMaskB64={result.segmentation?.vessel_mask}
                opticDiscB64={result.segmentation?.optic_disc_mask}
              />
            )}

            <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-4 space-y-3">
              <h3 className="text-sm font-semibold text-white">Diagnosis Summary</h3>
              <DiagnosisBadge
                grade={caseData.dr_grade}
                label={caseData.dr_label || getDRGradeLabel(caseData.dr_grade)}
                refer={!!caseData.dr_refer}
              />
              <div>
                <p className="text-xs text-slate-500 mb-1.5">Confidence</p>
                <ConfidenceMeter value={caseData.dr_confidence} showPercentage />
              </div>
              <div>
                <p className="text-xs text-slate-500 mb-1.5">Image Quality</p>
                <ConfidenceMeter value={caseData.quality_score} showPercentage />
              </div>
            </div>
          </div>

          {result && (
            <div className="lg:col-span-2 space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Eye className="w-4 h-4 text-violet-400" />
                    <h3 className="text-sm font-semibold text-white">AMD</h3>
                  </div>
                  <p className="text-sm text-white font-medium">{result.amd.label}</p>
                  <p className="text-xs text-slate-500 mt-1">Stage {result.amd.stage}</p>
                  <div className="mt-2">
                    <ConfidenceMeter value={result.amd.confidence} label="Confidence" />
                  </div>
                </div>

                <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Shield className="w-4 h-4 text-emerald-400" />
                    <h3 className="text-sm font-semibold text-white">Glaucoma</h3>
                  </div>
                  <p className={`text-sm font-medium ${result.glaucoma.suspect ? "text-amber-400" : "text-emerald-400"}`}>
                    {result.glaucoma.suspect ? "Glaucoma Suspect" : "No Suspicion"}
                  </p>
                  <p className="text-xs text-slate-500 mt-1">
                    C/D Ratio: {result.glaucoma.cup_disc_ratio.toFixed(3)}
                  </p>
                  <div className="mt-2">
                    <ConfidenceMeter value={result.glaucoma.confidence} label="Confidence" />
                  </div>
                </div>
              </div>

              {Object.keys(result.lesions ?? {}).length > 0 && (
                <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-4">
                  <div className="flex items-center gap-2 mb-4">
                    <Microscope className="w-4 h-4 text-slate-400" />
                    <h3 className="text-sm font-semibold text-white">Lesion Findings</h3>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries(result.lesions).map(([name, info]) => (
                      <div
                        key={name}
                        className={`p-2.5 rounded-xl border ${
                          info.present
                            ? "bg-rose-500/10 border-rose-500/20"
                            : "bg-slate-800/40 border-slate-700/40"
                        }`}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs font-medium text-slate-300 capitalize">
                            {name.replace(/_/g, " ")}
                          </span>
                          <span
                            className={`w-1.5 h-1.5 rounded-full ${
                              info.present ? "bg-rose-400" : "bg-slate-600"
                            }`}
                          />
                        </div>
                        <span className="text-xs text-slate-500">
                          {(info.probability * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {result.report?.full_text && (
                <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <FileText className="w-4 h-4 text-slate-400" />
                    <h3 className="text-sm font-semibold text-white">Clinical Report</h3>
                  </div>
                  <pre className="text-xs text-slate-300 leading-relaxed whitespace-pre-wrap font-mono bg-slate-950/60 rounded-xl p-4 border border-slate-800/60">
                    {result.report.full_text}
                  </pre>
                  {result.report.recommendation && (
                    <div className="mt-3 p-3 bg-sky-500/10 border border-sky-500/20 rounded-xl">
                      <p className="text-xs text-sky-300 font-medium mb-1">Recommendation</p>
                      <p className="text-sm text-sky-100">{result.report.recommendation}</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </AppLayout>
  );
}
