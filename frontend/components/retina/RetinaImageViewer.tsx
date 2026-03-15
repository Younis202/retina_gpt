"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn, base64ToDataUrl } from "@/lib/utils";
import { Eye, Layers, ZoomIn, ZoomOut, RotateCcw } from "lucide-react";

type ViewMode = "original" | "gradcam" | "attention" | "vessel" | "optic";

interface RetinaImageViewerProps {
  originalSrc?: string;
  gradcamB64?: string | null;
  attentionB64?: string | null;
  vesselMaskB64?: string | null;
  opticDiscB64?: string | null;
  className?: string;
}

export function RetinaImageViewer({
  originalSrc,
  gradcamB64,
  attentionB64,
  vesselMaskB64,
  opticDiscB64,
  className,
}: RetinaImageViewerProps) {
  const [mode, setMode] = useState<ViewMode>("original");
  const [zoom, setZoom] = useState(1);

  const tabs: { key: ViewMode; label: string; available: boolean }[] = [
    { key: "original", label: "Original", available: !!originalSrc },
    { key: "gradcam", label: "Grad-CAM", available: !!gradcamB64 },
    { key: "attention", label: "Attention", available: !!attentionB64 },
    { key: "vessel", label: "Vessels", available: !!vesselMaskB64 },
    { key: "optic", label: "Optic Disc", available: !!opticDiscB64 },
  ].filter((t) => t.available);

  const currentSrc =
    mode === "original"
      ? originalSrc
      : mode === "gradcam" && gradcamB64
      ? base64ToDataUrl(gradcamB64)
      : mode === "attention" && attentionB64
      ? base64ToDataUrl(attentionB64)
      : mode === "vessel" && vesselMaskB64
      ? base64ToDataUrl(vesselMaskB64)
      : mode === "optic" && opticDiscB64
      ? base64ToDataUrl(opticDiscB64)
      : undefined;

  return (
    <div className={cn("space-y-3", className)}>
      {tabs.length > 1 && (
        <div className="flex gap-1.5 p-1 bg-slate-800/60 rounded-xl border border-slate-700/40 flex-wrap">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setMode(tab.key)}
              className={cn(
                "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all",
                mode === tab.key
                  ? "bg-sky-500 text-white shadow-lg shadow-sky-500/25"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-700/60"
              )}
            >
              {tab.key === "original" ? <Eye className="w-3 h-3" /> : <Layers className="w-3 h-3" />}
              {tab.label}
            </button>
          ))}
        </div>
      )}

      <div className="relative bg-slate-900 rounded-xl border border-slate-700/60 overflow-hidden aspect-square">
        <AnimatePresence mode="wait">
          {currentSrc ? (
            <motion.div
              key={mode}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="w-full h-full flex items-center justify-center"
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={currentSrc}
                alt="Retinal scan"
                className="w-full h-full object-contain"
                style={{ transform: `scale(${zoom})`, transition: "transform 0.2s" }}
              />
            </motion.div>
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <div className="text-center">
                <Eye className="w-12 h-12 text-slate-600 mx-auto mb-2" />
                <p className="text-slate-500 text-sm">No image available</p>
              </div>
            </div>
          )}
        </AnimatePresence>

        {mode === "gradcam" && gradcamB64 && (
          <div className="absolute top-2 left-2 px-2 py-1 bg-orange-500/90 text-white text-xs font-bold rounded-md">
            GRAD-CAM ACTIVE
          </div>
        )}

        <div className="absolute bottom-3 right-3 flex gap-1.5">
          <button
            onClick={() => setZoom((z) => Math.min(z + 0.25, 3))}
            className="w-7 h-7 rounded-lg bg-slate-900/90 border border-slate-700 text-slate-300 flex items-center justify-center hover:bg-slate-800 transition-colors"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setZoom((z) => Math.max(z - 0.25, 0.5))}
            className="w-7 h-7 rounded-lg bg-slate-900/90 border border-slate-700 text-slate-300 flex items-center justify-center hover:bg-slate-800 transition-colors"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setZoom(1)}
            className="w-7 h-7 rounded-lg bg-slate-900/90 border border-slate-700 text-slate-300 flex items-center justify-center hover:bg-slate-800 transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
}
