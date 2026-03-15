"use client";

import { motion } from "framer-motion";
import { cn, getDRGradeBg, getDRGradeLabel } from "@/lib/utils";
import { Eye, TrendingUp } from "lucide-react";
import type { SearchResultItem } from "@/types";

interface SimilarCaseCardProps {
  result: SearchResultItem;
  index: number;
}

export function SimilarCaseCard({ result, index }: SimilarCaseCardProps) {
  const similarity = Math.round(result.score * 100);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      className="group bg-slate-800/40 border border-slate-700/60 rounded-xl p-4 hover:border-sky-500/30 hover:bg-slate-800/60 transition-all cursor-pointer"
    >
      <div className="flex items-start justify-between mb-3">
        <div className="w-10 h-10 rounded-lg bg-slate-700/60 border border-slate-600/40 flex items-center justify-center">
          <Eye className="w-5 h-5 text-slate-400" />
        </div>
        <div className="text-right">
          <div className="text-lg font-bold text-white">#{result.rank}</div>
          <div className="text-xs text-slate-500">rank</div>
        </div>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-400">Similarity</span>
          <div className="flex items-center gap-1.5">
            <TrendingUp className="w-3 h-3 text-sky-400" />
            <span className="text-sm font-bold text-sky-400">{similarity}%</span>
          </div>
        </div>

        <div className="h-1.5 bg-slate-700/60 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${similarity}%` }}
            transition={{ duration: 0.8, delay: index * 0.05 + 0.3 }}
            className="h-full bg-gradient-to-r from-sky-500 to-blue-500 rounded-full"
          />
        </div>

        {result.dr_grade !== null && (
          <div className="pt-1">
            <span
              className={cn(
                "inline-flex px-2 py-0.5 rounded-full text-xs font-medium border",
                getDRGradeBg(result.dr_grade ?? 0)
              )}
            >
              {result.dr_label ?? getDRGradeLabel(result.dr_grade ?? 0)}
            </span>
          </div>
        )}

        <p className="text-xs text-slate-500 truncate font-mono">
          {result.image_id}
        </p>
        {result.dataset && (
          <p className="text-xs text-slate-600">{result.dataset}</p>
        )}
      </div>
    </motion.div>
  );
}
