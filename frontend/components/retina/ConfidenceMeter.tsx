"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface ConfidenceMeterProps {
  value: number;
  label?: string;
  showPercentage?: boolean;
}

export function ConfidenceMeter({ value, label, showPercentage = true }: ConfidenceMeterProps) {
  const percentage = Math.round(value * 100);
  const getColor = () => {
    if (percentage >= 80) return "from-emerald-500 to-teal-500";
    if (percentage >= 60) return "from-sky-500 to-blue-500";
    if (percentage >= 40) return "from-yellow-500 to-amber-500";
    return "from-red-500 to-rose-500";
  };

  return (
    <div className="space-y-1.5">
      {(label || showPercentage) && (
        <div className="flex items-center justify-between">
          {label && (
            <span className="text-xs text-slate-400 font-medium">{label}</span>
          )}
          {showPercentage && (
            <span className="text-xs font-bold text-white ml-auto">{percentage}%</span>
          )}
        </div>
      )}
      <div className="h-2 bg-slate-700/60 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 1, ease: "easeOut", delay: 0.2 }}
          className={cn("h-full rounded-full bg-gradient-to-r", getColor())}
        />
      </div>
    </div>
  );
}
