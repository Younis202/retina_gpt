"use client";

import { motion } from "framer-motion";
import { cn, getDRGradeBg } from "@/lib/utils";
import { AlertTriangle, CheckCircle, AlertCircle } from "lucide-react";

interface DiagnosisBadgeProps {
  grade: number;
  label: string;
  refer?: boolean;
  size?: "sm" | "md" | "lg";
}

export function DiagnosisBadge({ grade, label, refer, size = "md" }: DiagnosisBadgeProps) {
  const Icon = grade === 0 ? CheckCircle : grade >= 3 ? AlertTriangle : AlertCircle;

  return (
    <motion.div
      initial={{ scale: 0.9, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className={cn(
        "inline-flex items-center gap-2 rounded-full border font-medium",
        getDRGradeBg(grade),
        size === "sm" && "px-2.5 py-1 text-xs",
        size === "md" && "px-3 py-1.5 text-sm",
        size === "lg" && "px-4 py-2 text-base"
      )}
    >
      <Icon className={cn(size === "sm" ? "w-3 h-3" : "w-4 h-4")} />
      <span>{label}</span>
      {refer && (
        <span className="ml-1 px-1.5 py-0.5 bg-red-500 text-white text-xs rounded-full">
          REFER
        </span>
      )}
    </motion.div>
  );
}
