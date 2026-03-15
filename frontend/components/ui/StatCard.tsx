"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import type { LucideIcon } from "lucide-react";

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  iconColor?: string;
  iconBg?: string;
  trend?: { value: number; label: string };
  index?: number;
}

export function StatCard({
  title,
  value,
  subtitle,
  icon: Icon,
  iconColor = "text-sky-400",
  iconBg = "bg-sky-500/10",
  trend,
  index = 0,
}: StatCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-5 hover:border-slate-600/60 transition-all group"
    >
      <div className="flex items-start justify-between mb-4">
        <div className={cn("w-10 h-10 rounded-xl flex items-center justify-center", iconBg)}>
          <Icon className={cn("w-5 h-5", iconColor)} />
        </div>
        {trend && (
          <div
            className={cn(
              "flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full",
              trend.value >= 0
                ? "text-emerald-400 bg-emerald-500/10"
                : "text-red-400 bg-red-500/10"
            )}
          >
            <span>{trend.value >= 0 ? "+" : ""}{trend.value}%</span>
            <span className="text-slate-500">{trend.label}</span>
          </div>
        )}
      </div>

      <div className="space-y-0.5">
        <p className="text-3xl font-bold text-white tabular-nums">{value}</p>
        <p className="text-sm font-medium text-slate-400">{title}</p>
        {subtitle && <p className="text-xs text-slate-600">{subtitle}</p>}
      </div>
    </motion.div>
  );
}
