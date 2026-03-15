import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatConfidence(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function getDRGradeColor(grade: number): string {
  const colors: Record<number, string> = {
    0: "text-emerald-500",
    1: "text-yellow-500",
    2: "text-orange-500",
    3: "text-red-500",
    4: "text-rose-600",
  };
  return colors[grade] ?? "text-gray-500";
}

export function getDRGradeBg(grade: number): string {
  const colors: Record<number, string> = {
    0: "bg-emerald-500/10 border-emerald-500/20 text-emerald-600 dark:text-emerald-400",
    1: "bg-yellow-500/10 border-yellow-500/20 text-yellow-600 dark:text-yellow-400",
    2: "bg-orange-500/10 border-orange-500/20 text-orange-600 dark:text-orange-400",
    3: "bg-red-500/10 border-red-500/20 text-red-600 dark:text-red-400",
    4: "bg-rose-500/10 border-rose-500/20 text-rose-600 dark:text-rose-400",
  };
  return colors[grade] ?? "bg-gray-500/10 border-gray-500/20 text-gray-600";
}

export function getDRGradeLabel(grade: number): string {
  const labels: Record<number, string> = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative",
  };
  return labels[grade] ?? "Unknown";
}

export function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatUptime(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${minutes}m`;
}

export function base64ToDataUrl(base64: string, mimeType = "image/png"): string {
  return `data:${mimeType};base64,${base64}`;
}

export function getRiskLevel(grade: number): { label: string; color: string } {
  if (grade === 0) return { label: "Low Risk", color: "text-emerald-500" };
  if (grade === 1) return { label: "Mild Risk", color: "text-yellow-500" };
  if (grade === 2) return { label: "Moderate Risk", color: "text-orange-500" };
  if (grade === 3) return { label: "High Risk", color: "text-red-500" };
  return { label: "Critical", color: "text-rose-600" };
}
