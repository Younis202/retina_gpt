"use client";

import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { AppLayout } from "@/components/layout/AppLayout";
import retinaApi from "@/lib/api";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar,
  AreaChart,
  Area,
  Legend,
} from "recharts";
import { BarChart3, TrendingUp, Activity, Eye, AlertTriangle } from "lucide-react";

const COLORS = ["#10b981", "#f59e0b", "#f97316", "#ef4444", "#dc2626"];
const DR_LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"];

export default function AnalyticsPage() {
  const { data: stats, isLoading } = useQuery({
    queryKey: ["cases-stats"],
    queryFn: retinaApi.getCasesStats,
    refetchInterval: 30000,
  });

  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: retinaApi.health,
  });

  const drDist = Object.entries(stats?.dr_grade_distribution ?? {}).map(
    ([grade, count]) => ({
      name: DR_LABELS[Number(grade)] ?? `Grade ${grade}`,
      grade: Number(grade),
      count,
      fill: COLORS[Number(grade)] ?? "#64748b",
    })
  );

  const mockTrend = Array.from({ length: 14 }, (_, i) => ({
    date: `D-${13 - i}`,
    cases: Math.floor(Math.random() * 30) + 5,
    referrals: Math.floor(Math.random() * 10) + 1,
    quality_pass: Math.floor(Math.random() * 25) + 5,
  }));

  const radarData = [
    { subject: "DR Grade 0", value: drDist.find((d) => d.grade === 0)?.count ?? 0 },
    { subject: "DR Grade 1", value: drDist.find((d) => d.grade === 1)?.count ?? 0 },
    { subject: "DR Grade 2", value: drDist.find((d) => d.grade === 2)?.count ?? 0 },
    { subject: "DR Grade 3", value: drDist.find((d) => d.grade === 3)?.count ?? 0 },
    { subject: "DR Grade 4", value: drDist.find((d) => d.grade === 4)?.count ?? 0 },
  ];

  const tooltipStyle = {
    background: "#0f172a",
    border: "1px solid #1e293b",
    borderRadius: "12px",
    color: "#f8fafc",
    fontSize: "12px",
  };

  return (
    <AppLayout title="Analytics">
      <div className="max-w-7xl mx-auto space-y-6">
        <div>
          <h2 className="text-2xl font-bold text-white mb-1">Analytics & Insights</h2>
          <p className="text-slate-400 text-sm">AI usage statistics and disease distribution</p>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            {
              icon: Eye,
              label: "Total Cases",
              value: stats?.total_cases ?? 0,
              color: "text-sky-400",
              bg: "bg-sky-500/10",
            },
            {
              icon: Activity,
              label: "This Week",
              value: stats?.this_week ?? 0,
              color: "text-emerald-400",
              bg: "bg-emerald-500/10",
            },
            {
              icon: AlertTriangle,
              label: "Referrals",
              value: stats?.referable_cases ?? 0,
              color: "text-rose-400",
              bg: "bg-rose-500/10",
            },
            {
              icon: TrendingUp,
              label: "Today",
              value: stats?.today ?? 0,
              color: "text-violet-400",
              bg: "bg-violet-500/10",
            },
          ].map((s, i) => (
            <motion.div
              key={s.label}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.07 }}
              className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-5"
            >
              <div className={`w-9 h-9 rounded-xl ${s.bg} flex items-center justify-center mb-3`}>
                <s.icon className={`w-5 h-5 ${s.color}`} />
              </div>
              <p className="text-3xl font-black text-white tabular-nums">{isLoading ? "—" : s.value}</p>
              <p className="text-sm text-slate-400 mt-0.5">{s.label}</p>
            </motion.div>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-5">
            <div className="flex items-center gap-2 mb-5">
              <BarChart3 className="w-4 h-4 text-slate-400" />
              <h3 className="text-sm font-semibold text-white">DR Grade Distribution</h3>
            </div>
            {drDist.length > 0 ? (
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={drDist} barSize={36}>
                  <XAxis dataKey="name" tick={{ fill: "#64748b", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#64748b", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
                  <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                    {drDist.map((entry, i) => (
                      <Cell key={i} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-52 flex items-center justify-center text-slate-600 text-sm">
                No cases to display
              </div>
            )}
          </div>

          <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-5">
            <div className="flex items-center gap-2 mb-5">
              <Activity className="w-4 h-4 text-slate-400" />
              <h3 className="text-sm font-semibold text-white">Grade Breakdown</h3>
            </div>
            {drDist.length > 0 ? (
              <div className="flex items-center gap-4">
                <ResponsiveContainer width="50%" height={220}>
                  <PieChart>
                    <Pie
                      data={drDist}
                      cx="50%"
                      cy="50%"
                      innerRadius={55}
                      outerRadius={90}
                      paddingAngle={3}
                      dataKey="count"
                    >
                      {drDist.map((entry, i) => (
                        <Cell key={i} fill={entry.fill} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={tooltipStyle} />
                  </PieChart>
                </ResponsiveContainer>
                <div className="flex-1 space-y-2">
                  {drDist.map((d) => (
                    <div key={d.name} className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full" style={{ background: d.fill }} />
                        <span className="text-slate-400">{d.name}</span>
                      </div>
                      <span className="text-white font-semibold">{d.count}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="h-52 flex items-center justify-center text-slate-600 text-sm">
                No cases to display
              </div>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-slate-900/60 border border-slate-700/60 rounded-2xl p-5">
            <div className="flex items-center gap-2 mb-5">
              <TrendingUp className="w-4 h-4 text-slate-400" />
              <h3 className="text-sm font-semibold text-white">14-Day Activity Trend</h3>
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={mockTrend}>
                <defs>
                  <linearGradient id="casesGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="refGrad2" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#f43f5e" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 10 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: "#64748b", fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: "11px", color: "#94a3b8" }} />
                <Area type="monotone" dataKey="cases" name="Total Scans" stroke="#0ea5e9" fill="url(#casesGrad)" strokeWidth={2} />
                <Area type="monotone" dataKey="referrals" name="Referrals" stroke="#f43f5e" fill="url(#refGrad2)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-5">
            <div className="flex items-center gap-2 mb-5">
              <BarChart3 className="w-4 h-4 text-slate-400" />
              <h3 className="text-sm font-semibold text-white">Severity Radar</h3>
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#1e293b" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: "#64748b", fontSize: 9 }} />
                <Radar name="Cases" dataKey="value" stroke="#0ea5e9" fill="#0ea5e9" fillOpacity={0.3} strokeWidth={2} />
                <Tooltip contentStyle={tooltipStyle} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {health && (
          <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-5">
            <h3 className="text-sm font-semibold text-white mb-4">System Status</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                { label: "AI Model", value: health.model_loaded ? "Loaded" : "Offline", ok: health.model_loaded },
                { label: "Device", value: health.device, ok: true },
                { label: "Version", value: health.version, ok: true },
                { label: "Uptime", value: `${Math.floor(health.uptime_seconds / 3600)}h`, ok: true },
              ].map(({ label, value, ok }) => (
                <div key={label} className="bg-slate-800/40 rounded-xl p-3 border border-slate-700/40">
                  <p className="text-xs text-slate-500 mb-1">{label}</p>
                  <div className="flex items-center gap-1.5">
                    <span className={`w-1.5 h-1.5 rounded-full ${ok ? "bg-emerald-400" : "bg-red-400"}`} />
                    <p className="text-sm font-medium text-white">{value}</p>
                  </div>
                </div>
              ))}
            </div>

            {health.capabilities && Object.keys(health.capabilities).length > 0 && (
              <div className="mt-4 pt-4 border-t border-slate-800/60">
                <p className="text-xs text-slate-500 mb-3">Model Capabilities</p>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(health.capabilities).map(([cap, enabled]) => (
                    <span
                      key={cap}
                      className={`px-2.5 py-1 rounded-full text-xs font-medium border ${
                        enabled
                          ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
                          : "bg-slate-800/60 border-slate-700/40 text-slate-500"
                      }`}
                    >
                      {cap.replace(/_/g, " ")}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </AppLayout>
  );
}
