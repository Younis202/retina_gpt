"use client";

import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { AppLayout } from "@/components/layout/AppLayout";
import { StatCard } from "@/components/ui/StatCard";
import { DiagnosisBadge } from "@/components/retina/DiagnosisBadge";
import { ConfidenceMeter } from "@/components/retina/ConfidenceMeter";
import retinaApi from "@/lib/api";
import { formatDate, getDRGradeLabel } from "@/lib/utils";
import {
  Activity,
  Eye,
  AlertTriangle,
  TrendingUp,
  Clock,
  Database,
  RefreshCw,
  ArrowRight,
} from "lucide-react";
import Link from "next/link";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";

const PIE_COLORS = ["#10b981", "#f59e0b", "#f97316", "#ef4444", "#dc2626"];

const DR_LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"];

export default function DashboardPage() {
  const { data: stats, isLoading: statsLoading, refetch: refetchStats } = useQuery({
    queryKey: ["cases-stats"],
    queryFn: retinaApi.getCasesStats,
    refetchInterval: 30000,
  });

  const { data: casesData, isLoading: casesLoading } = useQuery({
    queryKey: ["recent-cases"],
    queryFn: () => retinaApi.getCases({ limit: 8 }),
    refetchInterval: 30000,
  });

  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: retinaApi.health,
    refetchInterval: 10000,
  });

  const pieData = Object.entries(stats?.dr_grade_distribution ?? {}).map(
    ([grade, count]) => ({
      name: DR_LABELS[Number(grade)] ?? `Grade ${grade}`,
      value: count,
    })
  );

  const mockActivityData = Array.from({ length: 7 }, (_, i) => ({
    day: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i],
    scans: Math.floor(Math.random() * 40) + 10,
    referrals: Math.floor(Math.random() * 15) + 2,
  }));

  return (
    <AppLayout title="Dashboard">
      <div className="space-y-6 max-w-7xl">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-white">Clinical Overview</h2>
            <p className="text-slate-400 text-sm mt-0.5">
              Real-time AI analysis statistics
            </p>
          </div>
          <div className="flex items-center gap-3">
            {health && (
              <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-800/60 border border-slate-700/60 rounded-xl">
                <span
                  className={`w-2 h-2 rounded-full ${
                    health.model_loaded ? "bg-emerald-400 animate-pulse" : "bg-yellow-400"
                  }`}
                />
                <span className="text-xs text-slate-400 font-medium">
                  {health.model_loaded ? "AI Online" : "Initializing"}
                </span>
              </div>
            )}
            <button
              onClick={() => refetchStats()}
              className="flex items-center gap-2 px-3 py-1.5 bg-slate-800/60 border border-slate-700/60 rounded-xl text-slate-400 hover:text-slate-200 text-sm transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Refresh
            </button>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard
            title="Total Cases"
            value={statsLoading ? "—" : (stats?.total_cases ?? 0).toLocaleString()}
            icon={Database}
            iconColor="text-sky-400"
            iconBg="bg-sky-500/10"
            index={0}
          />
          <StatCard
            title="Cases Today"
            value={statsLoading ? "—" : stats?.today ?? 0}
            icon={Activity}
            iconColor="text-emerald-400"
            iconBg="bg-emerald-500/10"
            index={1}
          />
          <StatCard
            title="This Week"
            value={statsLoading ? "—" : stats?.this_week ?? 0}
            icon={TrendingUp}
            iconColor="text-violet-400"
            iconBg="bg-violet-500/10"
            index={2}
          />
          <StatCard
            title="Referrals"
            value={statsLoading ? "—" : stats?.referable_cases ?? 0}
            subtitle="Require ophthalmology"
            icon={AlertTriangle}
            iconColor="text-rose-400"
            iconBg="bg-rose-500/10"
            index={3}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-slate-900/60 border border-slate-700/60 rounded-2xl p-5">
            <div className="flex items-center justify-between mb-5">
              <div>
                <h3 className="text-sm font-semibold text-white">Scan Activity</h3>
                <p className="text-xs text-slate-500">Last 7 days</p>
              </div>
              <div className="flex items-center gap-3 text-xs">
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-sky-500" />
                  <span className="text-slate-400">Scans</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-rose-500" />
                  <span className="text-slate-400">Referrals</span>
                </div>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={mockActivityData}>
                <defs>
                  <linearGradient id="scanGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="refGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#f43f5e" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="day" tick={{ fill: "#64748b", fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: "#64748b", fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{
                    background: "#0f172a",
                    border: "1px solid #1e293b",
                    borderRadius: "12px",
                    color: "#f8fafc",
                    fontSize: "12px",
                  }}
                />
                <Area type="monotone" dataKey="scans" stroke="#0ea5e9" fill="url(#scanGrad)" strokeWidth={2} />
                <Area type="monotone" dataKey="referrals" stroke="#f43f5e" fill="url(#refGrad)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-5">
            <h3 className="text-sm font-semibold text-white mb-4">DR Distribution</h3>
            {pieData.length > 0 ? (
              <div className="space-y-3">
                <ResponsiveContainer width="100%" height={160}>
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%"
                      cy="50%"
                      innerRadius={45}
                      outerRadius={75}
                      paddingAngle={3}
                      dataKey="value"
                    >
                      {pieData.map((_, i) => (
                        <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        background: "#0f172a",
                        border: "1px solid #1e293b",
                        borderRadius: "12px",
                        color: "#f8fafc",
                        fontSize: "12px",
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
                <div className="space-y-1.5">
                  {pieData.map((item, i) => (
                    <div key={item.name} className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: PIE_COLORS[i % PIE_COLORS.length] }} />
                        <span className="text-slate-400">{item.name}</span>
                      </div>
                      <span className="text-white font-medium">{item.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="h-40 flex items-center justify-center text-slate-600 text-sm">
                No cases yet
              </div>
            )}
          </div>
        </div>

        <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl overflow-hidden">
          <div className="flex items-center justify-between p-5 border-b border-slate-800/60">
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-slate-400" />
              <h3 className="text-sm font-semibold text-white">Recent Cases</h3>
            </div>
            <Link href="/cases">
              <button className="flex items-center gap-1.5 text-sky-400 hover:text-sky-300 text-xs font-medium transition-colors">
                View all
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </Link>
          </div>

          {casesLoading ? (
            <div className="p-8 flex items-center justify-center">
              <div className="w-5 h-5 border-2 border-sky-500 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : casesData?.cases.length === 0 ? (
            <div className="p-12 text-center">
              <Eye className="w-10 h-10 text-slate-700 mx-auto mb-3" />
              <p className="text-slate-500 font-medium">No cases analyzed yet</p>
              <p className="text-slate-600 text-sm mt-1">
                Upload a retinal scan to get started
              </p>
              <Link href="/upload">
                <button className="mt-4 px-4 py-2 bg-sky-500 text-white rounded-xl text-sm font-medium hover:bg-sky-400 transition-colors">
                  Analyze First Scan
                </button>
              </Link>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-800/60">
                    <th className="text-left px-5 py-3 text-xs text-slate-500 font-medium">Patient ID</th>
                    <th className="text-left px-5 py-3 text-xs text-slate-500 font-medium">Diagnosis</th>
                    <th className="text-left px-5 py-3 text-xs text-slate-500 font-medium">Confidence</th>
                    <th className="text-left px-5 py-3 text-xs text-slate-500 font-medium">Date</th>
                    <th className="text-left px-5 py-3 text-xs text-slate-500 font-medium">Status</th>
                    <th className="px-5 py-3" />
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/40">
                  {casesData?.cases.map((c) => (
                    <motion.tr
                      key={c.id}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="hover:bg-slate-800/30 transition-colors"
                    >
                      <td className="px-5 py-3.5">
                        <div className="font-medium text-white text-xs font-mono">{c.patient_id}</div>
                        <div className="text-slate-600 text-xs">{c.image_name || "—"}</div>
                      </td>
                      <td className="px-5 py-3.5">
                        <DiagnosisBadge
                          grade={c.dr_grade}
                          label={getDRGradeLabel(c.dr_grade)}
                          refer={!!c.dr_refer}
                          size="sm"
                        />
                      </td>
                      <td className="px-5 py-3.5 w-32">
                        <ConfidenceMeter value={c.dr_confidence} showPercentage />
                      </td>
                      <td className="px-5 py-3.5 text-slate-400 text-xs whitespace-nowrap">
                        {formatDate(c.created_at)}
                      </td>
                      <td className="px-5 py-3.5">
                        <span className="px-2 py-0.5 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-full text-xs font-medium">
                          {c.status}
                        </span>
                      </td>
                      <td className="px-5 py-3.5">
                        <Link href={`/cases/${c.id}`}>
                          <button className="text-slate-500 hover:text-sky-400 transition-colors">
                            <ArrowRight className="w-3.5 h-3.5" />
                          </button>
                        </Link>
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </AppLayout>
  );
}
