"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { AppLayout } from "@/components/layout/AppLayout";
import { DiagnosisBadge } from "@/components/retina/DiagnosisBadge";
import { ConfidenceMeter } from "@/components/retina/ConfidenceMeter";
import retinaApi from "@/lib/api";
import { formatDate, getDRGradeLabel } from "@/lib/utils";
import {
  Search,
  Filter,
  Eye,
  ChevronLeft,
  ChevronRight,
  ArrowRight,
  AlertTriangle,
  Database,
} from "lucide-react";
import Link from "next/link";

const PAGE_SIZE = 20;

export default function CasesPage() {
  const [page, setPage] = useState(0);
  const [search, setSearch] = useState("");
  const [drFilter, setDrFilter] = useState<number | undefined>(undefined);
  const [referOnly, setReferOnly] = useState(false);

  const { data, isLoading, isFetching } = useQuery({
    queryKey: ["cases", page, drFilter, referOnly],
    queryFn: () =>
      retinaApi.getCases({
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
        dr_grade: drFilter,
        refer_only: referOnly,
      }),
    placeholderData: (prev) => prev,
  });

  const filtered = search
    ? data?.cases.filter(
        (c) =>
          c.patient_id.toLowerCase().includes(search.toLowerCase()) ||
          c.id.toLowerCase().includes(search.toLowerCase())
      )
    : data?.cases;

  return (
    <AppLayout title="Case Database">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-white">Case Database</h2>
            <p className="text-slate-400 text-sm mt-0.5">
              {data?.total ?? 0} total cases stored
            </p>
          </div>
          <Link href="/upload">
            <button className="flex items-center gap-2 px-4 py-2 bg-sky-500 hover:bg-sky-400 text-white rounded-xl text-sm font-medium shadow-lg shadow-sky-500/20 transition-all">
              <Eye className="w-4 h-4" />
              New Analysis
            </button>
          </Link>
        </div>

        <div className="flex items-center gap-3 flex-wrap">
          <div className="relative flex-1 min-w-48">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search by patient ID or case ID..."
              className="w-full pl-10 pr-4 py-2.5 bg-slate-800/60 border border-slate-700/60 rounded-xl text-sm text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-sky-500/50 transition-all"
            />
          </div>

          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-slate-500" />
            <select
              value={drFilter ?? ""}
              onChange={(e) =>
                setDrFilter(e.target.value === "" ? undefined : Number(e.target.value))
              }
              className="px-3 py-2.5 bg-slate-800/60 border border-slate-700/60 rounded-xl text-sm text-slate-300 focus:outline-none focus:border-sky-500/50"
            >
              <option value="">All DR Grades</option>
              <option value="0">Grade 0 — No DR</option>
              <option value="1">Grade 1 — Mild</option>
              <option value="2">Grade 2 — Moderate</option>
              <option value="3">Grade 3 — Severe</option>
              <option value="4">Grade 4 — Proliferative</option>
            </select>

            <label className="flex items-center gap-2 px-3 py-2.5 bg-slate-800/60 border border-slate-700/60 rounded-xl cursor-pointer hover:border-slate-600/60 transition-colors">
              <input
                type="checkbox"
                checked={referOnly}
                onChange={(e) => setReferOnly(e.target.checked)}
                className="accent-rose-500"
              />
              <span className="text-sm text-slate-300 flex items-center gap-1.5">
                <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
                Referrals only
              </span>
            </label>
          </div>
        </div>

        <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl overflow-hidden">
          {isLoading ? (
            <div className="p-16 flex items-center justify-center">
              <div className="w-6 h-6 border-2 border-sky-500 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : !filtered?.length ? (
            <div className="p-16 text-center">
              <Database className="w-12 h-12 text-slate-700 mx-auto mb-4" />
              <p className="text-slate-400 font-medium">No cases found</p>
              <p className="text-slate-600 text-sm mt-1">
                {search || drFilter !== undefined || referOnly
                  ? "Try adjusting your filters"
                  : "Upload a retinal scan to populate the database"}
              </p>
              {!search && !drFilter && !referOnly && (
                <Link href="/upload">
                  <button className="mt-4 px-4 py-2 bg-sky-500 text-white rounded-xl text-sm font-medium hover:bg-sky-400 transition-colors">
                    Upload First Scan
                  </button>
                </Link>
              )}
            </div>
          ) : (
            <>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-800/60">
                      {["Case ID", "Patient ID", "Diagnosis", "Confidence", "Quality", "Date", "Referral", ""].map(
                        (h) => (
                          <th
                            key={h}
                            className="text-left px-5 py-3 text-xs font-medium text-slate-500 whitespace-nowrap"
                          >
                            {h}
                          </th>
                        )
                      )}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/40">
                    {filtered?.map((c, i) => (
                      <motion.tr
                        key={c.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: i * 0.02 }}
                        className="hover:bg-slate-800/30 transition-colors group"
                      >
                        <td className="px-5 py-3.5">
                          <span className="font-mono text-xs text-slate-400">{c.id}</span>
                        </td>
                        <td className="px-5 py-3.5">
                          <span className="text-sm font-medium text-white">{c.patient_id}</span>
                        </td>
                        <td className="px-5 py-3.5">
                          <DiagnosisBadge
                            grade={c.dr_grade}
                            label={getDRGradeLabel(c.dr_grade)}
                            size="sm"
                          />
                        </td>
                        <td className="px-5 py-3.5 w-28">
                          <ConfidenceMeter value={c.dr_confidence} showPercentage />
                        </td>
                        <td className="px-5 py-3.5">
                          <span
                            className={`text-xs font-medium ${
                              c.quality_adequate ? "text-emerald-400" : "text-amber-400"
                            }`}
                          >
                            {(c.quality_score * 100).toFixed(0)}%
                          </span>
                        </td>
                        <td className="px-5 py-3.5 text-slate-400 text-xs whitespace-nowrap">
                          {formatDate(c.created_at)}
                        </td>
                        <td className="px-5 py-3.5">
                          {c.dr_refer ? (
                            <span className="flex items-center gap-1 text-xs text-rose-400 font-medium">
                              <AlertTriangle className="w-3 h-3" />
                              Refer
                            </span>
                          ) : (
                            <span className="text-xs text-slate-600">—</span>
                          )}
                        </td>
                        <td className="px-5 py-3.5">
                          <Link href={`/cases/${c.id}`}>
                            <button className="opacity-0 group-hover:opacity-100 text-slate-500 hover:text-sky-400 transition-all">
                              <ArrowRight className="w-4 h-4" />
                            </button>
                          </Link>
                        </td>
                      </motion.tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="flex items-center justify-between px-5 py-4 border-t border-slate-800/60">
                <p className="text-xs text-slate-500">
                  Showing {page * PAGE_SIZE + 1}–
                  {Math.min((page + 1) * PAGE_SIZE, data?.total ?? 0)} of{" "}
                  {data?.total ?? 0} cases
                  {isFetching && (
                    <span className="ml-2 text-sky-400">refreshing...</span>
                  )}
                </p>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setPage((p) => Math.max(0, p - 1))}
                    disabled={page === 0}
                    className="w-8 h-8 rounded-lg bg-slate-800/60 border border-slate-700/60 flex items-center justify-center text-slate-400 hover:text-slate-200 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                  <span className="text-xs text-slate-500 px-2">Page {page + 1}</span>
                  <button
                    onClick={() => setPage((p) => p + 1)}
                    disabled={(page + 1) * PAGE_SIZE >= (data?.total ?? 0)}
                    className="w-8 h-8 rounded-lg bg-slate-800/60 border border-slate-700/60 flex items-center justify-center text-slate-400 hover:text-slate-200 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </AppLayout>
  );
}
