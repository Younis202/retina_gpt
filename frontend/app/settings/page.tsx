"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { AppLayout } from "@/components/layout/AppLayout";
import retinaApi from "@/lib/api";
import { formatUptime } from "@/lib/utils";
import { useTheme } from "next-themes";
import toast from "react-hot-toast";
import {
  Settings,
  User,
  Server,
  Shield,
  Moon,
  Sun,
  CheckCircle,
  XCircle,
  RefreshCw,
  Cpu,
  Clock,
  Activity,
} from "lucide-react";

export default function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const [profile, setProfile] = useState({
    name: "Dr. Ahmed",
    specialty: "Ophthalmologist",
    institution: "Medical Center",
    email: "dr.ahmed@clinic.com",
  });

  const { data: health, refetch, isRefetching } = useQuery({
    queryKey: ["health"],
    queryFn: retinaApi.health,
    refetchInterval: 10000,
  });

  const { data: modelInfo } = useQuery({
    queryKey: ["model-info"],
    queryFn: retinaApi.modelInfo,
    retry: false,
  });

  const handleSaveProfile = () => {
    toast.success("Profile saved successfully");
  };

  return (
    <AppLayout title="Settings">
      <div className="max-w-4xl mx-auto space-y-6">
        <div>
          <h2 className="text-2xl font-bold text-white mb-1">Settings</h2>
          <p className="text-slate-400 text-sm">Manage your profile and system configuration</p>
        </div>

        <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-5">
            <div className="w-8 h-8 rounded-xl bg-sky-500/10 flex items-center justify-center">
              <User className="w-4 h-4 text-sky-400" />
            </div>
            <h3 className="text-base font-semibold text-white">Doctor Profile</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[
              { key: "name", label: "Full Name", placeholder: "Dr. Name" },
              { key: "specialty", label: "Specialty", placeholder: "Ophthalmologist" },
              { key: "institution", label: "Institution", placeholder: "Medical Center" },
              { key: "email", label: "Email", placeholder: "dr@clinic.com" },
            ].map(({ key, label, placeholder }) => (
              <div key={key}>
                <label className="block text-xs font-medium text-slate-400 mb-1.5">{label}</label>
                <input
                  value={profile[key as keyof typeof profile]}
                  onChange={(e) => setProfile((p) => ({ ...p, [key]: e.target.value }))}
                  placeholder={placeholder}
                  className="w-full px-3 py-2.5 bg-slate-800/60 border border-slate-700/60 rounded-xl text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-sky-500/50 transition-all"
                />
              </div>
            ))}
          </div>
          <div className="mt-4 pt-4 border-t border-slate-800/60 flex justify-end">
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleSaveProfile}
              className="px-4 py-2 bg-sky-500 hover:bg-sky-400 text-white rounded-xl text-sm font-medium shadow-lg shadow-sky-500/20 transition-all"
            >
              Save Profile
            </motion.button>
          </div>
        </div>

        <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-5">
            <div className="w-8 h-8 rounded-xl bg-violet-500/10 flex items-center justify-center">
              <Settings className="w-4 h-4 text-violet-400" />
            </div>
            <h3 className="text-base font-semibold text-white">Appearance</h3>
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-200">Theme</p>
              <p className="text-xs text-slate-500 mt-0.5">Choose your preferred color theme</p>
            </div>
            <div className="flex items-center gap-2 p-1 bg-slate-800/60 rounded-xl border border-slate-700/60">
              {["light", "dark"].map((t) => (
                <button
                  key={t}
                  onClick={() => setTheme(t)}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                    theme === t
                      ? "bg-sky-500 text-white"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  {t === "light" ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
                  {t.charAt(0).toUpperCase() + t.slice(1)}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-6">
          <div className="flex items-center justify-between mb-5">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-xl bg-emerald-500/10 flex items-center justify-center">
                <Server className="w-4 h-4 text-emerald-400" />
              </div>
              <h3 className="text-base font-semibold text-white">System Status</h3>
            </div>
            <button
              onClick={() => refetch()}
              disabled={isRefetching}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-slate-400 hover:text-slate-200 bg-slate-800/60 border border-slate-700/60 rounded-lg transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isRefetching ? "animate-spin" : ""}`} />
              Refresh
            </button>
          </div>

          {health ? (
            <div className="space-y-4">
              <div className="flex items-center gap-3 p-4 bg-slate-800/40 rounded-xl border border-slate-700/40">
                <div className={`w-3 h-3 rounded-full ${health.status === "healthy" ? "bg-emerald-400 animate-pulse" : "bg-amber-400"}`} />
                <div className="flex-1">
                  <p className="text-sm font-medium text-white capitalize">{health.status}</p>
                  <p className="text-xs text-slate-500">API Backend</p>
                </div>
                {health.model_loaded ? (
                  <CheckCircle className="w-5 h-5 text-emerald-400" />
                ) : (
                  <XCircle className="w-5 h-5 text-amber-400" />
                )}
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[
                  { icon: Cpu, label: "Device", value: health.device, color: "text-sky-400" },
                  { icon: Shield, label: "Version", value: health.version, color: "text-violet-400" },
                  { icon: Clock, label: "Uptime", value: formatUptime(health.uptime_seconds), color: "text-emerald-400" },
                  { icon: Activity, label: "AI Model", value: health.model_loaded ? "Loaded" : "Demo", color: health.model_loaded ? "text-emerald-400" : "text-amber-400" },
                ].map(({ icon: Icon, label, value, color }) => (
                  <div key={label} className="bg-slate-800/40 rounded-xl p-3 border border-slate-700/40">
                    <div className="flex items-center gap-1.5 mb-2">
                      <Icon className={`w-3.5 h-3.5 ${color}`} />
                      <span className="text-xs text-slate-500">{label}</span>
                    </div>
                    <p className="text-sm font-medium text-white">{value}</p>
                  </div>
                ))}
              </div>

              {health.capabilities && (
                <div>
                  <p className="text-xs font-medium text-slate-400 mb-2">Capabilities</p>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                    {Object.entries(health.capabilities).map(([cap, enabled]) => (
                      <div
                        key={cap}
                        className={`flex items-center gap-2 px-3 py-2 rounded-xl border text-xs font-medium ${
                          enabled
                            ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-300"
                            : "bg-slate-800/40 border-slate-700/40 text-slate-500"
                        }`}
                      >
                        {enabled ? <CheckCircle className="w-3.5 h-3.5" /> : <XCircle className="w-3.5 h-3.5" />}
                        {cap.replace(/_/g, " ")}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/20 rounded-xl">
              <XCircle className="w-5 h-5 text-red-400" />
              <div>
                <p className="text-sm font-medium text-red-300">Backend Unreachable</p>
                <p className="text-xs text-red-400/70 mt-0.5">
                  Make sure the FastAPI backend is running on port 8000
                </p>
              </div>
            </div>
          )}
        </div>

        {modelInfo && (
          <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-8 h-8 rounded-xl bg-amber-500/10 flex items-center justify-center">
                <Cpu className="w-4 h-4 text-amber-400" />
              </div>
              <h3 className="text-base font-semibold text-white">Model Information</h3>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {Object.entries(modelInfo).filter(([k]) => k !== "capabilities").map(([key, val]) => (
                <div key={key} className="bg-slate-800/40 rounded-xl p-3 border border-slate-700/40">
                  <p className="text-xs text-slate-500 mb-1 capitalize">{key.replace(/_/g, " ")}</p>
                  <p className="text-sm font-medium text-white">{String(val)}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </AppLayout>
  );
}
