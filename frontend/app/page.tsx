"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import {
  Eye,
  Brain,
  Zap,
  Shield,
  BarChart3,
  FileText,
  ArrowRight,
  CheckCircle,
  Layers,
  Activity,
  Moon,
  Sun,
} from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

const capabilities = [
  {
    icon: Brain,
    title: "DR Grading",
    desc: "5-level diabetic retinopathy classification with confidence scores",
    color: "from-sky-500 to-blue-600",
    glow: "shadow-sky-500/20",
  },
  {
    icon: Eye,
    title: "AMD Detection",
    desc: "Early, intermediate and late AMD staging with AI precision",
    color: "from-violet-500 to-purple-600",
    glow: "shadow-violet-500/20",
  },
  {
    icon: Shield,
    title: "Glaucoma Suspect",
    desc: "Cup-to-disc ratio analysis and glaucoma probability scoring",
    color: "from-emerald-500 to-teal-600",
    glow: "shadow-emerald-500/20",
  },
  {
    icon: Layers,
    title: "Lesion Detection",
    desc: "Microaneurysms, hemorrhages, exudates and IRMA detection",
    color: "from-orange-500 to-amber-600",
    glow: "shadow-orange-500/20",
  },
  {
    icon: Zap,
    title: "Grad-CAM XAI",
    desc: "Visual explanation of AI decisions — which region caused the diagnosis",
    color: "from-pink-500 to-rose-600",
    glow: "shadow-pink-500/20",
  },
  {
    icon: FileText,
    title: "PDF Reports",
    desc: "Professional clinical reports ready for print or EHR integration",
    color: "from-cyan-500 to-sky-600",
    glow: "shadow-cyan-500/20",
  },
];

const stats = [
  { value: "94.2%", label: "AUC Score", sub: "APTOS 2019" },
  { value: "0.81", label: "Kappa Score", sub: "Clinical grade" },
  { value: "<250ms", label: "Inference", sub: "GPU accelerated" },
  { value: "5+", label: "Diseases", sub: "Detected simultaneously" },
];

const benefits = [
  "FDA-standard sensitivity for referable DR detection",
  "Multi-task AI: detect 5+ conditions in one inference",
  "Grad-CAM explainability for clinical trust",
  "DICOM-compatible fundus image support",
  "Automatic case storage and longitudinal tracking",
  "PDF clinical report generation",
];

export default function LandingPage() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  return (
    <div className="min-h-screen bg-slate-950 text-white overflow-hidden">
      <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-8 py-4 border-b border-slate-800/60 bg-slate-950/80 backdrop-blur-xl">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-sky-500 to-blue-600 flex items-center justify-center shadow-lg shadow-sky-500/30">
            <Eye className="w-4 h-4 text-white" />
          </div>
          <span className="font-bold text-white text-lg tracking-tight">RetinaGPT</span>
          <span className="px-2 py-0.5 text-xs font-medium bg-sky-500/10 text-sky-400 border border-sky-500/20 rounded-full">
            v2.0
          </span>
        </div>

        <div className="flex items-center gap-4">
          {mounted && (
            <button
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              className="w-8 h-8 rounded-lg bg-slate-800/60 border border-slate-700/60 flex items-center justify-center text-slate-400 hover:text-white transition-colors"
            >
              {theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
          )}
          <Link
            href="/dashboard"
            className="flex items-center gap-2 px-5 py-2 bg-sky-500 hover:bg-sky-400 text-white rounded-xl font-medium text-sm transition-all shadow-lg shadow-sky-500/25 hover:shadow-sky-500/40 hover:scale-105"
          >
            Open Platform
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </nav>

      <section className="relative pt-32 pb-24 px-8 text-center overflow-hidden">
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[800px] h-[800px] rounded-full bg-sky-500/5 blur-3xl" />
          <div className="absolute top-1/3 left-1/4 w-[400px] h-[400px] rounded-full bg-blue-600/5 blur-3xl" />
          <div className="absolute top-1/3 right-1/4 w-[400px] h-[400px] rounded-full bg-violet-600/5 blur-3xl" />
          <div className="absolute inset-0" style={{
            backgroundImage: `radial-gradient(circle at 1px 1px, rgba(148,163,184,0.04) 1px, transparent 0)`,
            backgroundSize: "40px 40px"
          }} />
        </div>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="relative max-w-5xl mx-auto"
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1 }}
            className="inline-flex items-center gap-2 px-4 py-2 bg-sky-500/10 border border-sky-500/20 rounded-full text-sky-400 text-sm font-medium mb-8"
          >
            <span className="w-2 h-2 rounded-full bg-sky-400 animate-pulse" />
            Production-Grade Clinical AI Platform
          </motion.div>

          <h1 className="text-6xl md:text-8xl font-black mb-6 leading-none tracking-tight">
            <span className="text-white">AI-Powered</span>
            <br />
            <span className="bg-gradient-to-r from-sky-400 via-cyan-400 to-blue-500 bg-clip-text text-transparent">
              Retinal Analysis
            </span>
          </h1>

          <p className="text-slate-400 text-xl md:text-2xl max-w-3xl mx-auto mb-12 leading-relaxed">
            Detect diabetic retinopathy, AMD, and glaucoma with clinical-grade AI.
            Upload a fundus image — get a full diagnosis in under a second.
          </p>

          <div className="flex items-center justify-center gap-4 flex-wrap">
            <Link href="/upload">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.98 }}
                className="flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-sky-500 to-blue-600 text-white rounded-2xl font-semibold text-lg shadow-2xl shadow-sky-500/30 hover:shadow-sky-500/50 transition-all"
              >
                <Eye className="w-5 h-5" />
                Analyze a Retinal Scan
                <ArrowRight className="w-5 h-5" />
              </motion.button>
            </Link>
            <Link href="/dashboard">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.98 }}
                className="flex items-center gap-2 px-8 py-4 bg-slate-800/80 border border-slate-700 text-slate-200 rounded-2xl font-semibold text-lg hover:bg-slate-700/80 transition-all"
              >
                <BarChart3 className="w-5 h-5" />
                View Dashboard
              </motion.button>
            </Link>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="relative max-w-5xl mx-auto mt-16"
        >
          <div className="grid grid-cols-4 gap-4">
            {stats.map((stat, i) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 + i * 0.1 }}
                className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-5 text-center backdrop-blur"
              >
                <div className="text-3xl font-black bg-gradient-to-r from-sky-400 to-blue-500 bg-clip-text text-transparent">
                  {stat.value}
                </div>
                <div className="text-white font-semibold text-sm mt-1">{stat.label}</div>
                <div className="text-slate-500 text-xs mt-0.5">{stat.sub}</div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section>

      <section className="py-24 px-8">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-black text-white mb-4">
              Full AI Ophthalmology Suite
            </h2>
            <p className="text-slate-400 text-lg max-w-2xl mx-auto">
              One platform. Six AI capabilities. Clinical-grade results.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {capabilities.map((cap, i) => (
              <motion.div
                key={cap.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.08 }}
                className="group bg-slate-900/60 border border-slate-700/60 rounded-2xl p-6 hover:border-slate-600/60 transition-all hover:bg-slate-900/80"
              >
                <div
                  className={`w-12 h-12 rounded-xl bg-gradient-to-br ${cap.color} flex items-center justify-center shadow-lg ${cap.glow} mb-4`}
                >
                  <cap.icon className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-lg font-bold text-white mb-2">{cap.title}</h3>
                <p className="text-slate-400 text-sm leading-relaxed">{cap.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <section className="py-24 px-8 bg-slate-900/30">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            <div>
              <h2 className="text-4xl font-black text-white mb-6">
                Built for Clinical Excellence
              </h2>
              <p className="text-slate-400 text-lg mb-8 leading-relaxed">
                RetinaGPT meets the highest standards of clinical AI — from AUC scores
                that surpass published benchmarks to explainable AI that doctors can trust.
              </p>
              <div className="space-y-3">
                {benefits.map((benefit, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: i * 0.07 }}
                    className="flex items-start gap-3"
                  >
                    <CheckCircle className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
                    <span className="text-slate-300 text-sm">{benefit}</span>
                  </motion.div>
                ))}
              </div>

              <div className="mt-10">
                <Link href="/upload">
                  <motion.button
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.98 }}
                    className="flex items-center gap-2 px-6 py-3 bg-sky-500 hover:bg-sky-400 text-white rounded-xl font-semibold shadow-lg shadow-sky-500/25 transition-all"
                  >
                    Start Analysis
                    <ArrowRight className="w-4 h-4" />
                  </motion.button>
                </Link>
              </div>
            </div>

            <div className="relative">
              <div className="relative bg-slate-900 rounded-3xl border border-slate-700/60 overflow-hidden p-4 shadow-2xl">
                <div className="flex items-center gap-1.5 mb-3">
                  <span className="w-3 h-3 rounded-full bg-red-500" />
                  <span className="w-3 h-3 rounded-full bg-yellow-500" />
                  <span className="w-3 h-3 rounded-full bg-emerald-500" />
                  <span className="ml-2 text-xs text-slate-500 font-mono">AI Analysis — RetinaGPT v2.0</span>
                </div>
                <div className="bg-slate-950 rounded-2xl p-5 space-y-3 font-mono text-xs">
                  <div className="text-emerald-400">{"{"}</div>
                  <div className="pl-4 text-slate-300">
                    <span className="text-sky-400">"dr_grading"</span>: {"{"}
                  </div>
                  <div className="pl-8 text-slate-400">
                    <span className="text-violet-400">"grade"</span>: <span className="text-amber-400">2</span>,
                  </div>
                  <div className="pl-8 text-slate-400">
                    <span className="text-violet-400">"label"</span>: <span className="text-emerald-400">"Moderate NPDR"</span>,
                  </div>
                  <div className="pl-8 text-slate-400">
                    <span className="text-violet-400">"confidence"</span>: <span className="text-amber-400">0.874</span>,
                  </div>
                  <div className="pl-8 text-slate-400">
                    <span className="text-violet-400">"refer"</span>: <span className="text-rose-400">true</span>
                  </div>
                  <div className="pl-4 text-slate-300">{"}"}</div>
                  <div className="pl-4 text-slate-300">
                    <span className="text-sky-400">"report"</span>: {"{"}
                  </div>
                  <div className="pl-8 text-slate-400">
                    <span className="text-violet-400">"recommendation"</span>:
                  </div>
                  <div className="pl-12 text-emerald-400">"Ophthalmology referral</div>
                  <div className="pl-13 text-emerald-400 pl-12">within 3 months."</div>
                  <div className="pl-4 text-slate-300">{"}"}</div>
                  <div className="pl-4 text-slate-300">
                    <span className="text-sky-400">"inference_time_ms"</span>: <span className="text-amber-400">187.4</span>
                  </div>
                  <div className="text-emerald-400">{"}"}</div>
                </div>
                <div className="absolute top-4 right-4">
                  <div className="flex items-center gap-1.5 px-2.5 py-1 bg-emerald-500/10 border border-emerald-500/20 rounded-full">
                    <Activity className="w-3 h-3 text-emerald-400 animate-pulse" />
                    <span className="text-emerald-400 text-xs font-medium">Live</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="py-24 px-8 text-center relative overflow-hidden">
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-sky-500/5 blur-3xl" />
        </div>
        <div className="relative max-w-3xl mx-auto">
          <h2 className="text-5xl font-black text-white mb-6">
            Ready to analyze your first scan?
          </h2>
          <p className="text-slate-400 text-xl mb-10">
            Upload a retinal fundus image and get a clinical AI diagnosis in seconds.
          </p>
          <Link href="/upload">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.98 }}
              className="inline-flex items-center gap-3 px-10 py-5 bg-gradient-to-r from-sky-500 to-blue-600 text-white rounded-2xl font-bold text-xl shadow-2xl shadow-sky-500/30 hover:shadow-sky-500/50 transition-all"
            >
              <Eye className="w-6 h-6" />
              Start Free Analysis
              <ArrowRight className="w-6 h-6" />
            </motion.button>
          </Link>
        </div>
      </section>

      <footer className="py-8 px-8 border-t border-slate-800/60 text-center">
        <div className="flex items-center justify-center gap-3 mb-2">
          <div className="w-5 h-5 rounded bg-gradient-to-br from-sky-500 to-blue-600 flex items-center justify-center">
            <Eye className="w-3 h-3 text-white" />
          </div>
          <span className="font-bold text-slate-400">RetinaGPT</span>
        </div>
        <p className="text-slate-600 text-sm">
          AI-Powered Clinical Ophthalmology Platform — For Research & Clinical Use
        </p>
      </footer>
    </div>
  );
}
