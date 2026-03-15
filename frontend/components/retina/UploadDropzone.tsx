"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, Image as ImageIcon, X, CheckCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface UploadDropzoneProps {
  onFileSelect: (file: File) => void;
  selectedFile?: File | null;
  onClear?: () => void;
}

export function UploadDropzone({ onFileSelect, selectedFile, onClear }: UploadDropzoneProps) {
  const [preview, setPreview] = useState<string | null>(null);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (!file) return;
      onFileSelect(file);
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target?.result as string);
      reader.readAsDataURL(file);
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"] },
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024,
  });

  const handleClear = () => {
    setPreview(null);
    onClear?.();
  };

  if (selectedFile && preview) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="relative rounded-2xl border border-emerald-500/30 bg-emerald-500/5 overflow-hidden"
      >
        <div className="relative">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={preview}
            alt="Selected retinal scan"
            className="w-full h-72 object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-slate-950/80 via-transparent to-transparent" />
          <div className="absolute bottom-4 left-4 right-4 flex items-end justify-between">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <CheckCircle className="w-4 h-4 text-emerald-400" />
                <span className="text-emerald-400 text-sm font-medium">Image ready</span>
              </div>
              <p className="text-white font-medium text-sm">{selectedFile.name}</p>
              <p className="text-slate-400 text-xs">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
            <button
              onClick={handleClear}
              className="w-8 h-8 rounded-full bg-slate-900/80 border border-slate-700 flex items-center justify-center text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </motion.div>
    );
  }

  return (
    <div
      {...getRootProps()}
      className={cn(
        "relative rounded-2xl border-2 border-dashed p-12 text-center cursor-pointer transition-all duration-300",
        isDragActive
          ? "border-sky-500 bg-sky-500/5 scale-105"
          : "border-slate-700 hover:border-slate-500 hover:bg-slate-800/30"
      )}
    >
      <input {...getInputProps()} />

      <AnimatePresence>
        {isDragActive && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 rounded-2xl bg-sky-500/10 border-2 border-sky-500 flex items-center justify-center"
          >
            <div className="text-center">
              <Upload className="w-12 h-12 text-sky-400 mx-auto mb-2 animate-bounce" />
              <p className="text-sky-400 font-medium">Release to upload</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex flex-col items-center gap-4">
        <div className="w-20 h-20 rounded-2xl bg-slate-800/60 border border-slate-700/60 flex items-center justify-center group-hover:border-sky-500/30 transition-colors">
          <ImageIcon className="w-9 h-9 text-slate-500" />
        </div>

        <div>
          <p className="text-slate-200 font-medium text-lg mb-1">
            Drop your retinal scan here
          </p>
          <p className="text-slate-500 text-sm">
            or{" "}
            <span className="text-sky-400 hover:text-sky-300 underline underline-offset-2 cursor-pointer">
              browse files
            </span>
          </p>
        </div>

        <div className="flex items-center gap-4 text-xs text-slate-600">
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-slate-600" />
            JPG, PNG, BMP, TIFF
          </span>
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-slate-600" />
            Up to 50MB
          </span>
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-slate-600" />
            Fundus images
          </span>
        </div>
      </div>
    </div>
  );
}
