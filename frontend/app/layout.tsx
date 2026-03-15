import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Providers } from "@/components/layout/Providers";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "RetinaGPT — AI Ophthalmology Platform",
  description:
    "Production-grade AI platform for retinal fundus image analysis. Detect diabetic retinopathy, AMD, glaucoma, and more with clinical-grade AI.",
  keywords: ["AI", "ophthalmology", "retina", "diabetic retinopathy", "medical imaging"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.variable} font-sans`}>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
