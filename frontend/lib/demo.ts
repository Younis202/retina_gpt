import type { FullAnalysisResponse } from "@/types";

function generateGradCAMOnCanvas(imageFile: File): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image();
    const url = URL.createObjectURL(imageFile);
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = 224;
      canvas.height = 224;
      const ctx = canvas.getContext("2d")!;

      ctx.drawImage(img, 0, 0, 224, 224);

      const imageData = ctx.getImageData(0, 0, 224, 224);
      const data = imageData.data;
      for (let i = 0; i < data.length; i += 4) {
        const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        data[i] = gray;
        data[i + 1] = gray;
        data[i + 2] = gray;
      }
      ctx.putImageData(imageData, 0, 0);

      const gradient = ctx.createRadialGradient(112, 100, 10, 112, 100, 90);
      gradient.addColorStop(0, "rgba(255, 0, 0, 0.72)");
      gradient.addColorStop(0.3, "rgba(255, 165, 0, 0.55)");
      gradient.addColorStop(0.6, "rgba(255, 255, 0, 0.35)");
      gradient.addColorStop(1, "rgba(0, 0, 255, 0.0)");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, 224, 224);

      const gradient2 = ctx.createRadialGradient(155, 140, 5, 155, 140, 45);
      gradient2.addColorStop(0, "rgba(255, 80, 0, 0.6)");
      gradient2.addColorStop(0.5, "rgba(255, 200, 0, 0.3)");
      gradient2.addColorStop(1, "rgba(0, 0, 0, 0.0)");
      ctx.fillStyle = gradient2;
      ctx.fillRect(0, 0, 224, 224);

      const gradient3 = ctx.createRadialGradient(75, 145, 3, 75, 145, 35);
      gradient3.addColorStop(0, "rgba(255, 120, 0, 0.5)");
      gradient3.addColorStop(1, "rgba(0, 0, 0, 0.0)");
      ctx.fillStyle = gradient3;
      ctx.fillRect(0, 0, 224, 224);

      URL.revokeObjectURL(url);
      resolve(canvas.toDataURL("image/png").split(",")[1]);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      resolve("");
    };
    img.src = url;
  });
}

export async function generateDemoResult(
  file: File
): Promise<FullAnalysisResponse> {
  const gradcamB64 = await generateGradCAMOnCanvas(file);

  const drGrade = 2;
  const scenarios = [
    {
      dr: { grade: 2, label: "Moderate NPDR", confidence: 0.87, probabilities: [0.04, 0.06, 0.87, 0.02, 0.01], refer: true },
      amd: { stage: 1, label: "Early AMD", confidence: 0.76 },
      glaucoma: { suspect: false, cup_disc_ratio: 0.42, confidence: 0.91 },
      lesions: {
        microaneurysms: { present: true, probability: 0.88 },
        hemorrhages: { present: true, probability: 0.72 },
        hard_exudates: { present: false, probability: 0.18 },
        soft_exudates: { present: false, probability: 0.09 },
        neovascularization: { present: false, probability: 0.03 },
      },
      report: {
        structured_findings:
          "Bilateral moderate non-proliferative diabetic retinopathy (NPDR). Multiple microaneurysms and dot-blot hemorrhages identified in the posterior pole. No neovascularization detected. Early drusen deposits consistent with early AMD in the temporal macula. Cup-to-disc ratio within normal limits at 0.42.",
        recommendation:
          "Refer to ophthalmology within 3 months. Optimize systemic glycemic control (HbA1c target <7%). Blood pressure management recommended. Repeat fundus photography in 6 months post-treatment. Patient education on diabetic eye disease progression.",
        full_text:
          "RETINAL FUNDUS ANALYSIS REPORT\n\n" +
          "FINDINGS:\n" +
          "- Diabetic Retinopathy Grade: 2 / Moderate NPDR (Confidence: 87%)\n" +
          "- AMD Stage: Early (Confidence: 76%)\n" +
          "- Glaucoma Risk: Low (C/D ratio: 0.42)\n\n" +
          "LESION SUMMARY:\n" +
          "- Microaneurysms: PRESENT (88%)\n" +
          "- Hemorrhages: PRESENT (72%)\n" +
          "- Hard Exudates: Not detected\n" +
          "- Soft Exudates: Not detected\n" +
          "- Neovascularization: Not detected\n\n" +
          "RECOMMENDATION:\n" +
          "Urgent ophthalmology referral within 3 months. Optimize glycemic and blood pressure control. Repeat imaging in 6 months.",
      },
    },
  ];

  const s = scenarios[0];

  await new Promise((r) => setTimeout(r, 200));

  return {
    request_id: `demo-${Date.now()}`,
    image_id: `demo-img-${Math.random().toString(36).slice(2, 10)}`,
    quality: { score: 0.91, adequate: true },
    dr_grading: s.dr,
    amd: s.amd,
    glaucoma: s.glaucoma,
    lesions: s.lesions,
    report: s.report,
    explainability: {
      gradcam_image: gradcamB64 || null,
      attention_image: null,
      explanation_panel: null,
    },
    segmentation: {
      vessel_mask: null,
      optic_disc_mask: null,
    },
    inference_time_ms: 342 + Math.floor(Math.random() * 80),
    model_version: "retina-gpt-base-v1.0-demo",
  };
}
