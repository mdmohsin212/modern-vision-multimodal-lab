# Study Notes: Grounding DINO + SAM 2.1 Cascade Pipeline Evaluation

## ১. System Architecture & Bottleneck Analysis

এই পাইপলাইনে একটি **Cascade Detection-Segmentation Architecture** ব্যবহার করা হয়েছে:

```
┌────────────────────────┐
│  Image + Text Prompt   │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│     Grounding DINO     │  <-- Detection Step (Finds Named Boxes)
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│   Named Bounding Boxes │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│   SAM 2.1 Box Prompts  │  <-- Promptable Segmentation Step
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ Named Instance Masks   │
└────────────────────────┘

```

### The Upper-Bound Bottleneck

Calibration Dataset-এর ফিক্সড কনফিগারেশনে Detection-এর পরিসংখ্যান:

* **Ground-truth Instances:** $110$
* **Matched Detections:** $40$
* **Missed Detections:** $70$
* **Detection Recall:** $\frac{40}{110} = 0.364 \text{ (or } 36.4\% \text{)}$

> **Critical Limit:** SAM 2.1 একটি জেনারেটিভ বা প্রম্পটেবল সেগমেন্টার, স্বাধীন ডিটেক্টর নয়। এটি কেবল Grounding DINO-এর থেকে পাওয়া বাউন্ডারিং বক্সগুলোকে Prompt হিসেবে গ্রহণ করে। ফলে **Detector যে ৭০টি Object Miss করেছে, SAM 2.1 সেগুলোকে কোনোভাবেই নিজে খুঁজে বের করতে পারবে না।** পুরো সেগমেন্টেশন পাইপলাইনের End-to-End Recall-এর তাত্ত্বিক সর্বোচ্চ সীমা (Theoretical Ceiling) তাই $\mathbf{0.364}$-এ সীমাবদ্ধ।

---

## ২. Dual Evaluation Framework

পাইপলাইনকে নিখুঁতভাবে পরিমাপ করার জন্য মূল্যায়নকে **দুইটি আলাদা গাণিতিক প্রশ্নে** ভাগ করা হয়েছে:

### Question 1: Conditional Mask Quality

> *"Detector যদি কোনো অবজেক্ট সফলভাবে খুঁজে পায়, তবে SAM 2.1 কত ভালো মাস্ক তৈরি করতে পারে?"*

এখানে শুধু **Matched Detections ($40$ টি)** বিবেচনা করা হয় (যেখানে Class ঠিক আছে এবং Box $\text{IoU} \ge 0.50$):

$$\text{Mask IoU} = \frac{\vert{}M_{\text{pred}} \cap M_{\text{gt}}\vert{}}{\vert{}M_{\text{pred}} \cup M_{\text{gt}}\vert{}}$$

$$\text{Dice Coefficient} = \frac{2 \vert{}M_{\text{pred}} \cap M_{\text{gt}}\vert{}}{\vert{}M_{\text{pred}}\vert{} + \vert{}M_{\text{gt}}\vert{}}$$

### Question 2: End-to-End Pipeline Performance

> *"সমগ্র ডেটাসেটের মোট অবজেক্টের কতগুলোকে পাইপলাইন সফলভাবে সঠিক মাস্ক দিতে পেরেছে?"*

এখানে ডেটাসেটের **সবগুলো ($110$ টি) Ground-Truth Object** হর (Denominator) হিসেবে থাকবে:

$$\text{End-to-End Mask Recall@0.50} = \frac{\text{Class-correct masks with IoU} \ge 0.50}{110}$$

* **গাণিতিক তাৎপর্য:** Detector-এর ৭০টি মিস এখানে ব্যর্থতা হিসেবে গণনায় আসবে। ফলে ভালো SAM Mask তৈরি করে ডিটেক্টরের দুর্বলতা লুকিয়ে রাখা যাবে না।

---

## ৩. Implementation Details & Alignment Controls

### GT Mask Construction & Inclusive Boundary Correctness

Ground Truth Polygon থেকে Binary Mask এবং $\text{XYXY}$ Box তৈরিতে `+1` কোঅর্ডিনেট অফসেট ব্যবহার করা হয়:

```
Normalized Polygon ──► Original Pixel Coordinates ──► Binary GT Mask ──► Evaluation Box/Mask

```

`xs.max()` হলো মাস্কের শেষ Foreground পিক্সেল। কিন্তু Ultralytics SAM 2.1 pixel-coordinate $\text{XYXY}$ বক্সের শেষ coordinate-কে **exclusive boundary** হিসেবে ধরে, তাই বর্ডার পিক্সেল মিস না হওয়ার জন্য `+1` যোগ করা হয়।

### Index Array Alignment & Prompt Batching

1. **Parallel Array Integrity:** Grounding DINO থেকে আউটপুট পাওয়া `boxes`, `labels`, `scores`, এবং প্রাপ্ত `masks`-এর Index Alignment অপরিবর্তিত রাখা জরুরি:
* `Index 0`: Dog Box $\rightarrow$ Dog Mask
* `Index 1`: Person Box $\rightarrow$ Person Mask
* `Index 2`: Car Box $\rightarrow$ Car Mask
* *ঝুঁকি:* যদিBoxes সাজানো (sort) হয় কিন্তু Labels ও Scores সেভাবে না বদলানো হয়, তবে Class Miss-match ঘটবে (যেমন: Dog Mask-এ Car লেবেল বসে যাবে)।


2. **Single Mask Output Strategy:** `multimask_output=False` সেট করায় প্রতি বক্স প্রম্পটের জন্য SAM 2.1 একটি সেরা Ambiguity-free মাস্ক প্রদান করে।
3. **Inference Efficiency (Feature Reuse):** একটি ইমেজের সব প্রম্পট বক্স একসাথে ইনপুট দিলে ব্যাকবোন ভিজিয়ন এনকোডার ইমেজ ফিচার **একবারই** তৈরি করে। এতে ১০টি বক্স থাকলেও ১০ বার ইমেজ প্রসেস হয় না, ফলে মেমোরি ও ল্যাটেন্সি অপ্টিমাইজড থাকে।

---

## ৪. Result Interpretation Framework

### Box IoU vs. Mask IoU Scenarios

| Box IoU | Mask IoU | ব্যাখ্যার সারসংক্ষেপ |
| --- | --- | --- |
| **$0.58$** | **$0.82$** | **Boundary Refinement Success:** ডিটেক্টরের বক্স খুব নিখুঁত ছিল না, কিন্তু অবজেক্টের অংশ কভার করেছিল। SAM 2.1 সেই দুর্বল বক্স থেকে অবজেক্টের বাউন্ডারি খুব সুন্দরভাবে উদ্ধার করেছে। |
| **$0.70$** | **$0.31$** | **Segmentation Failure:** বক্স IoU যথেষ্ট ভালো থাকা সত্ত্বেও SAM 2.1 ভুল রিজিয়ন, অবজেক্টের আংশিক অংশ, অথবা পাশের অন্য অবজেক্ট সেগমেন্ট করে ফেলেছে। |

### Diagnostic Metrics Relationship

যদি কোনো আউটপুটে পাওয়া যায়:

* `conditional_mask_success_rate = 0.90` (৯০%)
* `end_to_end_mask_recall_iou50 = 0.33` (৩৩%)

**সিদ্ধান্ত:**

1. SAM 2.1-এর সেগমেন্টেশন ক্ষমতা অত্যন্ত চমৎকার (ডিটেক্ট করতে পারলে ৯০% ক্ষেত্রে ভালো মাস্ক দেয়)।
2. পুরো পাইপলাইনের মূল Bottleneck হলো **Grounding DINO-এর Detection Misses**, SAM 2.1 নিজে নয়।
3. `mean_mask_iou_on_matched_detections` কেবল সফল ডিটেকশনের গড় মাস্ক কোয়ালিটি দেখায়, এটিকে পুরো ডেটাসেটের ওভারঅল কোয়ালিটি হিসেবে ব্যাখ্যা করা ভুল।

---

## ৫. Evaluation Diagnostics Rules

1. **Negative Image Inspection:** Negative ইমেজগুলোতে (যেখানে নির্বাচিত ৫টি ক্লাসের কোনো GT নেই) যদি কোনো Mask ডিটেক্ট হয়, তবে তা Inspection Candidate হিসেবে চিহ্নিত করতে হবে।
2. **Annotation Incompleteness Warning:** বাস্তব ডেটাসেটে অ্যানোটেশন অসম্পূর্ণ থাকতে পারে (যেমন: দূরের কোনো ছোট গাড়ি বাদ পড়া)। তাই কেবল টেবিলের **False Positive** সংখ্যা দেখেই চূড়ান্ত সিদ্ধান্ত না নিয়ে ভিজ্যুয়াল ইনস্পেকশন করা জরুরি।