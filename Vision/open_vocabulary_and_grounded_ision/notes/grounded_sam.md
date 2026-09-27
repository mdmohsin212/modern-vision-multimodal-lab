# Study Note: Grounding DINO + SAM 2.1 Cascade Pipeline Evaluation

## ১. System Architecture & Bottleneck Analysis

এই সেগমেন্টেশন পাইপলাইনে মূলত একটি **Cascade (Detection $\rightarrow$ Segmentation) Architecture** ব্যবহার করা হয়েছে:

```text
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

### The Upper-Bound Bottleneck (সর্বোচ্চ সীমার সীমাবদ্ধতা)

Calibration Dataset-এর ফিক্সড কনফিগারেশনে Detection-এর পরিসংখ্যান:

* **Ground-truth Instances:** $110$
* **Matched Detections:** $40$
* **Missed Detections:** $70$
* **Detection Recall:** $\frac{40}{110} = 0.364 \text{ (বা } 36.4\% \text{)}$

> ⚠️ **Critical Limit:** SAM 2.1 একটি জেনারেটিভ বা প্রম্পটেবল সেগমেন্টার; এটি স্বাধীন ডিটেক্টর নয়। এটি কেবল Grounding DINO-এর দেওয়া বাউন্ডিং বক্সগুলোকে প্রম্পট হিসেবে গ্রহণ করে। ফলে **Detector যে ৭০টি অবজেক্ট মিস করেছে, SAM 2.1 নিজে থেকে সেগুলোকে খুঁজে বের করতে পারবে না।** তাই পুরো পাইপলাইনের 'End-to-End Recall'-এর তাত্ত্বিক সর্বোচ্চ সীমা (Theoretical Ceiling) $\mathbf{0.364}$-এ সীমাবদ্ধ।

---

## ২. Dual Evaluation Framework

পুরো পাইপলাইনকে নিখুঁতভাবে পরিমাপ করার জন্য ইভালুয়েশনকে **দুটি আলাদা গাণিতিক ধাপে** ভাগ করা হয়েছে:

### Question 1: Conditional Mask Quality

> *"Detector যদি কোনো অবজেক্ট সফলভাবে খুঁজে পায়, তবে SAM 2.1 কত ভালো মাস্ক তৈরি করতে পারে?"*

এখানে শুধু **Matched Detections (৪০টি)** বিবেচনা করা হয় (যেখানে Class ঠিক আছে এবং Box $\text{IoU} \ge 0.50$):

$$\text{Mask IoU} = \frac{\vert{}M_{\text{pred}} \cap M_{\text{gt}}\vert{}}{\vert{}M_{\text{pred}} \cup M_{\text{gt}}\vert{}}$$

$$\text{Dice Coefficient} = \frac{2 \vert{}M_{\text{pred}} \cap M_{\text{gt}}\vert{}}{\vert{}M_{\text{pred}}\vert{} + \vert{}M_{\text{gt}}\vert{}}$$

### Question 2: End-to-End Pipeline Performance

> *"সমগ্র ডেটাসেটের মোট অবজেক্টের কতগুলোকে পাইপলাইন সফলভাবে সঠিক মাস্ক দিতে পেরেছে?"*

এখানে ডেটাসেটের **সবগুলো (১১০টি) Ground-Truth Object** হর (Denominator) হিসেবে থাকবে:

$$\text{End-to-End Mask Recall@0.50} = \frac{\text{Class-correct masks with IoU} \ge 0.50}{110}$$

* **গাণিতিক তাৎপর্য:** Detector-এর মিস হয়ে যাওয়া ৭০টি অবজেক্ট এখানে ব্যর্থতা হিসেবে গণনায় আসবে। এর ফলে SAM 2.1-এর তৈরি ভালো মাস্ক দিয়ে ডিটেক্টরের মূল দুর্বলতা লুকিয়ে রাখা যায় না।

---

## ৩. Implementation Details & Alignment Controls

### A. GT Mask Construction & Inclusive Boundary Correctness

Ground Truth (GT) Polygon থেকে Binary Mask এবং $\text{XYXY}$ Box তৈরির ফ্লো:

```text
Normalized Polygon ──► Original Pixel Coordinates ──► Binary GT Mask ──► Evaluation Box/Mask

```

`xs.max()` হলো মাস্কের শেষ Foreground পিক্সেল। কিন্তু Ultralytics SAM 2.1 pixel-coordinate $\text{XYXY}$ বক্সের শেষ কোঅর্ডিনেটকে **Exclusive Boundary** হিসেবে ধরে। তাই কোনো বর্ডার পিক্সেল যেন মিস না হয়, সেজন্য কোঅর্ডিনেট অফসেটে `+1` যোগ করা হয়।

### B. Index Array Alignment & Prompt Batching (সতর্কতা)

1. **Parallel Array Integrity:** Grounding DINO থেকে আউটপুট পাওয়া `boxes`, `labels`, `scores`, এবং প্রাপ্ত `masks`-এর ইনডেক্স ক্রম (Index Alignment) অপরিবর্তিত রাখা বাধ্যতামূলক।
* `Index 0`: Dog Box $\rightarrow$ Dog Mask
* `Index 1`: Person Box $\rightarrow$ Person Mask
* *ঝুঁকি:* যদি বক্সগুলো সর্ট (sort) করা হয় কিন্তু লেবেল ও স্কোর সেভাবে পরিবর্তন না করা হয়, তবে **Class Miss-match** ঘটবে (যেমন: কুকুরের মাস্কে গাড়ির লেবেল বসে যাবে)।


2. **Single Mask Output Strategy:** `multimask_output=False` সেট করায়, প্রতিটি বক্স প্রম্পটের জন্য SAM 2.1 একটি সেরা ও দ্ব্যর্থহীন (Ambiguity-free) মাস্ক প্রদান করে।
3. **Inference Efficiency (Feature Reuse):** একটি ইমেজের সব প্রম্পট বক্স একসাথে ইনপুট দিলে ব্যাকবোন ভিশন এনকোডার ইমেজ ফিচার **একবারই** তৈরি করে। এতে ১০টি বক্স থাকলেও ১০ বার ইমেজ প্রসেস হয় না, ফলে ল্যাটেন্সি ও মেমোরি ব্যাপকভাবে সাশ্রয় হয়।

---

## ৪. Result Interpretation Framework

### Box IoU vs. Mask IoU Scenarios

| Box IoU | Mask IoU | ব্যাখ্যার সারসংক্ষেপ |
| --- | --- | --- |
| **$0.58$** | **$0.82$** | **Boundary Refinement Success:** ডিটেক্টরের বক্স খুব নিখুঁত ছিল না, কিন্তু অবজেক্টটিকে কভার করেছিল। SAM 2.1 সেই দুর্বল বক্স থেকেই অবজেক্টের বাউন্ডারি খুব নিখুঁতভাবে উদ্ধার করেছে। |
| **$0.70$** | **$0.31$** | **Segmentation Failure:** বক্স IoU যথেষ্ট ভালো থাকা সত্ত্বেও SAM 2.1 ভুল রিজিয়ন, অবজেক্টের আংশিক অংশ, অথবা পেছনের অন্য কোনো অবজেক্ট সেগমেন্ট করে ফেলেছে। |

### Diagnostic Metrics Relationship

ধরা যাক, কোনো পাইপলাইন আউটপুটে পাওয়া গেল:

* `conditional_mask_success_rate = 0.90` (৯০%)
* `end_to_end_mask_recall_iou50 = 0.33` (৩৩%)

**এর চূড়ান্ত সিদ্ধান্ত হলো:**

1. SAM 2.1-এর সেগমেন্টেশন ক্ষমতা চমৎকার (ডিটেক্ট করতে পারলে ৯০% ক্ষেত্রেই ভালো মাস্ক দেয়)।
2. পুরো পাইপলাইনের মূল Bottleneck (বাধা) হলো **Grounding DINO-এর Detection Miss**, SAM 2.1 নিজে নয়।
3. `mean_mask_iou_on_matched_detections` কেবল সফল ডিটেকশনগুলোর গড় মাস্ক কোয়ালিটি নির্দেশ করে; এটিকে পুরো ডেটাসেটের ওভারঅল পারফরম্যান্স হিসেবে দাবি করা ভুল।

---

## ৫. Evaluation Diagnostics Rules

1. **Negative Image Inspection:** Negative ইমেজগুলোতে (যেখানে নির্বাচিত ৫টি ক্লাসের কোনো গ্রাউন্ড-ট্রুথ নেই) যদি কোনো Mask প্রেডিক্ট হয়, তবে তা অবশ্যই ভিজ্যুয়াল ইনস্পেকশন (Inspection Candidate) করতে হবে।
2. **Annotation Incompleteness Warning:** বাস্তব ডেটাসেটে অ্যানোটেশন অসম্পূর্ণ থাকতে পারে (যেমন: দূরের কোনো ছোট গাড়ি বাদ পড়া)। তাই কেবল টেবিলের **False Positive (FP)** সংখ্যা দেখেই মডেল খারাপ—এমন চূড়ান্ত সিদ্ধান্ত না নিয়ে, ছবি দেখে যাচাই করা জরুরি।

---

## ৬. Q&A and Edge Cases (সচরাচর জিজ্ঞাসিত প্রশ্ন)

**Q1: Grounding DINO কোনো object সম্পূর্ণ miss করলে SAM 2.1 কেন সেটি recover করতে পারবে না?**

* **উত্তর:** SAM 2.1 নিজে থেকে কোনো অবজেক্ট খোঁজে না; সে প্রম্পটের (Bounding Box) ওপর নির্ভরশীল। Grounding DINO অবজেক্ট মিস করলে প্রম্পট তৈরি হয় না, আর প্রম্পট ছাড়া SAM 2.1-এর পক্ষে সেই অদৃশ্য অবজেক্ট সেগমেন্ট করা অসম্ভব।

**Q2: Matched detections-এর mean mask IoU 0.85, কিন্তু end-to-end mask recall 0.30 হতে পারে কীভাবে?**

* **উত্তর:** ডিটেক্টর যদি বেশিরভাগ অবজেক্ট মিস করে তবে এটি ঘটে। যে কয়েকটি অবজেক্ট ডিটেক্ট হয়েছে সেগুলোর মাস্ক নিখুঁত (IoU 0.85) হলেও, মূল অবজেক্টের সিংহভাগ (৭০%) ডিটেকশনে বাদ পড়ে যাওয়ায় সামগ্রিক End-to-End Mask Recall অনেক কম (0.30) হয়।

**Q3: ১১০টি ground-truth object-এর মধ্যে ৪০টি detection match হয়েছে। End-to-end mask recall-এর সর্বোচ্চ সম্ভাব্য মান কত?**

* **উত্তর:** যেহেতু মাত্র ৪০টি অবজেক্ট মাস্ক তৈরির সুযোগ পেয়েছে, তাই এই ৪০টি মাস্কই যদি সফল (IoU $\ge 0.50$) হয়, তবে সর্বোচ্চ সফল মাস্ক হবে ৪০টি।
সুতরাং সর্বোচ্চ মান: $\frac{40}{110} \approx \mathbf{0.3636} \text{ (বা } 36.36\% \text{)}$।

**Q4: একটি prediction-এর class সঠিক, box IoU 0.68, কিন্তু mask IoU 0.32। Detection evaluation ও final mask evaluation-এ এর ফলাফল কী হবে?**

* **উত্তর:**
* **Detection Evaluation:** বক্স IoU ($0.68 \ge 0.50$) এবং ক্লাস সঠিক হওয়ায় এটি নিশ্চিতভাবে **True Positive (TP)**।
* **Final Mask Evaluation:** মাস্ক IoU ($0.32 < 0.50$) কম হওয়ায় এটি সফল মাস্কের তালিকায় (`masks_iou50`) যোগ হবে না।
* **Instance Segmentation Metrics:** পূর্ণাঙ্গ Instance Segmentation ইভ্যালুয়েশনে (যেখানে IoU $\ge 0.50$ ব্যবহৃত হয়) প্রেডিক্টেড মাস্কটি **False Positive (FP)** এবং সংশ্লিষ্ট গ্রাউন্ড-ট্রুথ মাস্কটি **False Negative (FN)** হিসেবে গণ্য হবে।



**Q5: Boxes sort করার পর labels, scores ও masks একই permutation-এ sort না করলে কী সমস্যা হবে?**

* **উত্তর:** অ্যারে বা টেনসরগুলোর মধ্যে Index Misalignment তৈরি হবে। এর ফলে এক অবজেক্টের বক্সের সাথে অন্য অবজেক্টের ক্লাস লেবেল বা মাস্ক জুড়ে যাবে (যেমন: মানুষের মাস্কে গাড়ির স্কোর বা লেবেল)। এটি পুরো ইভ্যালুয়েশন লজিককে ধ্বংস করে দেবে।

**Q6: SAM 2.1 mask কি Grounding DINO box-এর বাইরে যেতে পারে? গেলে সেটি কি সবসময় ভুল?**

* **উত্তর:** হ্যাঁ, মাস্ক বক্সের বাইরে যেতে পারে এবং এটি সবসময় ভুল নয়। অনেক সময় ডিটেকশন বক্স খুব টাইট হলে অবজেক্টের সূক্ষ্ম অংশ (যেমন: পশুর লেজ বা অ্যান্টেনা) বক্সের বাইরে পড়ে যায়। SAM 2.1 অবজেক্টের বাস্তব সীমানা (Contour) অনুসরণ করে বলে এটি সঠিকভাবে সেই বাইরের অংশকেও সেগমেন্টেশনের আওতায় নিয়ে আসতে পারে।