# Study Note: Open-Vocabulary Object Detection with Grounding DINO

## ১. মূল ধারণা: Traditional YOLO vs. Grounding DINO

| বিষয় | Traditional YOLO | Grounding DINO |
| --- | --- | --- |
| **Class Architecture** | Fixed class list (ট্রেনিংয়ের নির্দিষ্ট চেকলিস্ট বা পয়েন্টে সীমাবদ্ধ)। | Open-vocabulary (জিরো-শট সুবিধাসহ টেক্সট প্রম্পট দিয়ে অবজেক্ট খোঁজা যায়)। |
| **Input Feature** | শুধুমাত্র ইমেজ (Image)। | ইমেজ + টেক্সট প্রম্পট (Image + Text Prompt)। |
| **Adaptability** | নতুন অবজেক্ট টাইপ যুক্ত করতে Retraining আবশ্যক। | রিট্রেনিং ছাড়াই নতুন অবজেক্ট বা কনসেপ্ট খোঁজা সম্ভব। |

---

## ২. Grounding DINO Text Prompting Types

Grounding DINO-তে অবজেক্ট চেনার জন্য সাধারণত ৩ ধরনের টেক্সট প্রম্পট ব্যবহৃত হয়:

1. **Single Category Prompt:** `"dog."`
2. **Multi-category Prompt:** `"dog. person. car."` *(ডট দিয়ে প্রতিটি ক্লাস আলাদা করা হয়)*
3. **Descriptive / Phrase Prompt:** `"white dog lying on the street."`

> 💡 **Concept:** টেক্সটের নির্দিষ্ট অংশের সাথে ছবির নির্দিষ্ট অঞ্চল (Region) মেলানোর প্রক্রিয়াটিকেই **Grounding** বলা হয়।

### Model Output Structure

মডেল থেকে ৩টি মূল বিষয় আউটপুট হিসেবে পাওয়া যায়:

```python
{
    "boxes": ...,   # Object Location: [x1, y1, x2, y2]
    "labels": ...,  # Matched Text Tokens (কোন টেক্সটের সাথে মিলেছে)
    "scores": ...,  # Token-to-Region Matching Scores
}

```

---

## ৩. Dual Thresholding Mechanism

Grounding DINO-তে আউটপুট বক্স ফিল্টার করার জন্য দুটি আলাদা থ্রেশহোল্ড (Threshold) কাজ করে:

```text
                  ┌──────────────────────┐
                  │ candidate box score  │
                  └──────────┬───────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
   Score > box_threshold        Score > text_threshold
   ─────────────────────        ──────────────────────
    Candidate Box সংরক্ষিত হবে    Box-এর Label নির্ধারণ করা হবে

```

| Parameter | দায়িত্ব |
| --- | --- |
| **`box_threshold`** | নির্ধারণ করে কোন ক্যান্ডিডেট বক্সগুলো ফাইনাল আউটপুটে রাখা হবে (যে কোনো টোকেনের সর্বোচ্চ স্কোর এর চেয়ে বেশি হতে হয়)। |
| **`text_threshold`** | রাখা ফিল্টারকৃত বক্সটির লেবেলে কোন কোন টেক্সট টোকেনগুলো অন্তর্ভুক্ত হবে তা ঠিক করে। |

### 🧮 গাণিতিক উদাহরণ:

* **Token Scores:** `white → 0.18`, `dog → 0.61`
* **Threshold Settings:** `box_threshold = 0.35`, `text_threshold = 0.25`

**ফলাফল:**

* **Box Status:** রাখা হবে (কারণ সর্বোচ্চ স্কোর $0.61 > 0.35$)।
* **Label:** শুধুমাত্র **"dog"** হবে (কারণ `white`-এর স্কোর $0.18$, যা `text_threshold` $0.25$-এর নিচে)।

> ⚠️ **Warning:** স্কোর `0.61` মানেই যে মডেলের অনুমান শতভাগ সঠিক বা ৬১% নিখুঁত—এমন সরাসরি সিদ্ধান্ত নেওয়া ভুল। এটি মূলত একটি আপেক্ষিক ম্যাচিং স্কোর (Relative Matching Score)।

---

## ৪. Image Processing & Box Coordinate Conversions

### A. Coordinate Target Size Formatting

PIL Image থেকে সাইজ নেওয়ার সময় ডায়মেনশন অর্ডারের বিষয়ে সতর্ক থাকতে হয়:

```python
# PIL Image size returns -> (width, height)
# Target sizes requires  -> (height, width)
target_sizes = [(image.height, image.width)]

```

> এটি আউটপুট বাউন্ডিং বক্সকে মূল ছবির পিক্সেল স্থানাঙ্কে (Pixel Coordinates) রূপান্তর করতে সাহায্য করে।

### B. Bounding Box Coordinate System

আউটপুট বক্সের ফরম্যাট হলো: `[x1, y1, x2, y2]`

* $(x1, y1)$: বক্সের উপরের-বাম (Top-Left) কোণার পিক্সেল স্থানাঙ্ক।
* $(x2, y2)$: বক্সের নিচের-ডান (Bottom-Right) কোণার পিক্সেল স্থানাঙ্ক।

**Box Dimension Equation:**

* $\text{Width} = x_2 - x_1$
* $\text{Height} = y_2 - y_1$

---

## ৫. Evaluation Dataset Split Strategy

মডেলের সঠিক মূল্যায়ন করার জন্য కাস্টম স্প্লিট (Custom Split) ব্যবহার করা হয়:

* **Calibration Split (২০টি ছবি):** Day 1–4 এর জন্য টেক্সট প্রম্পট এবং থ্রেশহোল্ড টিউন বা ক্যালিব্রেট করার কাজে ব্যবহৃত হয়।
* **Test Split (২০টি ছবি):** Day 5-এর চূড়ান্ত মূল্যায়নের (Final Evaluation) জন্য সংরক্ষিত রাখা হয়।
* **Negative Images:** প্রতি স্প্লিটে নির্বাচিত ৫টি ক্লাসের কোনো অ্যানোটেশন ছাড়া **২টি নেগেটিভ ছবি** রাখা হয়—যাতে মডেল ভুল জায়গায় অবজেক্ট ডিটেক্ট করে কি না (False Positives / Hallucination) তা পরীক্ষা করা যায়।

---

## ৬. Google Colab Execution Strategy (Workflow)

1. **Hardware:** Colab-এ **T4 GPU** এনভায়রনমেন্ট সিলেক্ট করা।
2. **Execution:** মোট ৫টি কোড সেল ক্রমানুসারে রান করা।
3. **Qualitative Output Analysis:** শেষ সেলে একই ছবির ওপর ৪টি ভিন্ন প্রম্পটের আউটপুট পাশাপাশি তুলনা করা:

| Prompt Type | টেস্ট প্রম্পট | পর্যবেক্ষণ তালিকা (Observation) |
| --- | --- | --- |
| **Single Category** | `dog.` | কুকুরের বাউন্ডিং বক্স সঠিক জায়গায় আছে কি না। |
| **Multi-category** | `dog. person. car.` | একাধিক অবজেক্ট দিলে ডিটেকশন কীভাবে পরিবর্তিত হয়। |
| **Description** | `white dog lying on the street.` | বর্ণনা অনুযায়ী নির্দিষ্ট অবজেক্ট অঞ্চল চিনতে পারে কি না। |
| **Negative Object** | `fire hydrant.` | ছবিতে অনুপস্থিত অবজেক্ট চাইলে ভুল (False Positive) বক্স আসে কি না। |

---

# ❓ Q&A Section: Key Insights & Edge Cases

**Q1: আগের fixed-class YOLO ও Grounding DINO-তে class নির্বাচন করার পার্থক্য কী? Open-vocabulary কি সব object detect করার নিশ্চয়তা দেয়?**

* **Class নির্বাচন:** Fixed-class YOLO শুধু ট্রেইনিংয়ের নির্দিষ্ট ক্লাসগুলোই (যেমন COCO ৮০ ক্লাস) ডিটেক্ট করে। কিন্তু Grounding DINO-তে কোনো Retraining ছাড়াই Text Prompt-এর মাধ্যমে যেকোনো শব্দ দিয়ে অবজেক্ট খোঁজা যায়।
* **নিশ্চয়তা:** Open-vocabulary মানেই সব অবজেক্ট শনাক্ত করার গ্যারান্টি নয়। মডেলটি অপ্রচলিত শব্দ, জটিল প্রম্পট বা খুব ছোট অবজেক্টের ক্ষেত্রে ব্যর্থ হতে পারে।

**Q2: Box threshold 0.35, text threshold 0.25; token scores white=0.18, dog=0.61। Box থাকবে কি? Box threshold 0.70 করলে কী হবে?**

* **Box ও Label:** যেহেতু সর্বোচ্চ স্কোর `dog` ($0.61$), যা `box_threshold` ($0.35$)-এর চেয়ে বড়, তাই বক্সটি **থাকবে**। আর `white`-এর স্কোর ($0.18$) থ্রেশহোল্ডের নিচে হওয়ায় লেবেলে শুধু **`dog`** থাকবে।
* **Threshold 0.70 করলে:** বক্স থ্রেশহোল্ড $0.70$ করলে, $0.61$ স্কোরের বক্সটি ফিল্টার আউট (বাদ) হয়ে যাবে।

**Q3: ছবির width 640, height 426। সঠিক target_sizes কী? xyxy format-এ x2, y2 কি width ও height?**

* **Target Sizes:** মডেলের জন্য সঠিক `target_sizes` হবে **`[(426, 640)]`** বা `[(height, width)]`।
* **x2, y2:** `xyxy` ফরম্যাটে $(x_2, y_2)$ বক্সের নিচের-ডান সীমানা নির্দেশ করে; এগুলো সরাসরি Width বা Height নয়। (Width = $x_2 - x_1$, Height = $y_2 - y_1$)।

**Q4: `fire hydrant.` prompt-এ zero boxes পাওয়া কেন program failure নয়? একটি box এলে কী দেখে False Positive বলবেন?**

* **Zero Boxes:** ছবিতে যদি বস্তুটি না থাকে এবং মডেল কোনো বক্স না দেয় ($0$ Box), তবে এটি প্রোগ্রাম ফেইলিয়র নয়, বরং এটি একটি সফল **True Negative (TN)** সিদ্ধান্ত।
* **False Positive:** ছবিতে বস্তুটি না থাকা সত্ত্বেও মডেল যদি কোনো বক্স প্রেডিক্ট করে, তবে সেটিই False Positive।

**Q5: Day 2-এর threshold বেছে নিতে final test ছবিগুলো বারবার দেখা কেন ঠিক নয়?**

* **Data Leakage:** ফাইনাল টেস্ট সেটের ছবিগুলো বারবার দেখে থ্রেশহোল্ড টিউন করলে **Data Leakage (তথ্য ফাঁস)** ঘটে। এতে টেস্ট স্কোর কৃত্রিমভাবে ভালো (Optimistic) হয়ে যায়। এর ফলে ভবিষ্যতে নতুন বা অজানা কোনো ডেটাতে মডেলটি বাস্তবে কেমন পারফর্ম করবে, তা আর সঠিকভাবে অনুমান করা সম্ভব হয় না। (টিউন করার জন্য সবসময় Calibration বা Validation সেট ব্যবহার করতে হয়)।