# Study Note: Optimizing Grounding DINO Prompting Strategy (Combined vs. Separate)

## ১. Context & Objectives

**পূর্ববর্তী ফলাফল ও সমস্যা (The Bottleneck):**

* মোট Ground Truth (GT) অবজেক্ট: **১১০টি**
* Detection Match: মাত্র **৪০টি**
* সফল SAM Mask: **৩৯টি** (৪০টির মধ্যে)
* `cup` ক্লাসের Detection Recall: মাত্র **২/২৩**। এছাড়া `cup car`, `dog person`-এর মতো অস্পষ্ট (Ambiguous) লেবেল পাওয়া গেছে।

**মূল প্রশ্ন (Core Objective):**
ডিটেকশনে অনেক অবজেক্ট মিস হওয়ার কারণ কি এক প্রম্পটে অনেকগুলো ক্লাস একসাথে দেওয়া? পাঁচটি ক্লাস একসাথে প্রম্পট করার বদলে আলাদা আলাদা (Separate) প্রম্পট দিলে কি Detection এবং Final Mask Recall উন্নত হয়? এবং এর জন্য ল্যাটেন্সি (অতিরিক্ত সময়) কতটা বাড়ে?

---

## ২. Experimental Setup: Baseline vs. Challenger

এই পরীক্ষায় শুধু **Prompt Strategy** পরিবর্তন করা হবে। কোনো নতুন ফিল্টারিং বা NMS (Non-Maximum Suppression) যোগ করা হবে না।

| প্যারামিটার | Baseline (`combined` mode) | Challenger (`separate` mode) |
| --- | --- | --- |
| **Prompt Format** | `chair. cup. dog. car. person.` | `chair.` $\rightarrow$ `cup.` $\rightarrow$ `dog.` $\rightarrow$ `car.` $\rightarrow$ `person.` |
| **Detector Calls (per image)** | ১ বার | ৫ বার |
| **Box / Text Threshold** | `0.35` / `0.30` | `0.35` / `0.30` |
| **SAM 2.1 Input** | সব Predicted Box একসাথে | ৫টি কলের প্রাপ্ত Box একত্র করে |
| **Dataset** | ২০টি Calibration Image | ২০টি Calibration Image |

> ⚠️ **Latency Note:** `separate` মোডে প্রতি ইমেজে ৫টি আলাদা ডিটেক্টর কল হওয়ায় স্বভাবতই এর প্রসেসিং সময় (Latency) Baseline-এর তুলনায় বেশি হবে।

---

## ৩. Evaluation Metrics & Success Criteria

আগের মতোই ইভ্যালুয়েশনের নিয়মগুলো অপরিবর্তিত থাকবে:

* **Detection TP:** সঠিক ক্লাস + এক-থেকে-এক (1:1) ম্যাচিং + Box IoU $\ge 0.50$।
* **সফল Final Mask:** ওই Matched Detection-এর Mask IoU $\ge 0.50$।
* **Detection FP:** Unmatched Prediction (অস্পষ্ট বা Unmapped লেবেলগুলোও এর অন্তর্ভুক্ত)।

### গাণিতিক সূত্র (Formulas):

$$\text{Conditional Mask Success} = \frac{\text{Good Masks}}{\text{Detection Matches}}$$


*(এটি প্রিসিশন নয়, এটি কেবল ডিটেক্ট হওয়া অবজেক্টগুলোর মধ্যে মাস্কের সফলতার হার নির্দেশ করে।)*

$$\text{Final Mask Recall} = \frac{\text{Good Masks}}{\text{All Target GT Instances}}$$

---

## ৪. The 4-Step Evaluation Workflow (A-B-C-D)

পুরো পরীক্ষাটি ৪টি সুনির্দিষ্ট ধাপে পরিচালিত হবে:

### Step A: Compare (তুলনা করা)

একই Calibration ডেটাসেটে `combined` এবং `separate` প্রম্পট চালিয়ে Quality এবং Latency-এর তুলনা করা হবে।

* **Net Gain Analysis (Failure Breakdown):**
ধরা যাক Baseline-এ ৩৯টি এবং Separate-এ ৪৫টি ভালো মাস্ক পাওয়া গেল।
$\text{Net Gain} = 6$ মানেই শুধু ৬টি নতুন অবজেক্ট পাওয়া নয়। এমন হতে পারে: নতুন পাওয়া গেছে ১০টি, আর আগে পাওয়া অবজেক্ট হারিয়েছে ৪টি ($10 - 4 = 6$)। তাই কোন অবজেক্ট উদ্ধার হলো আর কোনটি হারালো, তার তুলনামূলক বিশ্লেষণ জরুরি।

### Step B: Freeze (কৌশল চূড়ান্ত করা)

Calibration ডেটা থেকে পাওয়া ফলের ভিত্তিতে একটি স্থির কনফিগারেশন নির্বাচন করা হবে।

> 📌 **Selection Rule (সিদ্ধান্তের নিয়ম):**
> `separate` প্রম্পট গ্রহণ করা হবে **যদি** Final Mask Recall বৃদ্ধি পায় **এবং** Detection F1 Score না কমে। অন্যথায় `combined` প্রম্পট বহাল থাকবে। (বাস্তব ক্ষেত্রে Latency Budget-ও একটি শর্ত হতে পারে)।

* **লক (Lock):** এই সেলের পর প্রম্পট, থ্রেশহোল্ড বা মডেল আর পরিবর্তন করা যাবে না।

### Step C: Test (চূড়ান্ত পরীক্ষা)

নির্বাচিত এবং চূড়ান্ত (Frozen) স্ট্র্যাটেজিটি **Untouched Test Images (২০টি)**-এর ওপর চালানো হবে।

* টেস্ট সেটে কোনোভাবেই আর `combined` বনাম `separate` পরীক্ষা করা হবে না।
* টেস্ট সেটের GT Instance সংখ্যা ক্যালিব্রেশন থেকে ভিন্ন হতে পারে, তাই Denominator নতুন করে হিসাব করতে হবে।

### Step D: Close (ফলাফল বিশ্লেষণ)

ফলাফল থেকে চূড়ান্ত সিদ্ধান্ত গ্রহণ করা হবে।

* **GT Failure-এর দুটি রূপ:** ১. Detection Match হয়নি (বক্স আসেনি বা ভুল ক্লাস/কম IoU এসেছে) ২. Match হলেও মাস্ক খারাপ এসেছে।
* **Variance Acceptance:** Test Score এবং Calibration Score হুবহু এক হওয়া বাধ্যতামূলক নয়। ছবির ধরন, অবজেক্টের সাইজ ও কাঠিন্য (Difficulty) আলাদা হওয়ার কারণে স্কোরে ভিন্নতা আসতে পারে।

---

## ৫. 💡 Key Takeaways & Warnings

1. **Unmapped/Ambiguous Labels:** আলাদা প্রম্পট ব্যবহার করলে `cup car`-এর মতো অস্পষ্ট লেবেল কমে আসতে পারে, কিন্তু এটি ভুল বক্স বা মিসড অবজেক্ট শতভাগ দূর করবে—এমন কোনো গ্যারান্টি নেই।
2. **"Detection match হয়নি" মানেই Box না আসা নয়:** অনেক সময় মডেল বক্স প্রেডিক্ট করে, কিন্তু সেটি ভুল ক্লাস প্রেডিক্ট করলে বা Box IoU $0.50$-এর নিচে থাকলে সেটি আনম্যাচড (Unmatched) বা FP হিসেবে গণ্য হয়।
3. **Data Leakage Prevention:** টেস্ট সেটের ফলাফল দেখে কোনোভাবেই স্ট্র্যাটেজি বা থ্রেশহোল্ড পরিবর্তন করা যাবে না। করলে সেটি আর "Untouched Test Evaluation" থাকবে না, বরং তা ওভারফিটিংয়ের কারণ হবে।