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



আপনার স্টাডি ফাইলের জন্য এই প্রশ্নোত্তরগুলোকে নির্ভুলভাবে আপডেট এবং ফরম্যাট করে নিচে দেওয়া হলো। এগুলো সরাসরি আপনার নোটবুক বা Markdown ফাইলে সেভ করে নিতে পারেন।

---

# ❓ Q&A Section:

## Part 1: Conceptual & Threshold Optimization

**Q1: Separate prompt-এ ambiguous label শূন্য হলেও detection precision কেন 100% নাও হতে পারে?**

* **উত্তর:** অ্যাম্বিগুয়াস লেবেল (যেমন `cup car`) দূর হলেও, মডেল ব্যাকগ্রাউন্ড নয়েজকে অবজেক্ট মনে করে বক্স দিতে পারে (Background False Positive), একই অবজেক্টের ওপর ডুপ্লিকেট অতিরিক্ত বক্স আসতে পারে, অথবা সম্পর্কহীন বস্তুকে ভুল ক্লাসে প্রেডিক্ট করতে পারে। এই ধরনের False Positive-এর কারণেই প্রিসিশন ১০০% হতে পারে না।

**Q2: Baseline-এ ৩৯টি ভালো mask। gained=10, lost=4 হলে নতুন good-mask count এবং ১১০ GT-এর ওপর mask recall কত?**

* **উত্তর:**
* **নতুন Good-Mask Count:** $39 + 10 - 4 = \mathbf{45}$ টি।
* **Mask Recall:** $\frac{45}{110} \approx \mathbf{0.4091}$ (বা ৪০.৯১%)।



**Q3: Mask recall বাড়লেও precision কমে এবং latency চার গুণ হয়—কোন তথ্যের ভিত্তিতে system নির্বাচন করবে?**

* **উত্তর:** সিস্টেম নির্বাচন করতে হবে অ্যাপ্লিকেশনের নির্ধারিত **Latency Budget** এবং **False Positive বনাম False Negative-এর প্রায়োগিক খরচের (Cost)** ওপর ভিত্তি করে। যদি ড্রপ হওয়া প্রিসিশন বড় ধরনের ব্যবসায়িক ভুল তৈরি করে এবং চার গুণ প্রসেসিং সময় ল্যাটেন্সি বাজেটকে অতিক্রম করে, তবে `Baseline` (Combined) সিস্টেমই বহাল রাখতে হবে।

**Q4: Test result খারাপ দেখে threshold বদলে একই test set আবার ব্যবহার করলে final score-এর বিশ্বাসযোগ্যতায় কী সমস্যা হয়?**

* **উত্তর:** এতে টেস্ট সেটের তথ্য কনফিগারেশন নির্বাচনে ব্যবহৃত হয়ে যায়, ফলে ইভ্যালুয়েশন চরমভাবে পক্ষপাতদুষ্ট (Biased) হয়ে পড়ে। মডেলের Weights পরিবর্তন না হলেও কনফিগারেশনটি টেস্ট সেটের জন্য ওভারফিট হয়ে যায়, যার কারণে ফাইনাল স্কোর তার নিরপেক্ষতা ও বিশ্বাসযোগ্যতা সম্পূর্ণরূপে হারায়। (অর্থাৎ, অদেখা ডেটাতে মডেলটি কেমন করবে তা আর অনুমান করা যায় না)।

**Q5: তোমার actual test result-এ বেশি GT failure কোন stage-এ? সংখ্যা দিয়ে ব্যাখ্যা করো।**

* **উত্তর:** Day 3 Calibration ফলাফলে দেখা গেছে, সবচেয়ে বেশি ব্যর্থতা ঘটেছে **Detection stage** (Grounding DINO)-এ। ১১০টি গ্রাউন্ড ট্রুথের মধ্যে ডিটেকশন মিস (Matching Failure) ছিল **৭০টি**। যেখানে ম্যাচ হওয়া ৪০টি ডিটেকশনের মধ্যে মাস্ক ফেইলিউর ছিল মাত্র **১টি** এবং mean mask IoU ছিল অনেক ভালো ($0.801$)।

---

## Part 2: Evaluation Scenarios & SAM Limitations

**Q1: Separate prompt-এ `cup car` label আর নেই। এতে কি প্রমাণ হলো সব cup detection সঠিক? কেন?**

* **উত্তর:** না, এটি প্রমাণ করে না যে সব কাপ ডিটেকশন সঠিক। Separate prompt-এ কেবল টেক্সট আর্টফ্যাক্ট বা লেবেল কনফিউশন দূর হয়েছে, কিন্তু ব্যাকগ্রাউন্ডের কোনো বস্তুকে মডেল কাপ ভেবে ভুল বক্স দিতে পারে, অথবা অন্য ক্লাসের অবজেক্টকে কাপ হিসেবে প্রেডিক্ট (False Positive) করতে পারে।

**Q2: Baseline-এ ৩৯টি ভালো mask। Separate mode-এ gained=10, lost=4 হলে নতুন good-mask count এবং end-to-end mask recall কত?**

* **উত্তর:**
* **নতুন Good-Mask Count:** $39 + 10 - 4 = \mathbf{45}$ টি।
* **End-to-end Mask Recall:** $\frac{45}{110} \approx \mathbf{0.4091}$ (বা ৪০.৯১%)।



**Q3: Mask recall 0.355 → 0.420, কিন্তু precision 0.606 → 0.400 এবং latency 0.64 → 2.60 seconds। এক কথায় “উন্নত” বলা যাবে কি?**

* **উত্তর:** না, এক কথায় “উন্নত” বলা যাবে না। রিকল কিছুটা বাড়লেও প্রিসিশন উল্লেখযোগ্যভাবে কমেছে। এছাড়াও, ২.৬০ সেকেন্ড লেটেন্সি গ্রহণযোগ্য কি না তা সম্পূর্ণ নির্ভর করবে অ্যাপ্লিকেশনের নির্ধারিত Latency Budget-এর ওপর (যেমন: অফলাইন ব্যাচ প্রসেসিংয়ে এটি চললেও রিয়েল-টাইম ট্র্যাকিংয়ে এটি চলবে না)। এটি একটি বড়সড় ভারসাম্যহীন ট্রেড-অফ।

**Q4: দুই strategy-তেই একটি GT object detection miss। শুধু SAM পরিবর্তন করলে (আপডেট করলে) এই box-prompt pipeline-এ সেটি উদ্ধার হবে কি?**

* **উত্তর:** না, সেটি কোনোভাবেই উদ্ধার হবে না। কারণ SAM একটি বক্স-প্রম্পটেড (Box-prompted) মডেল; ডিটেক্টর (Grounding DINO) অবজেক্টটি মিস করায় সেখানে কোনো বাউন্ডিং বক্স তৈরি হবে না। আর বক্স-প্রম্পট ছাড়া উন্নত SAM-ও নিজে থেকে অজানা বা অদৃশ্য অবজেক্ট খুঁজে বের করতে পারে না।