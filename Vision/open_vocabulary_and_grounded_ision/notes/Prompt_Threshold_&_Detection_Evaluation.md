# Study Notes: Quantifying Grounding DINO with Precision, Recall & IoU Matching

## ১. Ground Truth Evaluation Setup & Matching Rules

Segmentation dataset-এর normalized polygon পয়েন্টগুলো থেকে প্রথমে আমরা অবজেক্টের গ্রাউন্ড ট্রুথ (Ground Truth/GT) বাউন্ডিং বক্স তৈরি করি। মডেলের একটি Prediction-কে **True Positive (TP)** হিসেবে গণ্য করার জন্য ৩টি শর্ত **একযোগে** পূরণ হতে হবে:

1. **Class Match:** Predicted class এবং GT annotation-এর class অবিকল একই হতে হবে।
2. **Spatial Overlap (IoU):** Predicted box এবং GT box-এর **$\text{IoU} \ge 0.50$** হতে হবে।
3. **Uniqueness Constraint:** ওই নির্দিষ্ট GT annotation-টি যেন আগে অন্য কোনো উচ্চ স্কোরের prediction-এর সাথে ম্যাচ না করে থাকে।

```
                        ┌───────────────────────────────┐
                        │     Prediction Candidate      │
                        └──────────────┬────────────────┘
                                       │
                         Is Class Match == True?
                                  │
                       ┌──────────┴──────────┐
                      YES                   NO ──► False Positive (FP)
                       │
             Is IoU >= 0.50?
                       │
              ┌────────┴────────┐
             YES               NO ──► False Positive (FP)
              │
      Is GT Unmatched?
              │
     ┌────────┴────────┐
    YES               NO ──► False Positive (FP)
     │
     ▼
True Positive (TP)

```

### Matching Edge Cases

* **False Positive (FP):** যেসব Prediction শর্ত পূরণ করতে পারে না অথবা অতিরিক্ত (Duplicate) হিসেবে আসে।
* **False Negative (FN):** যেসকল GT Annotation-এর সাথে কোনো Prediction সফলভাবে মিলতে পারে না।
* **Priority Rule:** যে Prediction-এর Confidence Score বেশি, সেটি আগে GT Annotation-এর সাথে ম্যাচ করার সুযোগ পাবে।
* **Duplicate Prediction Penalty:** একই অবজেক্টের জন্য দুটি Prediction বক্স আসলেও সর্বোচ্চ স্কোরেরটি TP হবে এবং অপরটি FP হবে (যদি IoU $\ge 0.50$ হয়)।

### Quantitative Metrics Equations

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

$$\text{F1-Score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## ২. Label Normalization & Unmapped FP Policy

Grounding DINO থেকে পাওয়া টেক্সট লেবেল সরাসরি ম্যাপ করার নিয়ম:

* **Clean Label Normalization:** `a car` $\rightarrow$ `car` (সহজ পোস্ট-প্রসেসিংয়ের মাধ্যমে ক্লিন করা হয়)।
* **Ambiguous Label Policy:** মডেল যদি অস্পষ্ট বা মাল্টি-ক্লাস লেবেল দেয় (যেমন: `person car`), তবে সেটিকে কোনো নির্দিষ্ট ক্লাসে বলপূর্বক (Forced) অ্যাসাইন করা হবে না।
* **Unmapped FP Handling:** অস্পষ্ট লেবেলগুলোকে **Unmapped FP** হিসেবে ট্রিট করা হয়। এর ফলে অস্পষ্ট লেবেল লুকিয়ে কৃত্রিমভাবে Precision বাড়ানোর কোনো সুযোগ থাকে না।

---

## ৩. Experimental Design: 3 Prompts × 4 Threshold Settings

### Prompt Engineering Principle

পরীক্ষা করার তিনটি প্রম্পটেই **একই পাঁচটি অবজেক্ট ক্লাস** অন্তর্ভুক্ত রাখা হয়েছে, যাতে তাদের মধ্যে Precision ও Recall তুলনা করা অর্থপূর্ণ হয়:

1. **Standard Class Names:** সাধারণ ক্লাসের নাম।
2. **Article Prefixed:** নামের পূর্বে `a` বা `an` যুক্ত প্রম্পট।
3. **Reversed Order:** ক্লাস নামগুলোর বিপরীত ক্রম।

> **Key Insight:** প্রম্পট লেখার সামান্য পরিবর্তনের কারণে Score বা Box পরিবর্তন হতে পারে। কিন্তু একক কোনো ছবির স্কোরের ওপর নির্ভর না করে ২০টি Calibration ছবির **TP, FP, FN** হিসাব করে সেরা প্রম্পট নির্ধারণ করতে হয়।

### Threshold Evaluation Trade-offs

* **`box_threshold` Dynamics:**
* কম `box_threshold` বেশি ক্যান্ডিডেট বক্স আউটপুটে রাখে $\rightarrow$ Recall বাড়তে পারে $\uparrow$, তবে FP বাড়ার ঝুঁকি থাকে $\uparrow$।


* **`text_threshold` Dynamics:**
* `text_threshold` পরিবর্তন করলে বক্সের স্থানাঙ্ক একই থাকলেও লেবেলে কোন টোকেন অন্তর্ভুক্ত হবে তা বদলে যায় $\rightarrow$ এটি সরাসরি **Class Matching** কে প্রভাবিত করে।



---

## ৪. Multi-Threshold Execution Strategy

কোডের তিনটি সেলের কার্যপ্রণালী এবং অপ্টিমাইজেশন লজিক:

| Cell Number | প্রধান কাজ | মূল টেকনিক্যাল বিষয় ও অপ্টিমাইজেশন |
| --- | --- | --- |
| **Cell 1** | Polygon to Box Conversion & Matching | Polygon থেকে বক্স বানিয়ে Class, IoU ($\ge 0.50$), এবং Uniqueness ম্যাচিং লজিক অ্যাপ্লাই করে। |
| **Cell 2** | Model Inference & Multi-threshold Testing | প্রতি ছবি ও প্রম্পটে মডেল ১ বার চলে। একই Model Output-এ ৪টি থ্রেশহোল্ড টিউন করা হয়। ফলে ১২টি সেটিংসের রেজাল্ট পেলেও মোট মডেল রান হয় মাত্র **$20 \text{ images} \times 3 \text{ prompts} = 60 \text{ times}$**। |
| **Cell 3** | Global & Per-Class Metric Evaluation | ১২টি সেটিংসের জন্য Micro F1 হিসাব করে সর্বোচ্চ **Micro F1** অর্জনকারী সেটিং নির্বাচন করে এবং তার **Per-Class Table** প্রদর্শন করে। |

---

## ৫. Small Dataset Limitations & Negative Image Diagnostics

* **`negative_image_fp` Diagnostic:** নির্বাচিত ৫টি ক্লাসের কোনো অ্যানোটেশন নেই এমন ২টি নেতিবাচক (Negative) ছবিতে প্রাপ্ত Prediction সংখ্যা। ছোট বা অসম্পূর্ণ গ্রাউন্ড ট্রুথ অ্যানোটেশনের কারণে কোনো দূরের অবজেক্ট বাস্তবে সঠিক হলেও অ্যানোটেশনে না থাকায় ইভালুয়েশনে **FP** হিসেবে গণ্য হতে পারে।
* **Small Sample Sensitivity:** ২০টি Calibration ছবিতে নির্দিষ্ট অবজেক্ট (যেমন: `dog` রয়েছে ৩টি ছবিতে, `car` ৪টি ছবিতে) খুব কম থাকায় **মাত্র ১টি ভুল Prediction** সেই নির্দিষ্ট ক্লাসের পারফরম্যান্স মেট্রিক অনেক বেশি পরিবর্তন করে দিতে পারে।
* **Per-class Table Necessity:** Micro F1 পুরো সিস্টেমের গাণিতিক গড় ভালো দেখালেও একক কোনো ক্লাস (যেমন: `car`) দুর্বল হতে পারে। তাই সিদ্ধান্ত নেওয়ার ক্ষেত্রে Per-class table রিভিউ করা অত্যন্ত জরুরি।


আপনার উত্তরগুলোকে সংশোধিত পর্যবেক্ষণ অনুসারে সম্পূর্ণ নির্ভুল ও সূক্ষ্ম গাণিতিক ব্যাখ্যাসহ আপডেট করা হলো। স্টাডি নোট বা ফাইলে সংরক্ষণ করার জন্য প্রশ্ন এবং সংশোধিত উত্তরসমূহ নিচে দেওয়া হলো:

---



# Q&A Note

### Q1: একই annotated dog-এর ওপর dog label-সহ দুটি prediction আছে; দুটির IoU-ই 0.5-এর বেশি। TP, FP, FN কত?

* **মেট্রিক মান:** **$\text{TP} = 1$, $\text{FP} = 1$, $\text{FN} = 0$**।
* **কোডের ভেতরের লজিক:** সিস্টেম সরাসরি উচ্চ IoU দেখে প্রথম ম্যাচ বেছে নেয় না; বরং দুটি প্রসেসড প্রেডিকশনের মধ্যে যার **Confidence Score বেশি**, সেটি আগে ম্যাচ করার সুযোগ পায় এবং অব্যবহৃত গ্রাউন্ড ট্রুথ (GT)-এর মধ্যে সর্বোচ্চ IoU থাকা অবজেক্টের সাথে ম্যাচ করে **TP** হিসেবে গণ্য হয়। একই GT অবজেক্টের জন্য পরের অতিরিক্ত প্রেডিকশনটি ডুপ্লিকেট হিসেবে **FP** হয়। কোনো গ্রাউন্ড ট্রুথ মিস না হওয়ায় **$\text{FN} = 0$**।


---

### Q2: Ground truth হলো একটি car, কিন্তু তার ওপর সঠিক জায়গায় person label-এর box এসেছে। Class-aware matching-এ কী গণনা হবে?

* **মেট্রিক মান:** **$\text{FP} = 1$ এবং $\text{FN} = 1$** গণনা হবে।
* **ব্যাখ্যা:** যেহেতু ক্লাসের মিল নেই (Class-aware matching), তাই ভুল লেবেলের `person` বক্সটি একটি অবাস্তব/ভুল ডিটেকশন হিসেবে **False Positive (FP)** হবে। অপরদিকে, মূল `car` অবজেক্টটির জন্য কোনো সঠিক প্রেডিকশন না পাওয়ায় আসল গাড়িটি মিসিং হিসেব করে **False Negative (FN)** গণনা করা হবে।



---

### Q3: Box threshold 0.35 থেকে 0.25 করলে precision ও recall—দুটিই কি নিশ্চিতভাবে বাড়বে? কেন?

* **না, নিশ্চিতভাবে দুটিই বাড়বে না।**
* **গাণিতিক সত্যতা:** থ্রেশহোল্ড কমালে ক্যান্ডিডেট বক্সের সংখ্যা বৃদ্ধি পায়। সাধারণত এর ফলে Recall বৃদ্ধি পাওয়ার এবং Precision হ্রাস পাওয়ার একটি প্রবণতা দেখা যায়, তবে এটি কোনো সার্বজনীন বা নিশ্চিত নিয়ম নয়। ডেটাসেটের ওপর ভিত্তি করে Precision এবং Recall উভয়ই অপরিবর্তিত থাকতে পারে, এমনকি নির্দিষ্ট পরিস্থিতিতে Precision বাড়তেও পারে।
* **পরীক্ষার রেজাল্ট (Reverse prompt, text threshold 0.30):** বক্স থ্রেশহোল্ড `0.35` থেকে কমানোর ফলে Recall `0.364` $\rightarrow$ `0.455`-এ **বৃদ্ধি পেয়েছে**, আর Precision `0.606` $\rightarrow$ `0.352`-এ **হ্রাস পেয়েছে**। এটি কেবল ওই নির্দিষ্ট রানের ফলাফল, কোনো বাধ্যবাধকতামূলক সাধারণ নিয়ম নয়।



---

### Q4: `a car`-কে `car` হিসেবে normalize করা হচ্ছে কেন? `person car`-কেও জোর করে একটি class দিলে কী সমস্যা?

* **`a car` Normalization:** আর্টিকেলের (`a`, `an`, `the`) ওপর নির্ভরতা কমিয়ে টেক্সট প্রম্পটের নির্দিষ্ট ক্লাসের সাথে নিখুঁতভাবে মেলানোর (Standardization) উদ্দেশ্যে `a car`-কে `car` হিসেবে নরম্যালাইজ করা হয়।
* **`person car` সমস্যা:** `person car`-এর মতো মাল্টি-ভার্বাল লেবেল মানেই যে একই বাউন্ডারি বক্সে দুটি অবজেক্ট আছে—এমনটা নাও হতে পারে; টেক্সট পোস্ট-প্রসেসিংয়ের সময় প্রম্পটের শব্দগুলো সঠিকভাবে বিভক্ত না হলেও এটি ঘটতে পারে। এ ধরনের অস্পষ্ট লেবেলকে জোর করে কোনো একটি নির্দিষ্ট ক্লাসে অনুমিত মান বসিয়ে দিলে ভুল TP বা FP গণনা হতে পারে। তাই আমাদের ইভালুয়েশন কোড সেটিকে কোনো ক্লাসে না বসিয়ে নিরপেক্ষভাবে **Unmapped FP** হিসেবে ধরে নেয়।



---

### Q5: আজকের ২০টি calibration ছবি দিয়ে setting নির্বাচন করার পর একই কাজ ২০টি final test ছবিতে আবার করলে final score সম্পর্কে কী সমস্যা হবে?

* টেস্ট সেট দেখে পুনরায় সেটিং বা থ্রেশহোল্ড টিউন করলে **Data Leakage (তথ্য ফাঁস)** ঘটবে।
* এর ফলে যে চূড়ান্ত ইভালুয়েশন স্কোর আসবে তা অতিরিক্ত আশাবাদী (Optimistic) এবং পক্ষপাতদুষ্ট (Biased) হবে। যদিও এতে মডেলের মূল ওয়েট (Model Weights) পরিবর্তিত হয় না, তবুও টেস্ট সেটে ওভারফিট হয়ে যাওয়ায় এই স্কোর দিয়ে নতুন বা অজানা যেকোনো বাস্তব ডেটাতে মডেলটির নিরপেক্ষ পারফরম্যান্স অনুমান করা অসম্ভব হয়ে পড়বে।