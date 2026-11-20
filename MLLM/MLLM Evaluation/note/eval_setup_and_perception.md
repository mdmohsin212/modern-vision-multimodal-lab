**Q1: Expected `no`, model বলল `yes?`। আজকের scorer কেন এটিকে `invalid` বলবে?**

**Ans:** আজকের `closed_choice_v1` scorer-এ থাকা `normalize_choice()` ফাংশনটি টেক্সটের শেষের শুধুমাত্র ডট (`.`) সরায় (`removesuffix(".")`), প্রশ্নবোধক চিহ্ন (`?`) সরায় না। ফলে পুরো প্রক্রিয়াটি নিচের মতো ঘটে:

* `normalize_choice("yes?")` চালানোর পর `parsed` মান `"yes?"`-ই থেকে যায়।
* এরপর scorer চেক করে `valid = parsed in sample["choices"]`—অর্থাৎ `"yes?"` অনুমোদিত অপশন `["yes", "no"]`-এর ভেতর আছে কি না।
* যেহেতু `"yes?"` ওই তালিকায় নেই, তাই `valid` হয় `False` এবং `correct`-ও হয় `False`।
* `score_sample()`-এর নিয়ম অনুযায়ী `valid` যখন `False` হয়, তখন `status` আর `"wrong_answer"` না হয়ে সরাসরি `"invalid_output"` হয়ে যায় (যদি মডেল শুধু `"yes"` বলত, তবে সেটি valid choice হতো এবং `status` হতো `"wrong_answer"`)।


---

**Q2: Receipt raw answer-এ Markdown fence আছে, ভেতরের JSON ও values ঠিক। Raw validity, schema validity ও correctness কী হবে?**

**Ans:** `score_doc()` ও `score_sample()` ফাংশনের লজিক অনুযায়ী ফলাফল হবে:

* **Raw validity (`raw_json_valid`):** **`False`** — কারণ `json.loads(answer)` সরাসরি raw স্ট্রিংটি parse করতে গিয়ে Markdown fence-এর কারণে `JSONDecodeError` খাবে এবং `raw_valid = False` সেট করবে। এরপর regex দিয়ে ফেন্স পরিষ্কার করায় `used_fence_cleanup` হবে `True`।
* **Schema validity (`schema_valid`):** **`True`** — ফেন্স সরিয়ে `fence.group(1)` থেকে `json.loads()` সফলভাবে dict তৈরি করবে; যেহেতু ভেতরের JSON-এর keys (`FIELDS`) ও datatype (`str` বা `None`) সঠিক, তাই `schema_valid` হবে `True`।
* **Correctness (`all_correct` / `score` / `status`):** **`True`** (`score: 1`, `status: "correct"`) — কারণ `schema_valid` সত্য এবং সব ফিল্ডের মান রেফারেন্সের সাথে মিলে যাওয়ায় `all_correct` হবে `True`। আর `score_sample()` ফাইনাল স্কোর ও স্ট্যাটাস নির্ধারণে `raw_json_valid` নয়, বরং `schema_valid` ও `all_correct` ব্যবহার করে।



---

**Q3: দুই model-এর accuracy একই হলেও `sample_id` দিয়ে join করলে কী নতুন তথ্য পাওয়া যায়?**

**Ans:** গড় accuracy একটি সামগ্রিক সংখ্যা, যা দুই মডেলের ভেতরের আসল আচরণ লুকিয়ে রাখে। একই স্যাম্পলগুলোতে (`sample_id` দিয়ে join করে) পেয়ার্ড তুলনা করলে যে নতুন তথ্যগুলো পাওয়া যায়:

* **Improvement বনাম Regression (Wins/Losses):** কোন নির্দিষ্ট স্যাম্পলগুলো `base` ভুল করেছিল কিন্তু নতুন মডেল (`llm_lora`) ঠিক করেছে (`0 → 1`), আর কোন স্যাম্পলগুলো `base` আগে ঠিক পারত কিন্তু নতুন মডেল সেগুলো নষ্ট করে ভুল করেছে (`1 → 0`)—তা স্পষ্ট ধরা পড়ে। Accuracy সমান মানে হতে পারে দুই মডেল একদম একই স্যাম্পলে জিতছে-হারছে, অথবা নতুন মডেল যতগুলো ঠিক করেছে ঠিক ততগুলোতেই regression ঘটিয়েছে।
* **Shared Failures (দুই মডেলই যেখানে ব্যর্থ):** কোন কঠিন বা ত্রুটিপূর্ণ স্যাম্পলগুলোতে দুই মডেলই `0` পাচ্ছে (`0 → 0`), তা আলাদা করে চিহ্নিত করা যায়।
* **Error Type ও Output-এর সূক্ষ্ম পরিবর্তন:** একই `sample_id`-এর পাশে দুই মডেলের `status` (`wrong_answer` বনাম `invalid_output`), `raw_answer` (যেমন Markdown fence কমল কি না) এবং ফিল্ড-লেভেল ভুল (যেমন `date` ঠিক হয়ে `total` ভাঙল কি না) পাশাপাশি মিলিয়ে দেখা যায়।