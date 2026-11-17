**Q1: Projector-only run-এ LLM frozen থাকা সত্ত্বেও merger gradient পেল কীভাবে?**

**Ans:** Forward pass-এর সময় পুরো মডেলের (Vision Encoder → Merger → Frozen LLM → Loss) ওপর একটি **computation graph** তৈরি হয়।

* **Gradient-এর উল্টো পথ (Backpropagation):** Loss হিসাব হওয়ার পর gradient সেই computation graph ধরে উল্টো পথে frozen LLM-এর ভেতর দিয়ে প্রবাহিত হয়ে `merger` পর্যন্ত পৌঁছায়।


* **Freeze বনাম `no_grad()`:** কোনো মডিউল বা LLM-কে freeze করা (`requires_grad=False`) মানে শুধু সেই অংশের নিজস্ব weight update বন্ধ রাখা—এটি gradient-এর প্রবাহ বা পথ বন্ধ করে না। একমাত্র `torch.no_grad()` ব্যবহার করলে computation graph তৈরি হতো না এবং gradient-এর পথ কেটে যেত। যেহেতু `requires_grad=False`-এ gradient চলাচলের পথ খোলা থাকে, তাই `merger` ঠিকই gradient পায় এবং আপডেট হতে পারে।



---

**Q2: `requires_grad=True` আছে, কিন্তু parameter optimizer-এ নেই। Backward-এর পর কী হতে পারে, আর optimizer step-এ কী হবে?**

**Ans:**

* **Backward-এর পর যা হবে:** প্যারামিটারটিতে `requires_grad=True` থাকায় `loss.backward()` চালানোর পর এর `.grad` অ্যাট্রিবিউটে ঠিকই gradient হিসাব হয়ে জমা হবে (এবং এর জন্য সামান্য মেমরিও খরচ হবে)।


* **Optimizer step-এ যা হবে:** `optimizer.step()` চালানোর সময় optimizer কেবল তার কাছে পাস করা বা রেজিস্টার করা প্যারামিটারগুলোকেই আপডেট করে। যেহেতু এই প্যারামিটারটি optimizer-এ নেই, তাই gradient জমা থাকা সত্ত্বেও এর **weight কখনোই update হবে না**—অর্থাৎ কোনো শেখা জমা হবে না। (অনেকটা হোমওয়ার্ক করে খাতায় লিখে রাখার মতো, কিন্তু জমা দেওয়ার শিক্ষক নেই—খাতা টেবিলেই পড়ে থাকবে।)



---

**Q3: `model.visual.merger` trainable করলে `deepstack_merger_list`-ও কি trainable হয়? তোমার printout দিয়ে কীভাবে যাচাই করবে?**

**Ans:** **না**, সরাসরি `model.visual.merger` মডিউলটিকে trainable করলে `deepstack_merger_list` স্বয়ংক্রিয়ভাবে trainable হয় না।

* **কারণ:** Qwen3-VL আর্কিটেকচারে `model.visual.merger` এবং `model.visual.deepstack_merger_list` হলো `model.visual`-এর অধীনে দুটি সম্পূর্ণ আলাদা সাব-মডিউল (sibling modules)। `deepstack_merger_list` মূল `merger`-এর ভেতরে নেস্টেড (nested) নয়। তাই শুধু `model.visual.merger.parameters()`-এ `requires_grad=True` সেট করলে `deepstack_merger_list` আগের মতোই frozen (`requires_grad=False`) থেকে যায়। *(তবে কেউ যদি নামের সাবস্ট্রিং `"merger" in name` দিয়ে ফিল্টার করে আনফ্রিজ করে, কেবল সেক্ষেত্রেই দুটি একসাথে trainable হবে।)*
* **Printout দিয়ে যাচাই করার উপায়:**
1. **Trainable Parameter Names চেক:** প্রিন্টআউটে যেসব প্যারামিটারের `requires_grad=True` দেখানো হচ্ছে, সেই তালিকায় শুধু `visual.merger...` নামগুলো আছে নাকি `visual.deepstack_merger_list...` নামগুলোও প্রিন্ট হয়েছে তা দেখে।
2. **Trainable Parameter Count ও `.grad` চেক:** প্রিন্টআউটে মোট trainable parameter-এর সংখ্যা মিলিয়ে এবং `backward()`-এর পর `deepstack_merger_list`-এর প্যারামিটারগুলোর `.grad` মান `None` (বা `grad_norm=0`) আছে কি না তা দেখেই নিশ্চিত হওয়া যাবে যে সেটি trainable হয়নি।


**Q4: LoRA-এর trainable parameters কম হলেও কেন পুরো training peak একই অনুপাতে কমে না?**

**Ans:** কারণ training-এর সময় GPU VRAM শুধুমাত্র trainable parameters-এর জন্য খরচ হয় না। LoRA মূলত **gradients** এবং **optimizer states** (যেমন AdamW-এর momentum ও variance)-এর মেমরি অনেক কমিয়ে দেয়, কিন্তু বাকি দুটি বড় মেমরি খরচ আগের মতোই থেকে যায়:

* **Base Model Weights:** মূল মডেলের weights frozen থাকলেও forward pass এবং gradient flow-এর জন্য পুরো base model-টি GPU মেমরিতে লোড রাখতেই হয়।
* **Activations:** Backward pass-এ gradient হিসাব করার জন্য forward pass-এর সময় লেয়ারগুলোতে তৈরি হওয়া মধ্যবর্তী **activations** মেমরিতে জমা রাখতে হয়। এই activation memory নির্ভর করে batch size, sequence length এবং image tokens-এর সংখ্যার ওপর—trainable parameter কম-বেশির ওপর নয়।

এ কারণে trainable parameters ১%-এর নিচে নেমে এলেও মোট peak GPU memory সেই অনুপাতে কমে না।

---

**Q5: একই receipt-এ loss কমেছে কিন্তু validation total accuracy বাড়েনি—কীভাবে সম্ভব?**

**Ans:** এর পেছনে দুটি মূল কারণ কাজ করে:

* **গড় Token Loss বনাম Strict Exact Match:** Training loss হলো আউটপুটের সব টোকেনের গড় cross-entropy loss। মডেল হয়তো JSON-এর কাঠামোগত টোকেনগুলো (যেমন `{"item":`, ব্র্যাকেট, কমা বা সহজ শব্দ) আগের চেয়ে বেশি আত্মবিশ্বাসের সাথে প্রেডিক্ট করছে, ফলে গড় loss কমেছে। কিন্তু `total` accuracy বাড়তে হলে `total` ফিল্ডের প্রতিটি ডিজিট হুবহু মিলতে হয়। একটি ডিজিট ভুল হলেই গড় loss কমলেও accuracy শূন্যই থেকে যায়।
* **Memorization (Overfitting) ও Visual Detail:** মডেলটি নির্দিষ্ট receipt-এর টেক্সট প্যাটার্ন মুখস্থ করে ফেলেছে (ফলে ওই receipt-এ loss কমেছে), কিন্তু নতুন validation receipt-এ generalize করতে পারছে না। আবার অনেক সময় ছবির রেজোলিউশন কম থাকলে `total`-এর সংখ্যাটি ঝাপসা থাকে—ফলে মডেল ফরম্যাট শিখে loss কমালেও ছবি থেকে সঠিক সংখ্যাটি পড়তে ব্যর্থ হয়।

---

**Q6: `best_adapter` আর `resume_step8.pt` কোন দুটি আলাদা কাজের জন্য?**

**Ans:** এই দুটি ফাইল সম্পূর্ণ ভিন্ন দুটি উদ্দেশ্যে সংরক্ষণ করা হয়:

* **`best_adapter` (Evaluation ও Inference-এর জন্য):** এতে শুধুমাত্র validation-এ সেরা ফলাফল দেওয়া **trained LoRA adapter weights** এবং কনফিগারেশন থাকে। এটি আকারে খুব ছোট হয় এবং ট্রেনিং শেষে base model-এর সাথে যুক্ত করে **নতুন ছবিতে প্রেডিকশন (inference) বা ডেপ্লয় করার জন্য** ব্যবহৃত হয়। এতে optimizer-এর কোনো তথ্য থাকে না।
* **`resume_step8.pt` (মাঝপথে থেমে যাওয়া Training আবার চালু করার জন্য):** এটি একটি সম্পূর্ণ **training checkpoint**। এতে মডেলের weights-এর পাশাপাশি **optimizer states** (AdamW momentum/variance), **learning rate scheduler state**, বর্তমান **step/epoch নম্বর** এবং **random seed states** সেভ থাকে। GPU crash বা সময় শেষ হওয়ার কারণে ট্রেনিং বন্ধ হয়ে গেলে ঠিক step 8 থেকে আগের গতি ও শেখার ধারাবাহিকতা বজায় রেখে **ট্রেনিং পুনরায় শুরু (resume) করার জন্য** এটি ব্যবহৃত হয়।