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