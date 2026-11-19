# Section D — Multimodal Fine-tuning: Batch থেকে Merge পর্যন্ত

**কাজ:** Qwen3-VL-2B-Instruct দিয়ে SROIE receipt থেকে `company`, `date` ও `total` বের করা। Section C-তে আমরা QLoRA দিয়ে task শেখানো দেখেছি। Section D-তে একই task ব্যবহার করে শিখেছি **কোন tokens loss পায়, কোন parameters শেখে, সেই পরিবর্তন fair ভাবে কীভাবে মাপি, আর saved model কীভাবে আবার চালাই।**

**এক মিনিটের flow:** Receipt + instruction + target JSON → processor দিয়ে batch → assistant answer-এ loss → নির্বাচিত weights-এ gradient → optimizer update → validation দিয়ে checkpoint বাছাই → held-out receipts-এ paired evaluation → resume বা inference export।

| মনে রাখার প্রশ্ন | কোথায় উত্তর খুঁজবে? |
|---|---|
| Model কী দেখতে পাচ্ছে? | `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw` |
| কোন token predict করতে ভুল হলে loss হবে? | `labels`-এ যে positions `-100` নয় |
| কোন weights শেখার সুযোগ পাচ্ছে? | `requires_grad`, optimizer parameter list ও backward-এর পরে `.grad` |
| শেখা কাজে লেগেছে? | একই held-out receipt-এ field accuracy এবং gained/lost examples |
| Training চালাব, নাকি শুধু prediction নেব? | Resume checkpoint, adapter/merger export, অথবা merged model |

## ১. Day 1 — Receipt থেকে VLM training batch

একটি training example-এ তিনটি অংশ: **receipt image**, user instruction, আর **assistant target JSON**। `messages` format-এ user ও assistant turn একই list-এ থাকে। Conversational `prompt/completion` format-এ user turn `prompt`-এ এবং assistant turn `completion`-এ থাকে। আমাদের collator দুই format থেকেই একই conversation ও labels বানিয়েছে।

**Image placeholder ছবির অবস্থান বলে।** `{"type": "image"}` দেখে chat template user message-এ visual marker বসায়। সেই marker কোনো pixel data নয়। **Tokenizer** text-কে IDs বানায়। **Multimodal processor** text tokenization-এর পাশাপাশি image resize ও patch preparation করে এবং grid অনুযায়ী image slots বসায়। **Collator** কয়েকটি example থেকে batch ও loss labels তৈরি করে।

| Tensor | Shape | সহজ অর্থ |
|---|---|---|
| `input_ids` | `[B, L]` | Prompt, image slots, answer ও padding-এর IDs |
| `attention_mask` | `[B, L]` | সাধারণত ১ = ব্যবহারযোগ্য position, ০ = padding; causal order model আলাদাভাবে সামলায় |
| `pixel_values` | `[P, D]` | Vision encoder-এর processed patch input; `P` সব ছবির মোট patch rows |
| `image_grid_thw` | `[num_images, 3]` | প্রতি image-এর temporal, height ও width patch grid |
| `labels` | `[B, L]` | Prediction targets; loss-এ ignore করার index `-100` |

**Patch rows আর image slots এক জিনিস নয়।** Qwen3-VL-এ grid `[1, 20, 30]` থেকে pre-merge patch rows = `1 × 20 × 30 = 600`। Spatial merge size 2 হলে vision features-এর প্রতি 2×2 group থেকে একটি visual embedding হয়। তাই image slots = `600 ÷ 4 = 150`। Processor এই ১৫০টি জায়গা রাখে; model সেখানে merged image features বসায়। `pixel_values` তাই LLM embedding নয়। Debug করার সময় **প্রতি ছবির grid থেকে expected merged slots** বের করে তার image-token count-এর সঙ্গে মিলাও।

**Visual-token budget** image resize নিয়ন্ত্রণ করে। একই receipt-এ budget কমালে সাধারণত visual slots ও sequence length কমে। এতে memory বাঁচতে পারে, কিন্তু ছোট print কম স্পষ্টও হতে পারে। Shape দেখে resource cost-এর ইঙ্গিত পাও; actual extraction quality জানতে prediction মাপো। Image slots বা answer কেটে যায় এমন arbitrary truncation এড়িয়ে চলো।

## ২. SFT ও assistant-only loss — context আর target আলাদা

**Supervised Fine-Tuning (SFT)**-এ target answer দিয়ে model-কে পরের token predict করতে শেখাই। Training-এর সময় model-এর context-এ image, instruction এবং আগের সঠিক answer tokens থাকে (**teacher forcing**)। Inference-এ image + instruction থেকে শুরু করে model নিজের generated tokens ব্যবহার করে। কোনো reference answer inference input-এ যায় না।

| Position | Input-এ থাকে? | সরাসরি loss target? |
|---|---|---|
| User prompt + image slots | হ্যাঁ | না, label `-100` |
| Assistant header | হ্যাঁ | না, আমাদের mask-এ header পর্যন্ত prompt |
| Target JSON + assistant end token | হ্যাঁ, training-এ | হ্যাঁ |
| Batch padding | Tensor-এ আছে; attention mask ০ | না, label `-100` |

**Mask বানানোর ধাপ:** একই image settings দিয়ে full conversation এবং prompt-only conversation (assistant header সহ) process করো। Prompt tokens full sequence-এর **exact prefix** কি না check করো। তারপর `input_ids.clone()` থেকে labels বানিয়ে prefix, image slot ও padding positions-এ `-100` বসাও। Model ভেতরে next-token shift করে; হাতে দ্বিতীয়বার shift কোরো না।

```python
# Training batch থেকে শুধু supervised tokens পড়ো।
target_ids = batch["labels"][0]
target_ids = target_ids[target_ids != -100]
print(processor.tokenizer.decode(target_ids, skip_special_tokens=False))
```

প্রত্যাশিত: **শুধু target JSON ও `<|im_end|>`**। User text দেখা গেলে prefix boundary ভুল হতে পারে। কিছুই দেখা না গেলে truncation বা masking-এ answer চলে গেছে কি না দেখো। `labels=-100` image/prompt মুছে দেয় না; answer loss থেকে trainable visual অংশেও gradient যেতে পারে।

**TRL note:** format দেখে loss policy অনুমান কোরো না। আমরা custom collator-এর actual labels পরীক্ষা করেছি। Reference করা TRL **v0.29.0**-এ VLM prompt/completion collator `completion_only_loss=True` দিয়ে prompt mask করতে পারে। ওই version-এর VLM trainer-এ `assistant_only_loss=True` সরাসরি supported নয়। Version ও template বদলালে আবার batch খুলে দেখো।

## ৩. Day 2 — কোথায় training হবে? Vision, merger, LLM

**Vision tower** patches থেকে features বানায়। **Final visual merger** সেগুলোকে LLM-এর image slots-এর উপযোগী করে। **DeepStack mergers** কয়েকটি intermediate vision block-এর features LLM-এর শুরুর layers-এ পাঠায়। `model.visual.merger` ও `model.visual.deepstack_merger_list` আলাদা modules।

| Setup | আমরা কী trainable করেছি? | কী বদলাতে পারে? |
|---|---|---|
| **LLM LoRA** | Text attention-এর `q_proj` ও `v_proj`-এ LoRA A/B; original weights frozen | LLM কীভাবে image ও instruction context ব্যবহার করে |
| **Projector-only** | শুধু `model.visual.merger`-এর original weights; DeepStack mergers frozen | Final visual features কীভাবে LLM-এ পৌঁছায় |
| **Partial vision** | শুধু vision tower-এর শেষ block-এর original weights | Merger-এ যাওয়ার আগের visual features |

Partial vision ছিল **এক batch-এর backward check**। Pilot training হয়েছে প্রথম দুটো setup-এ।

**Freeze মানে weight update বন্ধ; gradient path কাটার দরকার নেই।** Frozen LLM-এর operations দিয়ে answer loss-এর gradient trainable merger পর্যন্ত যেতে পারে। পুরো LLM forward `torch.no_grad()`-এ চালালে বা merger output detach করলে সেই পথ কেটে যেতে পারে। `model.eval()`-ও freeze নয়; এটি dropout-এর মতো training-mode behaviour বদলায়।

**তিনটি প্রমাণ আলাদা করে দেখো:**

1. **Intended:** `named_parameters()`-এ intended names ও `requires_grad=True`।
2. **Registered:** Optimizer ওই parameter objects-ই পেয়েছে; count-এর সঙ্গে identity-ও মিলিয়েছি।
3. **Connected:** Actual receipt loss backward করে intended অংশে finite nonzero gradient আছে। আগে gradients clear করলে frozen parameters-এর `.grad is None` থাকা উচিত। Zero gradient একা freeze প্রমাণ করে না।

Day 2-তে `optimizer.step()` ছিল না, তাই ওই check প্রমাণ করেছে **gradient ও optimizer wiring ঠিক**, weights বদলেছে এমন প্রমাণ দেয়নি।

## ৪. Day 3 — LoRA, NF4 এবং একটি optimizer update

LoRA frozen linear layer-এর পাশে ছোট trainable branch যোগ করে। Input dimension `d_in`, output dimension `d_out`, rank `r` হলে `A`-এর shape `[r, d_in]` এবং `B`-এর shape `[d_out, r]`।

$$
W_{\mathrm{effective}} = W_0 + \frac{\alpha}{r}BA
$$

`rank` update-এর capacity, `alpha/r` আমাদের standard LoRA setup-এ update-এর scale। Target modules বলে **কোন layers-এ** এই update বসবে। Pilot-এ `q_proj` ও `v_proj`, rank 8, alpha 16। Default initialization-এ `B` zero হলে শুরুতে base output বদলায় না, আর প্রথম backward-এ `A`-এর gradient zero হতে পারে।

| Memory knob | কী নিয়ন্ত্রণ করে? | কী মনে রাখবে? |
|---|---|---|
| **LoRA** | কত weights-এর gradients ও optimizer states লাগবে | Frozen base ও backward activations এখনো memory নেয় |
| **QLoRA / NF4** | Frozen base-এর নির্বাচিত linear weights কীভাবে stored | 4-bit storage মানে সব arithmetic বা সব tensors 4-bit নয় |
| **Mixed precision** | অনেক forward operations-এর compute dtype | FP16-এ loss scaling কাজে লাগে; supported GPU-তে BF16 সাধারণত scaler ছাড়াই চলে |
| **Gradient accumulation** | কয়টি micro-batch শেষে optimizer update | Micro-batch 1, accumulation 8 → এক GPU-তে effective batch 8 |
| **Gradient checkpointing** | কত forward activation ধরে রাখব | কিছু activation backward-এ recompute করে memory বাঁচায়; disk-এ saved checkpoint থেকে আলাদা |
| **Image budget** | কত visual tokens process হবে | কম tokens memory বাঁচাতে পারে; text reading খারাপও হতে পারে |

**আমাদের manual update:** gradients clear → আটটি receipt একে একে forward/backward, প্রতিটির loss ৮ দিয়ে ভাগ → FP16-এ শেষে একবার unscale → gradient clip → optimizer step → scaler update → scheduler step। এই loop-এ আমরা প্রতি receipt-এর mean supervised loss সমান weight দিয়েছি। এটি সব receipt-এর tokens একসঙ্গে ধরে token-weighted mean করার মতো নয়, যদি answer lengths আলাদা হয়।

**Training peak কেন trainable count-এর অনুপাতে কমে না?** Base weights, activations, gradients, optimizer moments ও temporary allocations সবই লাগে। কোন module trainable হলো তা backward path বদলায়। Day 2-এর peak ছিল এক batch backward পর্যন্ত; Day 3-এর peak-এ optimizer step-ও ছিল। তাই দুই day-এর peak সরাসরি একই মাপ নয়। Step timing-এ আমরা GPU transfer ও training update রেখেছি; CPU preprocessing ও validation বাদ।

**Pilot design:** একই base revision, train order, image budget, compute dtype, LR, effective batch ও update budget দিয়ে **unquantized LoRA বনাম final merger** চালিয়েছি। আলাদা চার-update probe-এ একই LoRA target ও batch groups রেখে unquantized বনাম NF4 মেপেছি। ছোট NF4 probe **memory/time check**, accuracy leaderboard নয়। কম memory মানেই বেশি speed নয়।

**Checkpoint selection:** Step 8 ও 16-এ একই validation receipts-এ শুধু image + prompt দিয়ে generate। বেশি all-fields-correct প্রথমে, তারপর বেশি correct fields, তারপর raw JSON validity। পুরো tie হলে আগের checkpoint। Training loss কমলেই validation ভালো হবে না: JSON punctuation শেখা সহজ হতে পারে, কিন্তু একটি digit ভুল হলে ওই receipt-এর total fail।

## ৫. Day 4 — Fair held-out evaluation

Section C-এর train/validation থেকে শেখা ও checkpoint selection করেছি। Section D-এর ৩০টি held-out receipt আগে lock করা ছিল; ID ও exact image hash দিয়ে C-এর train, validation এবং পুরোনো test থেকে overlap বাদ দিয়েছি। Exact hash identical bytes ধরে; resize/crop/re-encode করা near-duplicate ধরতে আলাদা check দরকার হতে পারে।

তিনটি setting: **base**, **base + selected LoRA**, **base + selected saved merger**। একই image IDs, processor, base revision, dtype, image budget, prompt, generation limits ও greedy decoding রেখে score করেছি। যে checkpoint test-এ যাবে, সেটি আগে validation দিয়ে স্থির করেছি।

| Metric | প্রশ্ন |
|---|---|
| Raw JSON validity | কোনো cleanup ছাড়াই parse করা যায়? |
| Schema validity | অনুমোদিত cleanup-এর পরে ঠিক keys ও types আছে? |
| Field correctness | নির্ধারিত normalization-এর পরে company/date/total reference-এর সঙ্গে মেলে? |
| All-fields-correct | একই receipt-এর **তিনটি field-ই** সঠিক? |

`score_doc()` raw parse আলাদা রাখে; আগে অনুমোদিত একটিমাত্র outer Markdown fence সরিয়ে schema check করতে পারে। Schema-তে company/date/total-এর value string বা JSON null হতে পারে। Invalid output denominator-এ fail থাকে। Scorer-এর rules test দেখার আগে স্থির: `"$8.50"` ও `"8.5"` normalize হয়ে সমান হতে পারে; date-এর সঙ্গে অতিরিক্ত time থাকলে আমাদের date rule-এ fail হতে পারে। **Per-field scores-এর গড় all-fields-correct নয়।**

**Paired analysis:** দুই model-এর output receipt **ID** দিয়ে join করো। Base wrong → trained correct = gained; base correct → trained wrong = lost। Net gain = `gained - lost`; percentage-point change = `100 × (gained - lost) / N`। একই মোট score-র আড়ালেও gained আর lost থাকতে পারে। আবার একই `total` accuracy মানেই একই receipts ঠিক পড়েছে, তা নয়।

**আমাদের run থেকে মনে রাখার মতো ঘটনা:**

- Held-out all-fields-correct: **base ১৯/৩০, LoRA ২১/৩০, projector ১৫/৩০**। LoRA: ২ gained, ০ lost। Projector: ২ gained, ৬ lost। সব setting-এ raw JSON ও schema ১০০%; তাই এই ফলের পার্থক্য এসেছে field values থেকে।
- একটি gained case-এ base date-এর সঙ্গে time দিয়েছিল। Trained দুটো model শুধু date দিয়েছে। একটি projector lost case-এ company-এর সঙ্গে brand ও registration number এসেছে। এই দুই উদাহরণ **field boundary/selection** বোঝায়; শুধু final output দেখে সব ভুলকে OCR error বলা যায় না।
- NF4 probe-এ roughly **10.0 → 8.1 GiB**, আর step time বেড়েছিল। এই run-এ memory–speed trade-off দেখা গেছে; সেটি সব hardware-এ একই হবে ধরে নিও না।

**সিদ্ধান্ত:** এই ছোট receipt pilot ও GPU budget-এ LLM LoRA রাখার পক্ষে evidence আছে। ৩০টি receipt-এ দুইটির নেট improvement সব document task-এ একই ranking প্রমাণ করে না।

## ৬. Day 5 — Resume, inference reload ও merge

| Artifact | কী আছে | ব্যবহার |
|---|---|---|
| `resume_step8.pt` | Trainable LoRA state, optimizer, scheduler, scaler, global step ও Torch RNG state | Matching base ও training setup recreate করে step 9 থেকে চালাতে |
| `best_adapter/` | LoRA weights ও config | Matching original base + adapter দিয়ে inference |
| `best_merger.safetensors` | Final merger-এর updated weights | Matching base-এর একই module-এ strict load; আমাদের run-এ merger FP32 |
| `merged_llm_lora/` | LoRA update-সহ পুরো model, সঙ্গে processor | আলাদা adapter attach ছাড়াই model load |

**Resume checklist:** একই base revision ও LoRA targets → একই trainable parameter order → optimizer ও একই total schedule তৈরি → saved training states load → setup-এর পরে RNG restore → recipe থেকে একই batch order → next step ও LR check। Step 8 complete হলে পরের update 9। আমাদের ১৬-step linear schedule-এ expected **step 9 = 0.00005000**, **step 10 = 0.00004375**। শুধু কয়েক step যাচাই করলেও total schedule বদলাবে না। Same settings থাকলেও আলাদা hardware/runtime-এ bit-for-bit replay ধরে নিও না।

**Reload check:** Fresh session-এ LoRA adapter attach করে এবং projector weights সঠিক module-এ বসিয়ে পৃথক prediction নাও। Model load, schema valid, এবং field correct তিনটি আলাদা check। আমাদের projector reload চলেছে, JSON valid ছিল, তবু sample total `8.00` ছিল যেখানে সঠিক `8.50`।

**Merge check:** একই unquantized base-এ selected LoRA load করে আগে predictions রাখো → `merge_and_unload(safe_merge=True)` → full weights ও processor save → disk থেকে নতুন model load → একই receipts, dtype ও decode settings-এ compare। Merge LoRA update base weight-এ বসিয়ে branch সরায়। Standalone export বড়; ছোট adapter base ভাগ করে রাখতে পারে এবং সহজে বদলানো যায়।

**তিনটি আলাদা comparison:** (১) raw strings এক? (২) schema-valid normalized fields এক? (৩) প্রত্যেকটির prediction ground truth-এর সঙ্গে মেলে? একই উত্তর দুটো model-এই ভুল হতে পারে। Low-precision rounding-এর কারণে আলাদা token-ও আসতে পারে। আমাদের তিনটি validation receipt-এ **raw ও field match ৩/৩**, এবং দুটো version-ই সঠিক। এটি **local reload/merge check**, পুরো test set-এর parity benchmark নয়। **Resume-এর step 9/10 log এই আলোচনায় দেখা হয়নি;** workflow ও expected output উপরে রাখা হলো।

## ৭. Notebook map ও দ্রুত debugging

| Notebook | যে functions/outputs খুঁজবে |
|---|---|
| `01_vlm_training_batch.ipynb` | `make_example()` → `render()` → `encode()` → `collate_training()`; held-out lock, image-budget ও label decode |
| `02_training_scope_and_sft.ipynb` | `configure_scope()`, `check_scope()`, `one_update()`, `validate_pilot()`; training recipe, best exports ও resume state |
| `03_evaluation_resume_merge.ipynb` | `load_eval_model()`, `predict_receipt()`, `score_doc()`, paired report, `resume_two_updates()`, merge/reload checks |

Drive root: `/content/drive/MyDrive/MLLM/section_d/`। `data/heldout_manifest.json` test IDs ধরে; `outputs/sft_pilot_v1/recipe.json` train order, base revision, settings ধরে। Held-out outputs `outputs/day4_heldout/`-তে, merged export `outputs/day5/merged_llm_lora/`-তে।

| লক্ষণ | প্রথমে কী দেখব? |
|---|---|
| Image-feature/token mismatch | কয়টি image placeholder, কয়টি actual image, per-image grid ও merged slots; truncation হয়েছে কি না |
| Loss শূন্য/NaN বা model prompt নকল করছে | `labels != -100` count ও decoded targets, prefix equality, তারপর dtype/scaler ও gradient norm |
| Backward হয়েছে, weight বদলায়নি | `requires_grad`, loss dependency, `.grad`, optimizer parameter identity ও actual step |
| ছোট LoRA হলেও OOM | Micro-batch, image slots, sequence length, activation checkpointing ও optimizer peak; তারপর quantization |
| Good validation, poor held-out | Sample size, leakage/duplicates, checkpoint selection, paired failed receipts এবং scorer rules |
| Resume-এ LR আবার শুরুতে | Schedule horizon, scheduler/optimizer state, global step ও batch offset |
| Merge-এর পর fields বদলে গেছে | Base revision, correct adapter, dtype, same processor/prompt/decode; তারপর rounding প্রভাব |

## ৮. Job interview — সহজ ইংরেজিতে বলার practice

**৪০ সেকেন্ডের project answer:**

> I fine-tuned Qwen3-VL-2B to extract company, date, and total from receipts. I verified that only the assistant’s answer tokens contributed to the loss, then compared language-model LoRA with training only the visual merger. I selected checkpoints on validation data and tested all three settings on the same 30 held-out receipts. LoRA improved all-fields correctness from 19 to 21 receipts. I also saved and reloaded a merged LoRA model, and its outputs matched the adapter version on three validation examples.

| Interview question | Short answer |
|---|---|
| **Tokenizer vs multimodal processor?** | The tokenizer maps text to token IDs. The processor also prepares images and matches visual placeholders to the image features expected by the model. |
| **Why do image slots and patch rows have different counts?** | Patch rows are inputs to the vision encoder. Spatial merging combines features from several patches into one embedding for an image slot in the language model. |
| **Can the model see tokens with label minus one hundred?** | Yes. They stay in the input context. That label only tells the loss function to ignore those target positions. |
| **How can a frozen LLM train a projector?** | The LLM weights do not receive updates, but gradients with respect to the LLM input can flow back to the projector. I keep autograd enabled through that path. |
| **How do you prove only the intended weights train?** | I check the trainable names, the exact parameters in the optimizer, and their gradients after a real backward pass. |
| **LoRA vs QLoRA?** | Both train small adapters on a frozen base. QLoRA stores selected base weights in a quantized format. Storage precision and compute precision are separate choices. |
| **Accumulation vs activation checkpointing?** | Accumulation combines gradients before an optimizer update. Activation checkpointing saves memory by recomputing parts of the forward pass during backward. |
| **Why can lower training loss fail to improve extraction?** | Training uses correct previous answer tokens. Inference generates the entire answer, and one wrong digit can fail a field even if the average token loss improved. |
| **How did you compare the methods fairly?** | I kept the base, inputs, prompt, decoding, and scoring rules fixed. I chose checkpoints on validation data and compared the same held-out receipts by ID. |
| **Resume checkpoint vs adapter export?** | A resume checkpoint restores training state such as the optimizer, scheduler, and step. An adapter export contains the learned update for inference with a matching base. |
| **What does a successful merge prove?** | It produces a standalone model with the adapter update folded into its weights. I verify it by reloading from disk and comparing outputs under the same inference settings. |
| **What would you do on a new document task?** | I would check the data split, inspect one real batch and its labels, then run a small LoRA pilot. I would increase image detail or change the training scope only after looking at specific failure cases and memory costs. |

**নিজে মনে করতে পারছ কি না দেখো:** ৬০০ patch rows কেন ১৫০ slots? Frozen LLM দিয়ে merger gradient কীভাবে পায়? Base ১৯, trained ২১, gained ২ হলে lost কত? Valid JSON-এ ভুল total হলে কোন check pass, কোনটি fail? Adapter file থাকলেও আগের step থেকে training চালাতে আর কী কী লাগবে?

আরও পড়তে: [Qwen3-VL implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py) · [TRL v0.29.0 SFT](https://huggingface.co/docs/trl/v0.29.0/en/sft_trainer) · [PEFT LoRA ও merge](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/lora) · [QLoRA paper](https://arxiv.org/abs/2305.14314) · [PyTorch checkpoint guide](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)।
