# Section C — Qwen3-VL: Grounding, QLoRA, Evaluation ও Serving

প্রথমে ছবিতে নির্দিষ্ট object কোথায় আছে তা box দিয়ে বের করেছি। তারপর receipt থেকে `company`, `date`, `total` বের করার কাজ বেছে নিয়েছি। পরিষ্কার split বানিয়ে base model-এর ফল মেপেছি, 4-bit base freeze রেখে LoRA train করেছি, এবং একই test receipt-এ আগে–পরে তুলনা করেছি। শেষে দেখেছি model অনুপস্থিত তথ্য বানায় কি না এবং FastAPI দিয়ে structured response ফেরাতে পারে কি না।

| প্রশ্ন | কোন পরীক্ষা উত্তর দেয়? |
|---|---|
| Model ঠিক object-এ box বসিয়েছে? | Grounding: predicted box বনাম GT box, IoU |
| Receipt-এর তিনটি value ঠিক পড়েছে? | Per-field accuracy এবং all-fields-correct |
| Client output parse করে ব্যবহার করতে পারবে? | Raw JSON parse এবং schema validation |
| ছবিতে তথ্য না থাকলে model থামবে? | Present/absent probes এবং `null` check |
| Image upload থেকে response পাওয়া যাচ্ছে? | Local API request, status, response fields এবং latency |

## ১. Visual grounding: “কী” থেকে “কোথায়”

**Referring phrase** হলো কোন object খুঁজতে হবে তার বর্ণনা। “একটি bowl” আর “অন্য bowl-গুলোর পেছনের আংশিক দেখা bowl” এক কাজ নয়। দ্বিতীয়টিতে একই ধরনের object-এর মধ্যে relation ও অবস্থান দেখে target বেছে নিতে হয়।

আমাদের grounding prompt-এ Qwen-এর box ছিল `[x1, y1, x2, y2]`, normalized **0–1000** scale-এ। এই convention সব model বা dataset-এর জন্য ধরে নেবে না। Dataset GT যদি COCO `[x, y, width, height]` হয়, আগে `[x, y, x + width, y + height]` করো। তারপর দুই box-কে একই coordinate system-এ আনো।

```python
# box: Qwen prediction in 0–1000 coordinates
# W, H: যে original image-এর ওপর box আঁকব তার width ও height
pixel_box = [
    box[0] * W / 1000, box[1] * H / 1000,
    box[2] * W / 1000, box[3] * H / 1000,
]
```

উদাহরণ: `[100, 200, 600, 800]`, image `800 × 600` → `[80, 120, 480, 480]` pixels। **x-এর জন্য width, y-এর জন্য height।** Resize বা orientation বদলালে যে image-এ box আঁকছ, GT ও prediction যেন তার একই frame-এ থাকে।

**IoU = intersection area ÷ union area**, যেখানে union = দুই box-এর area-এর যোগফল থেকে intersection একবার বাদ। GT `[10, 10, 50, 50]` আর prediction `[30, 10, 70, 50]` হলে দুইটির area 1600 করে, overlap 800, IoU = 800/2400 ≈ 0.33। তাই আমাদের **IoU ≥ 0.5** rule-এ fail।

Debug করার ক্রম: **JSON parse → চারটি valid coordinate ও box order → GT/prediction overlay → IoU**। Valid box ভুল bowl ধরতে পারে। আবার target সঠিক হলেও box ঢিলা হলে IoU কমে। Invalid output-ও মোট query-র মধ্যে fail হিসেবে থাকবে; শুধু parse হওয়া prediction দিয়ে success rate হিসাব করলে score ফুলে যায়।

## ২. Receipt task: OCR, extraction ও data split

**OCR** ছবির লেখা পড়ে। **Extraction** নির্দিষ্ট field-এর জন্য কোন লেখাটা দরকার তা বেছে নেয়। Receipt-এ subtotal, tax, cash এবং change সব পড়া গেলেও final payable amount ভুল বেছে নিলে `total` ভুল।

আমাদের model: `Qwen/Qwen3-VL-2B-Instruct`। Dataset: SROIE-এর image ও reference fields। Task: issuer `company`, printed `date`, final `total`। Prompt-এ বলেছি একটিমাত্র JSON object, value string অথবা দৃশ্যমান না হলে `null`, এবং markdown ছাড়া:

```json
{"company": "Example Store", "date": "27/03/2018", "total": "8.50"}
```

**Split-এর নিয়ম:** published train থেকে train/validation; published test থেকে fixed 100-receipt subset। একই receipt-এর ভিন্ন ID থাকলেও content একই হতে পারে। তাই exact duplicate ও train–test overlap image hash দিয়ে ধরেছি। Official test-এর matching images training থেকে বাদ দিয়েছি। Image hash exact copy ধরে; crop বা recompress করা একই receipt ধরার জন্য অতিরিক্ত near-duplicate check দরকার হতে পারে। একই document থেকে একাধিক QA বানালে পুরো document এক split-এ রাখবে।

| Split | ব্যবহার |
|---|---|
| Train | Assistant target দেখে adapter update |
| Validation | Baseline probe, settings এবং checkpoint selection |
| Fixed test | সব সিদ্ধান্তের পরে একবার before/after comparison |

Annotation-এ field খালি মানেই ছবিতে field অনুপস্থিত নয়। ভুল/অসম্পূর্ণ label যাচাই বা বাদ দাও; অন্ধভাবে `null` target বানালে model দৃশ্যমান তথ্যেও abstain করা শিখতে পারে।

**Scoring-এর তিনটি স্তর:**

1. **Raw JSON valid:** model-এর answer সরাসরি `json.loads()`-এ চলে?
2. **Schema valid:** ঠিক তিনটি key এবং প্রতিটি value string বা JSON `null`? আমরা আলাদা করে একটিমাত্র সম্পূর্ণ outer markdown fence সরানোর অনুমতি দিয়েছিলাম। Raw metric আগের মতোই strict ছিল।
3. **Field correct:** valid object-এর value reference-এর সঙ্গে নির্ধারিত normalization-এর পরে মেলে? Invalid parse/schema হলে সব field ভুল; receipt denominator-এ থাকে।

আমাদের normalization-এ Unicode NFKC, casefold ও whitespace collapse ছিল; total-এর জন্য `Decimal` দিয়ে `"8.50"` ও `"8.5"` সমান ধরা যায়। Date-এর printed format নিজের মতো বদলাইনি। `"null"` একটি string; `null` হলো value নেই। Normalization যতটুকু formatting noise সরায় ততটুকুই করো: OCR-এ `0` আর `O` এক করা তথ্য বদলে দেয়। Rules test দেখার **আগে** স্থির করো।

## ৩. Multimodal SFT: image ও answer থেকে loss পর্যন্ত

একটি training example-এ **user: image + instruction**, আর **assistant: reference JSON**। Inference-এ assistant reference থাকে না। Reference শুধুই training target বা generation-এর পরে scorer-এর উত্তরপত্র।

| অংশ | কী করে |
|---|---|
| Messages | Role, image placeholder, instruction ও training answer ধরে |
| Chat template | Model-এর শেখা user/assistant token format-এ conversation সাজায় |
| Processor | Text tokenize করে, image preprocess করে এবং visual placeholders-এর সঙ্গে input মিলিয়ে দেয় |
| Collator | Batch বানায়, padding করে এবং কোথায় loss হবে তা `labels`-এ নির্ধারণ করে |

`input_ids`, `attention_mask` ও `labels` সাধারণত `[B, L]`। Qwen-এর `pixel_values` vision encoder-এর জন্য processed image data; এটি LLM embedding নয়। `image_grid_thw` visual grid-এর shape ধরে। Image budget বড় হলে সাধারণত visual tokens, activation memory ও inference work বাড়ে।

**Assistant-only loss মনে রাখো: context দেখা আর loss পাওয়া আলাদা।**

1. Full training conversation tokenize করো: image + instruction + assistant answer।
2. একই image ও preprocessing দিয়ে শুধু prompt এবং assistant header tokenize করো। এটি full sequence-এর prefix কি না যাচাই করো।
3. Full `input_ids` clone করে `labels` বানাও। Prefix, image placeholders ও padding-এর target `-100` করো।
4. Assistant JSON এবং ending token supervised রাখো। Causal model নিজেই next-token shift করে।

`-100` loss থেকে token বাদ দেয়; input থেকে user prompt সরায় না। `attention_mask`-এর padding rule আর `labels=-100`-এর loss rule এক নয়। Sanity check: **`labels != -100` tokens decode করলে শুধু assistant answer ও ending token দেখা উচিত।**

**Teacher forcing:** training-এ পরের answer token predict করার সময় model আগের সঠিক answer tokens পায়। Inference-এ নিজের generated tokens-ই পরের context। তাই token loss কমা useful signal, কিন্তু unseen receipt-এ পুরো JSON সঠিক হওয়ার প্রমাণ নয়।

## ৪. Quantization: কোথায় memory বাঁচে?

**Weight storage dtype**, **compute dtype**, আর **trainable adapter dtype** আলাদা সিদ্ধান্ত। আমাদের setup-এ language linear weights NF4-এ, 4-bit operations-এর compute FP16, adapters FP32 parameters। সব tensor 4-bit হয়নি: vision অংশ frozen FP16 ছিল, আর PEFT preparation কিছু unquantized tensors FP32-এ রাখে।

- **NF4** 4-bit-এ weight values represent করে। প্রয়োজনে kernel quantized weights decode করে compute dtype-এ arithmetic চালায়; এতে পুরো model স্থায়ীভাবে FP16 করতে হয় না।
- **Double quantization** quantization-এর scale/constant-গুলোকেও compress করে metadata memory কমায়।
- **Peak VRAM শুধু weights নয়।** Vision/text activations, inference KV cache, training gradients/optimizer state এবং temporary buffers-ও লাগে। PyTorch peak allocated memory পুরো GPU process memory নয়।
- **4-bit সবসময় দ্রুত নয়।** Dequantization ও kernel overhead, hardware এবং generated tokens-এর সংখ্যা latency বদলায়। তাই memory ও latency আলাদা মাপো।

Day 3-এর controlled pairs: **2B FP16 → 2B NF4** দিয়ে precision-এর প্রভাব; **2B NF4 → 4B NF4** দিয়ে model size-এর প্রভাব। একই sample, image budget ও generation settings রাখলে তুলনা বোঝা যায়। ২০টির probe-তে ৫ percentage points মানে মাত্র ১টি receipt; এটি configuration বাছার ইঙ্গিত।

## ৫. QLoRA: base freeze, ছোট update শেখা

**LoRA** একটি linear layer-এর frozen weight `W₀`-এর পাশে ছোট matrices `A` ও `B` শেখায়। Input `x`-এর জন্য:

```text
y = W₀x + (alpha / rank) × B(Ax)
```

`A` input dimension থেকে rank-এ নামায়, `B` rank থেকে output dimension-এ তোলে। পূর্ণ weight matrix বদলানোর বদলে এই ছোট branch শেখে। **Rank** update-এর capacity; **alpha/rank** এই setup-এ তার scale। আমাদের rank 8, alpha 16। **QLoRA** হলো quantized frozen base-এর ওপর এই LoRA training।

আমরা language model-এর quantized linear layers target করেছি; vision encoder ও base weights frozen ছিল। Frozen মানে ওই weights update হবে না। Frozen operation-এর মধ্য দিয়েও gradient আগের trainable অংশে পৌঁছাতে পারে। তাই পুরো training forward-এ `torch.no_grad()` দিলে adapter-এর শেখা বন্ধ হবে।

**Training code-এর কাজের ক্রম:**

1. NF4 base load → `prepare_model_for_kbit_training()` → `get_peft_model()` দিয়ে LoRA যোগ।
2. Trainable parameter names দেখো: শুধু `lora_A`/`lora_B`। এক training sample-এ forward/backward চালিয়ে finite loss, কিছু finite nonzero adapter gradient এবং frozen weights-এ gradient না থাকা দেখো। Initial LoRA `B` zero হলে কিছু `A` gradient প্রথম ধাপে zero হতে পারে।
3. Trainer দিয়ে SFT চালাও। Micro-batch 1, accumulation 8 মানে সাধারণত ৮টি sample-এর forward/backward আলাদা করে করে gradients জমিয়ে এক optimizer step। শেষ group ছোট হতে পারে। একসঙ্গে ৮টি image-এর activation রাখতে হয় না।
4. **Gradient checkpointing** কিছু activations রেখে বাকিগুলো backward-এ আবার হিসাব করে। VRAM কম লাগে, সময় বাড়তে পারে। এটি disk-এ saved checkpoint থেকে আলাদা।
5. প্রতি epoch শেষে validation image + prompt দিয়ে answer **generate** করে score করো। বেশি all-fields-correct আগে; tie হলে বেশি correct fields, তারপর raw JSON validity। সব সমান হলে আগের epoch রাখো।

**দুই ধরনের save:**

| Save | পরে কী কাজে লাগে |
|---|---|
| Trainer checkpoint | Training resume: adapter ও optimizer/scheduler/RNG state |
| Best adapter + processor/config | Inference: matching base revision load করে `PeftModel.from_pretrained()` দিয়ে adapter attach |

Adapter file পুরো model নয়। Matching base, processor, prompt ও image settings লিখে রাখো এবং নতুন session-এ reload করে একটি prediction দেখো। Inference-এর জন্য adapter merge বাধ্যতামূলক নয়। Training smoke-step-এর memory-তে optimizer state তখনো না-ও থাকতে পারে।

## ৬. Fair evaluation: “কত বাড়ল?” এবং “কোথায় বাড়ল?”

একই test IDs, base revision, quantization ও dtype policy, processor, prompt, image budget, token cap ও decoding রাখো। **যেটা বদলাবে: adapter on/off।** অন্য precision-এর পুরোনো baseline টেনে causal comparison করো না।

- **Per-field accuracy:** company/date/total আলাদাভাবে কত receipt-এ ঠিক।
- **All-fields-correct:** একটি receipt-এর তিনটিই ঠিক হলে এক success। তিনটি field accuracy-র গড় এ metric নয়।
- **Usability:** raw parse ও schema validity আলাদা। JSON সুন্দর হলেও value ভুল হতে পারে।
- **Resource cost:** একই warm-up ও hardware; GPU timing-এর আগে/পরে synchronize। আমাদের latency preprocessing, generation ও decode ধরে, Drive থেকে image read ও scoring বাদ দেয়।

`merge(..., on="id")` দিয়ে একই receipt-এর দুই prediction পাশাপাশি আনো:

| Base | Adapted | Case |
|---|---|---|
| Correct | Correct | Both correct |
| Wrong | Correct | Gained |
| Correct | Wrong | Lost |
| Wrong | Wrong | Both wrong |

**Net change in percentage points = `100 × (gained − lost) / N`**। উদাহরণ: ১০০টির মধ্যে ১৫টি ঠিক হলো আর ১টিতে নতুন ভুল → **+১৪ points**। একটি lost case-এ company-এর `SDN.BHD.` বনাম `SDN. BHD.` spacing difference দেখা গেছে। এমন case raw answer ও reference খুলে দেখো। Test দেখার পরে scorer বদলে পুরোনো test-কে fresh evidence হিসেবে ব্যবহার কোরো না।

General VQA-র কয়েকটি পুরোনো image/question reuse করলে সম্ভাব্য regression ধরা যায়। আমাদের ৫টি প্রশ্ন ছিল ৩টি ছবির ওপর; সেখানে একই ফল মানে **ওই ছোট check-এ regression দেখা যায়নি**।

## ৭. Hallucination ও abstention: valid JSON-এও মিথ্যা value থাকতে পারে

একটি question set-এ **present** এবং **absent** দুটোই দরকার। সব উত্তর `no` হলে absent-এ ঠিক দেখাবে, কিন্তু present objects miss হবে। আমাদের চারটি object probe-তে দুই model-ই সঠিক ছিল। আলাদা probe-তে receipt না থাকা flower image দিয়েও দুই model-ই company/date ইত্যাদি বানিয়েছে। **Receipt extraction-এ উন্নতি মানেই কখন উত্তর দেওয়া উচিত নয় সেটি শেখা নয়।**

| অবস্থা | কীভাবে চেনা যায় |
|---|---|
| Unsupported answer | Image-এ dog নেই, তবু “yes”; non-receipt image থেকে invented company |
| Miss / field error | Evidence আছে, কিন্তু model object খুঁজে পায়নি বা দৃশ্যমান value ভুল পড়েছে |
| Correct abstention | তথ্য সত্যিই নেই, তাই field `null` |
| Unnecessary abstention | তথ্য দৃশ্যমান, তবু field `null` |
| Format failure | Raw parse বা schema ব্যর্থ; value quality যাচাইয়ের আগে format check দরকার |

আমাদের flower probe **non-receipt input-এ আচরণ** পরীক্ষা করে। একটি real receipt-এর শুধু total অনুপস্থিত হলে কী হবে, সেটি আলাদা প্রশ্ন। Masked/cropped receipt দিয়ে পরীক্ষা করলে সেটিকে আলাদা robustness case হিসেবে রাখবে। Production-এ non-receipt input চেনা বা uncertain extraction review-এ পাঠানো পরবর্তী উন্নতির ভালো জায়গা।

## ৮. Serving: notebook prediction থেকে client response

`POST /extract` multipart request-এ **image + optional instruction** নেয়। `lifespan` startup-এ matching base ও adapter একবার load করে। Endpoint request-এ file bytes ও image dimensions check করে, RGB-তে decode করে, আগের processor ও generation settings দিয়ে inference চালায়, তারপর **strict raw JSON + schema check** করে।

| Status | এই API-তে অর্থ |
|---|---|
| `200` | Structured fields পাওয়া গেছে। Field সত্যি সঠিক কি না আলাদা প্রশ্ন। |
| `400` | Upload আছে, কিন্তু image decode করা যাচ্ছে না। |
| `413` | App-এর file-byte বা pixel limit ছাড়িয়েছে। |
| `422` | Required request field নেই বা request validation fail। |
| `502` | Image গ্রহণ হয়েছে, কিন্তু model valid raw JSON/schema দেয়নি। |

GPU `Lock` একই process-এ একবারে একটি generation চালায়। Concurrent request এলে অপেক্ষা বাড়তে পারে। FastAPI request/response ও validation সামলায়; Transformers generation চালায়। vLLM-এর মতো inference engine আলাদা করে request scheduling ও batching সামলাতে পারে। **Local `TestClient`** দিয়ে আমরা multipart → inference → response path চালিয়েছি। এর timing public network বা production throughput-এর মাপ নয়।

**Debug কোথা থেকে শুরু করবে?** `400` → bytes ও image decode। `502` → raw model answer, JSON fences, keys/types। `200` কিন্তু field ভুল → image crop, prompt, prediction ও reference পাশাপাশি দেখো। High latency → image tokens, generated tokens, queue wait ও inference time আলাদা করো।

## ৯. Notebook-এ ফিরে গেলে code map

| Notebook / কাজ | খুঁজবে | মনে করার প্রশ্ন |
|---|---|---|
| `01_grounding_document_baseline.ipynb` | Box parse, pixel conversion, overlay, IoU; data audit ও manifest | Pred ও GT একই image frame-এ? Train/test overlap আছে? |
| `02_quantization_qlora.ipynb` | `load_base()`, `collate_receipts()`, `SelectAdapter` | কোন layers quantized? কোন labels supervised? কোন checkpoint best এবং কেন? |
| `03_evaluation_comparison.ipynb` | `answer_image()`, `score_doc()`, `evaluate_receipts()`, paired join, `run_probes()`, `/extract` | একই test IDs? Valid হলেও ভুল? Missing evidence-এ কী করছে? |

`manifest.json` split ও image IDs ধরে; `task_config.json` prompt, fields ও image/generation rules ধরে। `outputs/qlora_run1/best_adapter/`-এ selected adapter, processor ও training recipe আছে। Colab/Kaggle বদলালে actual root path দেখো; task settings ও split একই রাখো।

**Run থেকে মনে রাখার মতো evidence:** প্রায় **০.৪১%** parameters trainable ছিল। Fixed ১০০ receipt-এ all-fields-correct **৬৫% → ৭৯%** এবং strict raw JSON **০% → ১০০%**; cleanup-পরবর্তী schema validity দুইটিতেই ১০০%। Adapted run-এ mean latency বেড়েছিল। Final local API test-এ real receipt **200 এবং তিনটি field সঠিক**, broken upload **400**। Non-receipt hallucination রয়ে গেছে।

## ১০. Job interview — সহজ ইংরেজিতে বলার practice

**৪০ সেকেন্ডের project answer:**

> I fine-tuned Qwen3-VL-2B to extract the company, date, and total from receipt images. I cleaned the data splits and kept a fixed test set. I used QLoRA to train small adapters on a frozen 4-bit base model. On the same 100 test receipts, all-fields accuracy increased from 65 to 79 percent, and raw JSON became much more reliable. I tested the adapted model through a local FastAPI endpoint. One limitation was that it still invented fields when the input was not a receipt.

| Interview question | Short answer |
|---|---|
| **How is grounding different from VQA?** | VQA answers a question about the image. Grounding points to the region described by a phrase. I measured the predicted box with IoU. |
| **How did you prevent data leakage?** | I checked image content hashes as well as IDs. I removed exact overlaps and kept the test set separate from checkpoint selection. |
| **What is assistant-only loss?** | The model can see the image and prompt, but I compute loss only on assistant answer tokens. I mask the other labels with minus one hundred. |
| **What changes in QLoRA?** | The base model stays frozen, and its selected linear weights are stored in 4-bit. Only the small LoRA matrices learn. The forward pass still uses the base model. |
| **Does 4-bit always make inference faster?** | No. It reduces weight memory, but decoding the quantized weights and running the kernels can add work. I measure speed and memory separately. |
| **Accumulation versus gradient checkpointing?** | Accumulation combines gradients from multiple small batches before an update. Checkpointing saves activation memory by recomputing part of the forward pass. |
| **Why not pick the lowest training loss?** | Training loss predicts the next token with the correct previous tokens. I select the checkpoint by generated answer quality on validation receipts. |
| **How did you make the comparison fair?** | I used the same base revision, precision, images, prompt, and decoding. I paired results by receipt ID to see gains and losses. |
| **How can valid JSON be wrong?** | The keys and types can be correct while the values are wrong or invented. I checked format, field correctness, and abstention separately. |
| **What would you improve next?** | I would test more cases where information is missing, especially images that are not receipts, and check whether the model can abstain without missing visible fields. |

**Notebook না খুলে নিজেকে পরীক্ষা করো:** (১) normalized box আঁকতে x ও y-কে কী দিয়ে scale করবে? (২) `-100` দিলে user prompt কি model দেখতে পায়? (৩) ১৫ gained, ১ lost, ১০০ test হলে net gain কত? (৪) flower image থেকে valid receipt JSON এলে সেটা format success হলেও কোন quality failure? (৫) `200` আর all-fields-correct কেন আলাদা?

যদি আটকে যাও, উপরের সংশ্লিষ্ট অংশে ফিরে যাও। সবচেয়ে কাজে লাগবে এই অভ্যাস: **input ঠিক কি না → output parse হয় কি না → ছবির evidence-এর সঙ্গে মেলে কি না → benchmark fair কি না**।

আরও পড়তে: [Qwen3-VL official repo](https://github.com/QwenLM/Qwen3-VL) · [SROIE dataset used here](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE) · [QLoRA paper](https://arxiv.org/abs/2305.14314) · [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization) · [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/) · [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/).
