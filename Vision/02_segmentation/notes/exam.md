## Section A — Concepts

### Q1 — Semantic এবং instance segmentation-এর পার্থক্য কী? একই image-এ তিনটি আলাদা crack থাকলে U-Net ও YOLO26-seg কী output দেবে?

**উত্তর:**

Semantic segmentation ছবির প্রতিটি pixel-কে একটি class দেয়। একই class-এর আলাদা object থাকলেও তাদের পৃথক পরিচয় সংরক্ষণ করে না।

Instance segmentation প্রতিটি object-কে আলাদা instance হিসেবে শনাক্ত করে। প্রতিটি instance-এর জন্য পৃথক mask, bounding box, class এবং confidence পাওয়া যায়।

একটি ছবিতে তিনটি crack থাকলে:

- **U-Net:** তিনটি crack-এর সব foreground pixel একত্রে একটি binary semantic mask-এ দেখাবে। Output সাধারণত `[B, 1, H, W]` logits, যা sigmoid ও threshold-এর পরে `0 = background`, `1 = crack` mask হবে। কোনটি crack-1, crack-2 বা crack-3—এই পরিচয় থাকবে না।
- **YOLO26-seg:** তিনটি crack সঠিকভাবে শনাক্ত হলে তিনটি পৃথক mask, তিনটি bounding box, তিনটি confidence score এবং তিনটি class ID দেবে।

---

### Q2 — SAM2/SAM3 এবং trained U-Net/YOLO-seg-এর production role কেন আলাদা?

**উত্তর:**

**SAM2** একটি class-agnostic promptable segmentation model। এটি point, box বা mask prompt দিয়ে নির্দেশ করা target segment করে। এটি নিজে থেকে target-এর semantic class নিশ্চিত করে না।

**SAM3** text অথবা exemplar prompt থেকে একটি concept খুঁজে তার matching instances segment করতে পারে। এটি SAM2-এর চেয়ে semantic concept সম্পর্কে বেশি সচেতন, কিন্তু domain-specific trained detector-এর মতো নির্দিষ্ট production class-এর জন্য calibrated নয়।

SAM2/SAM3 উপযুক্ত:

- দ্রুত annotation draft তৈরি
- human-in-the-loop mask refinement
- interactive segmentation
- নতুন বা পরিবর্তনশীল concept নিয়ে zero-shot experiment

অন্যদিকে, trained U-Net এবং YOLO-seg নির্দিষ্ট domain-এর annotated data থেকে শেখে।

- **U-Net:** কোনো human prompt ছাড়াই automatic semantic mask তৈরি করে।
- **YOLO-seg:** automatic instance mask, box, class ও confidence তৈরি করে।

তাই fixed-domain production inference-এর জন্য trained U-Net/YOLO-seg বেশি উপযুক্ত। SAM সাধারণত annotation assistance বা interactive workflow-তে বেশি উপযোগী।

---

## Section B — Math and Metrics

### Q3 — নিচের মান থেকে Precision, Recall, IoU, Dice এবং Pixel Accuracy হিসাব করো।

```text
TP = 600
FP = 150
FN = 250
TN = 9000
```

**উত্তর:**

#### Precision

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{600}{600 + 150} = 0.8000$$

#### Recall

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{600}{600 + 250} = 0.7059$$

#### IoU

$$\text{IoU} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}} = \frac{600}{600 + 150 + 250} = 0.6000$$

#### Dice

$$\text{Dice} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}} = \frac{1200}{1200 + 150 + 250} = 0.7500$$

#### Pixel Accuracy

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} = \frac{600 + 9000}{10000} = 0.9600$$

Final results:

```text
Precision      = 0.8000
Recall         = 0.7059
IoU            = 0.6000
Dice           = 0.7500
Pixel Accuracy = 0.9600

```

---

### Q4 — Instance evaluation-এর নিচের তথ্য থেকে FP, FN, Precision, Recall এবং F1 হিসাব করো।

```text
GT instances        = 20
Predicted instances = 18
Matched TP          = 14

```

**উত্তর:**

#### False Positive

$$\text{FP} = 18 - 14 = 4$$

#### False Negative

$$\text{FN} = 20 - 14 = 6$$

#### Precision

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{14}{18} = 0.7778$$

#### Recall

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{14}{20} = 0.7000$$

#### F1-score

$$\text{F1} = \frac{2 \cdot \text{P} \cdot \text{R}}{\text{P} + \text{R}} = \frac{2(0.7778)(0.7000)}{0.7778 + 0.7000} = 0.7368$$

Final results:

```text
TP        = 14
FP        = 4
FN        = 6
Precision = 0.7778
Recall    = 0.7000
F1        = 0.7368

```

---

### Q4 — Instance evaluation-এর নিচের তথ্য থেকে FP, FN, Precision, Recall এবং F1 হিসাব করো।

```text
GT instances        = 20
Predicted instances = 18
Matched TP          = 14
```

**উত্তর:**

#### False Positive

$$\text{FP} = 18 - 14 = 4$$

#### False Negative

$$\text{FN} = 20 - 14 = 6$$

#### Precision

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{14}{18} = 0.7778$$

#### Recall

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{14}{20} = 0.7000$$

#### F1-score

$$\text{F1} = \frac{2 \cdot \text{P} \cdot \text{R}}{\text{P} + \text{R}} = \frac{2(0.7778)(0.7000)}{0.7778 + 0.7000} = 0.7368$$

Final results:

```text
TP        = 14
FP        = 4
FN        = 6
Precision = 0.7778
Recall    = 0.7000
F1        = 0.7368
```

---

## Section C — Code and Debugging

### Q5 — নিচের code-এর সমস্যা কী এবং কীভাবে ঠিক করবে?

```python
probabilities = torch.sigmoid(
    model(images)
)

loss = torch.nn.BCEWithLogitsLoss()(
    probabilities,
    masks,
)
```

**উত্তর:**

`BCEWithLogitsLoss` input হিসেবে raw logits প্রত্যাশা করে। এটি internally numerically stable পদ্ধতিতে sigmoid এবং binary cross-entropy একসঙ্গে হিসাব করে।

কিন্তু code-এ আগে sigmoid প্রয়োগ করে probability দেওয়া হয়েছে। ফলে loss function probability-কে logits হিসেবে ধরে আবার sigmoid-জাতীয় transformation প্রয়োগ করবে। এতে:

- training objective ভুল হয়ে যায়
- gradient দুর্বল বা compressed হতে পারে
- `BCEWithLogitsLoss`-এর numerical-stability সুবিধা নষ্ট হয়

Correct code:

```python
logits = model(images)

criterion = (
    torch.nn.BCEWithLogitsLoss()
)

loss = criterion(
    logits,
    masks,
)
```

Probability শুধু evaluation, thresholding বা visualization-এর সময় তৈরি করতে হবে:

```python
probabilities = torch.sigmoid(
    logits
)

predicted_masks = (
    probabilities >= 0.45
)
```

---

### Q6 — নিচের resizing code কেন ভুল? Correct order লিখো।

```python
class_mask = logits.argmax(
    dim=1
)

class_mask = F.interpolate(
    class_mask.float(),
    size=(512, 512),
    mode="bilinear",
)
```

**উত্তর:**

এখানে দুটি সমস্যা আছে।

প্রথমত, `argmax(dim=1)` করার পরে tensor-এর shape `[B, H, W]` হয়ে যায়। কিন্তু 2D bilinear resizing-এর জন্য সাধারণ input shape `[B, C, H, W]` প্রয়োজন।

দ্বিতীয়ত, `argmax`-এর পরে tensor-এ discrete class ID থাকে। Bilinear interpolation করলে class ID-এর মধ্যে fractional value তৈরি হতে পারে, যা valid class নয়।

Correct order:

1. Continuous logits-কে bilinear interpolation দিয়ে resize করতে হবে।
2. Resized logits-এর ওপর `argmax` করতে হবে।

```python
resized_logits = F.interpolate(
    logits,
    size=(512, 512),
    mode="bilinear",
    align_corners=False,
)

class_mask = resized_logits.argmax(
    dim=1
)
```

যদি আগে থেকেই তৈরি করা ground-truth বা predicted class-ID mask resize করতে হয়, তখন nearest-neighbour ব্যবহার করতে হবে:

```python
resized_mask = F.interpolate(
    class_mask.unsqueeze(1).float(),
    size=(512, 512),
    mode="nearest",
).squeeze(1).long()
```

---

## Section D — Failure Analysis

### Q7 — Pixel accuracy `0.991`, কিন্তু Dice `0.764`। Pixel accuracy বেশি হলেও model perfect নয় কেন?

**উত্তর:**

Crack segmentation-এ প্রায় 95–99% pixel background এবং খুব অল্প pixel crack হতে পারে। এ কারণে dataset-এ severe foreground-background imbalance থাকে।

Model যদি প্রায় সব pixel-কে background বলে, তাহলেও অধিকাংশ background pixel সঠিক হওয়ায় pixel accuracy খুব বেশি হতে পারে। কিন্তু model crack pixel miss করবে।

Pixel accuracy-তে প্রচুর true negative প্রাধান্য পায়:

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

অন্যদিকে Dice foreground crack overlap-এর ওপর গুরুত্ব দেয়:

$$\text{Dice} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$$

তাই:

```text
Pixel accuracy = 0.991
Dice           = 0.764
```

এর অর্থ background classification খুব ভালো হলেও crack mask-এ এখনও false positive ও false negative আছে। Crack segmentation-এর জন্য Dice, IoU, precision এবং recall pixel accuracy-এর চেয়ে বেশি informative।

---

### Q8 — নিচের result কী ধরনের weakness নির্দেশ করে? অন্তত তিনটি investigation বা improvement বলো।

```text
Box mAP75  = 0.466
Mask mAP75 = 0.041
```

**উত্তর:**

Box mAP75 তুলনামূলক ভালো হওয়ায় model crack-এর আনুমানিক location ও bounding box ধরতে পারছে।

কিন্তু Mask mAP75 অত্যন্ত কম। অর্থাৎ predicted mask-এর precise shape, thin boundary এবং ground-truth polygon-এর সঙ্গে high-IoU overlap দুর্বল।

প্রথমে investigation:

1. Ground-truth polygon thin crack boundary ঠিকভাবে অনুসরণ করছে কি না এবং annotation অতিরিক্ত মোটা, অসম্পূর্ণ বা fragmented কি না পরীক্ষা করা।
2. Original image, resized ground truth ও predicted mask একসঙ্গে দেখে resizing বা rasterization-এর কারণে thin crack হারাচ্ছে কি না পরীক্ষা করা।
3. Small/thin instance-এ failure বেশি কি না size-wise mask performance পরীক্ষা করা।
4. Correct box but poor mask cases আলাদাভাবে দেখা।
5. একটানা visible crack dataset-এ একাধিক GT instance হিসেবে ভাগ হয়েছে কি না পরীক্ষা করা।

সম্ভাব্য improvements:

- Higher input resolution বা tiled inference ব্যবহার
- Mask/prototype resolution বাড়ানো
- Thin-crack ও low-contrast sample বৃদ্ধি করা
- Geometry-preserving augmentation ব্যবহার
- Annotation quality উন্নত করা
- প্রয়োজনে larger segmentation model পরীক্ষা করা
- উপরোক্ত সমস্যা যাচাইয়ের পরে mask-loss configuration পরিবর্তন করা

Blindly loss weight বাড়ানোর আগে annotation এবং resolution পরীক্ষা করা উচিত।

---

## Section E — System Design and Viva

### Q9 — নিচের দুই requirement-এর জন্য কোন system ব্যবহার করবে?

```text
A. প্রতিটি image-এ মোট crack area দরকার
B. প্রতিটি আলাদা crack-এর mask, box ও confidence দরকার
```

**উত্তর:**

### Requirement A — মোট crack area

**নির্বাচন: U-Net**

কারণ U-Net একটি merged semantic crack mask তৈরি করে। সেই binary mask-এর foreground pixel count থেকে total crack area বের করা যায়।

```python
crack_pixels = (
    predicted_mask == 1
).sum()

crack_ratio = (
    crack_pixels
    / predicted_mask.size
)
```

এখানে পৃথক instance identity প্রয়োজন নেই।

### Requirement B — পৃথক crack mask, box ও confidence

**নির্বাচন: YOLO26-seg**

কারণ এটি প্রতিটি detected instance-এর জন্য দেয়:

```text
Mask
Bounding box
Class ID
Confidence score
```

### SAM কখন ব্যবহার করব?

SAM2/SAM3 interactive segmentation বা annotation assistance-এর জন্য ব্যবহার করা যায়। Fixed-domain automatic production inference-এর primary model হিসেবে এটি এই দুই requirement-এর প্রথম পছন্দ নয়।

---

### Q10 — নিচের CPU deployment result থেকে format এবং system নির্বাচন করো।

```text
U-Net:
PyTorch = 648 ms
ONNX    = 452 ms

YOLO-seg:
PyTorch = 167 ms
ONNX    = 144 ms
```

**উত্তর:**

### Format selection

CPU deployment-এর জন্য **ONNX Runtime** নির্বাচন করব, কারণ দুটি model-এর ক্ষেত্রেই ONNX দ্রুত এবং parity পরীক্ষায় accuracy loss পাওয়া যায়নি।

```text
U-Net ONNX parity difference = 0.0
YOLO ONNX parity difference  = 0.0
```

### System selection

System নির্বাচন application requirement-এর ওপর নির্ভর করবে।

- মোট crack area বা merged semantic map প্রয়োজন হলে: **U-Net ONNX**
- পৃথক crack mask, box, count ও confidence প্রয়োজন হলে: **YOLO26-seg ONNX**

যদি production requirement পৃথক crack inspection হয়, তাহলে YOLO26-seg ONNX নির্বাচন করব।

### Latency

```text
U-Net ONNX    = 452 ms ≈ 2.21 FPS
YOLO-seg ONNX = 144 ms ≈ 6.93 FPS
```

YOLO-seg ONNX দ্রুত, কিন্তু `6.93 FPS` সাধারণ `25–30 FPS` video target অনুযায়ী real-time বা near-real-time নয়। এটিকে low-frame-rate inference বলা উচিত।

আরও দ্রুত করতে প্রয়োজন হতে পারে:

- frame skipping
- smaller input resolution
- region-of-interest processing
- asynchronous decode/inference
- hardware acceleration
- OpenVINO বা TensorRT
- faster target CPU/GPU

### Accuracy purpose

U-Net-এর test Dice প্রায় `0.764`, তাই merged crack-pixel map-এর জন্য এটি বেশি উপযোগী।

YOLO26-seg-এর test mask mAP50–95 প্রায় `0.179` এবং mask mAP75 প্রায় `0.041`; তাই instance output পাওয়া গেলেও precise boundary একটি গুরুত্বপূর্ণ limitation।

### Production monitoring

অন্তত নিচের তথ্য log করব:

- preprocessing, inference ও postprocessing latency
- effective FPS এবং dropped-frame count
- detected instance count
- confidence distribution
- predicted crack-pixel ratio বা mask area
- input brightness, blur ও resolution
- CPU/RAM/GPU utilization
- empty অথবা unusually large mask rate
- runtime error ও failed-frame count
- data-drift indicators

**Final decision:**

```text
Format:
ONNX Runtime

Semantic area system:
U-Net ONNX

Separate-instance system:
YOLO26-seg ONNX

CPU real-time status:
Neither system is currently real-time
```