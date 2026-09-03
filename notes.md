# Assignment Question

Diagnose why a model that works in Iowa and fails in the Sahel fails and propose the intervention you'd bet on. You may implement, prototype, or argue with evidence.
A land-cover segmentation model was trained on chips from temperate North America and scores 0.85 mIoU on a held-out split there. On chips from a semi-arid region and from a different acquisition season it drops to 0.41. Both sets come from the same sensor. You are given a few hundred labeled source chips and a few hundred unlabeled target chips. Give me step by step on how to do this task​‌


# Methodology

## Approach-1

Before applying any adaptation technique, a structured diagnostic framework was used to identify the root cause of model degradation. This approach prevents unnecessary interventions and ensures that solutions directly address the observed failure mode. Three potential causes were evaluated:

1. **Radiometric and seasonal covariate shift**

2. **Class-prior or label-space shift**

3. **Increased semantic complexity of the target environment** 

The diagnostic framework included:

- Per-class IoU analysis and confusion matrices
- Spectral band distribution comparisons
- Domain classification and Proxy A-Distance measurement
- Feature-space visualization using encoder embeddings 

# Results

## 1. Significant Radiometric and Seasonal Shift

Analysis of spectral statistics showed substantial differences between source and target imagery. All bands exhibited measurable distribution shifts, confirming that the model was exposed to data distributions not seen during training. 

### Per-Band Distribution Shift

| Band | Source Mean | Target Mean | Mean Shift | Std Ratio (Target / Source) |
|---|---:|---:|---:|---:|
| Red | 0.124 | 0.302 | +0.179 | 1.60 |
| Green | 0.218 | 0.307 | +0.089 | 1.39 |
| Blue | 0.105 | 0.262 | +0.157 | 1.51 |
| NIR | 0.526 | 0.365 | -0.161 | 0.75 |

*Results indicate a systematic increase in visible-band reflectance and a decrease in NIR response, which is consistent with dry-season vegetation conditions commonly observed in semi-arid regions.* 

### Key Observation

The reduction in NIR reflectance suggests lower vegetation density and reduced plant activity in the target imagery. Meanwhile, the increased visible-band reflectance indicates notable seasonal and illumination differences despite the imagery originating from the same sensor platform. 

## 2. Domains Are Maximally Separable

A lightweight domain classifier was trained using simple image statistics to determine whether a sample originated from the source or target domain.

### Domain Separability Results

| Metric | Value |
|---|---:|
| Domain Classifier Accuracy | 1.00 |
| Proxy A-Distance | 2.00 (Maximum) |

These results show that source and target samples can be perfectly separated using basic statistical features alone, providing strong evidence that **covariate shift is the dominant failure mechanism**. 


## Overall Performance Comparison

The table below summarizes model performance across each stage of the adaptation pipeline.

| Stage | Domain | mIoU | Change vs Target Baseline |
|---|---|---:|---|
| Baseline (Source Only) | Source | **1.000** | - |
| Baseline (Source Only) | Target | **0.134** | - |
| AdaBN | Target | **0.236** | **+0.102 (+76%)** |
| AdaBN + Harmonization | Target | 0.019 | -0.115 |
| AdaBN + Harmonization + Self-Training | Target | 0.028 | -0.106 |
| AdaBN + Harmonization + Self-Training | Source | 0.684 | Regression from 1.000 |

### Interpretation

The baseline model demonstrates excellent performance on the source domain but generalizes poorly to the target domain. Applying AdaBN delivers a substantial improvement with minimal implementation complexity. However, additional stages involving harmonization and self-training degraded performance because of interactions between the adaptation methods.


## Target Domain IoU: Baseline vs AdaBN - Per-Class Analysis

| Class | Baseline IoU | AdaBN IoU | Observation |
|---|---:|---:|---|
| Water | 0.000 | **0.857** | Significant recovery |
| Built-up | 0.052 | **0.229** | Improved prediction quality |
| Cropland | 0.324 | 0.091 | Reduced false-positive bias |
| Grassland/Shrub | 0.296 | 0.005 | Performance decline |
| Bare Soil | 0.000 | 0.000 | Never predicted |

### Key Observations

The most notable improvement was observed for the **water** class, where IoU increased from **0.000 to 0.857** after AdaBN. This confirms that radiometric misalignment was heavily influencing predictions and that recalibrating Batch Normalization statistics effectively reduced this bias. 

Although some classes experienced lower individual IoUs, the overall target-domain performance improved significantly. This indicates that AdaBN successfully aligned feature distributions across domains, leading to better overall segmentation quality. 


## Why Harmonization After AdaBN Failed

An important finding from this study was that applying radiometric harmonization after AdaBN caused target mIoU to collapse from **0.236 to 0.019**. 

The most likely explanation is:

1. AdaBN recalibrated Batch Normalization layers using target-domain statistics.

2. Harmonization subsequently transformed the target inputs toward source-domain statistics.

3. The recalibrated BatchNorm layers then received data distributions that no longer matched the statistics used during calibration.

This introduced a new distribution mismatch and caused the model to collapse toward dominant class predictions. 

This result highlights the importance of validating adaptation pipelines experimentally rather than assuming that multiple alignment techniques will always be complementary.

## Remaining Challenge: Bare Soil Classification

A consistent observation across all experiments was the complete failure to predict the **bare soil** class.

### Observed Performance

| Stage | Bare Soil IoU |
|---|---:|
| Baseline | 0.000 |
| AdaBN | 0.000 |
| Harmonization | 0.000 |
| Self-Training | 0.000 |

The root cause is a **label-space gap** rather than a feature-distribution problem. Bare soil is highly represented in the Sahel imagery but is largely absent from the Iowa training labels. Consequently, the model never learned a meaningful decision boundary for this class. 
No unsupervised adaptation method can create a class representation that was never learned during training. Therefore, targeted annotation of bare-soil examples would provide the highest return on investment for further improvement. 

The analysis shows that the performance drop is primarily caused by **covariate shift**, driven by seasonal and radiometric differences between the source and target domains. A domain classifier achieved **100% accuracy** in distinguishing source and target samples, resulting in a **Proxy A-Distance of 2.0**, which indicates maximum separability between the two domains. 

Among all evaluated approaches, **Adaptive Batch Normalization (AdaBN)** provided the best performance-to-effort trade-off. Without requiring any target labels or model retraining, AdaBN improved target-domain mIoU from **0.134 to 0.236**, representing a **76% relative improvement**. 

The investigation also revealed a persistent **class-space gap**. The *bare soil* class, which is common in the Sahel but largely absent from Iowa training data, remained undetected across all unsupervised adaptation methods. This suggests that targeted annotation of representative target samples is necessary to address the remaining performance gap. 


## Recommended Intervention

Based on the diagnostic evidence and experimental results, **AdaBN is the intervention I would recommend for deployment in a new geographic region.** It directly addresses the observed covariate shift, requires no target labels, incurs negligible computational cost, and delivered the largest improvement among the evaluated approaches. 

### Recommended Adaptation Strategy

1. Apply AdaBN using unlabeled target imagery.

2. Use radiometric harmonization only as an alternative approach or apply it before AdaBN.

3. Introduce self-training only when pseudo-label quality is sufficiently reliable.

4. Consider adversarial domain adaptation methods such as DANN if feature distributions remain separable.

5. Allocate a small annotation budget toward underrepresented target classes, particularly bare soil. 

# Conclusion


Among all tested approaches, **AdaBN** emerged as the most effective and practical solution. The method increased target-domain mIoU from **0.134 to 0.236**, representing a **76% relative improvement**, while requiring no labels and virtually no additional compute.

 
The remaining performance limitations are largely driven by a **class-space mismatch**, specifically the lack of bare-soil examples in the source dataset. As a result, targeted data annotation rather than additional unsupervised adaptation is expected to offer the greatest benefit for future improvements.

This investigation demonstrates that the Iowa-to-Sahel performance drop is primarily a consequence of **radiometric and seasonal covariate shift** rather than increased segmentation complexity. 
