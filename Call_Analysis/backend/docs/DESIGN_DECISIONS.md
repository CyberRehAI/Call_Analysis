# Design Decisions Document

This document outlines the key architectural and design decisions made in the Call Analysis system, along with their rationale, trade-offs, and limitations.

## Table of Contents

1. [Emotion Recognition Architecture](#emotion-recognition-architecture)
2. [Normalization Strategy](#normalization-strategy)
3. [Data Augmentation](#data-augmentation)
4. [Sentiment Analysis](#sentiment-analysis)
5. [Segment-Level Processing](#segment-level-processing)
6. [Model Architecture Choices](#model-architecture-choices)
7. [Training Strategy](#training-strategy)
8. [Limitations and Future Work](#limitations-and-future-work)

---

## Emotion Recognition Architecture

### Decision: CNN+LSTM with Temporal Attention

**Rationale:**
- **CNN layers** extract local spectral patterns from mel-spectrograms
- **LSTM layers** capture temporal dependencies across time steps
- **Temporal attention** learns to weight important time steps (better than simple averaging)
- **Bidirectional LSTM** captures both forward and backward temporal context

**Alternatives Considered:**
- Transformer-based models (more complex, requires more data)
- Pure CNN (loses temporal structure)
- Pure LSTM (inefficient for 2D spectrograms)

**Trade-offs:**
- ✅ Good balance between complexity and performance
- ✅ Handles variable-length sequences well
- ⚠️ Requires careful normalization and augmentation
- ⚠️ More parameters than simpler models

**References:**
- Similar architectures used in IEMOCAP and other SER benchmarks

---

## Normalization Strategy

### Decision: CMVN (Cepstral Mean and Variance Normalization) as Default

**Rationale:**
- **Preserves energy cues**: Unlike min-max normalization, CMVN maintains relative energy differences between frequency bands
- **Speaker-independent**: Normalizes each frequency band independently, removing speaker-dependent variations
- **Better generalization**: Works well across different speakers and recording conditions

**Alternatives Available:**
- **Min-Max Normalization**: Removes energy cues (not suitable for emotion recognition)
- **Z-Score Normalization**: Requires pre-computed statistics, less flexible
- **Log-Mel Normalization**: Preserves energy but less robust to speaker variations

**Implementation:**
- Normalizes each mel band (frequency dimension) independently
- Each band: `(band - mean) / std`
- No dataset-level statistics required (per-sample normalization)

**Trade-offs:**
- ✅ Best for speaker-independent SER
- ✅ Preserves important acoustic features
- ⚠️ Slightly more computation than min-max
- ⚠️ May not work as well for speaker-dependent tasks

**Future Work:**
- Compare CMVN vs Z-score with dataset statistics
- Experiment with hybrid normalization strategies

---

## Data Augmentation

### Decision: SpecAugment Instead of Gaussian Noise

**Rationale:**
- **More realistic**: SpecAugment masks time/frequency regions (simulates real-world variations)
- **Proven effectiveness**: Widely used in speech recognition and SER
- **Better generalization**: Helps model learn robust features

**Implementation:**
- **Time masking**: Masks consecutive time steps (simulates temporal variations)
- **Frequency masking**: Masks consecutive mel bands (simulates frequency variations)
- Applied only during training (not validation)

**Alternatives Considered:**
- **Gaussian noise**: Less realistic, can introduce artifacts
- **Time warping**: More complex, limited benefit for emotion recognition
- **Speed/pitch perturbation**: Requires waveform-level processing

**Trade-offs:**
- ✅ More realistic augmentation
- ✅ Better model robustness
- ⚠️ Requires careful tuning of mask sizes
- ⚠️ May mask important features if too aggressive

**Parameters:**
- Time masks: 2, mask size: 27 frames
- Frequency masks: 2, mask size: 13 mel bands

---

## Sentiment Analysis

### Decision: DistilBERT Binary Classification with Score Approximation

**Rationale:**
- **Pre-trained model**: DistilBERT fine-tuned on SST-2 provides good baseline
- **Fast inference**: DistilBERT is lighter than full BERT
- **Binary classification**: Simpler than multi-class sentiment

**Limitations:**
- **Binary-to-continuous approximation**: The model outputs binary classification probabilities, which are mapped to continuous sentiment scores. This is an approximation, not true sentiment intensity.
- **Not calibrated**: The score mapping is heuristic, not based on calibration curves

**Alternatives for Production:**
1. **Fine-tune regression model**: Train DistilBERT for continuous sentiment regression
2. **Calibrated mapping**: Use proper calibration curves to map probabilities to scores
3. **Multi-class model**: Use fine-grained sentiment classes (very positive, positive, neutral, negative, very negative)

**Current Implementation:**
- Binary classification: POSITIVE/NEGATIVE
- Score mapping: `score = probability` for positive, `score = -probability` for negative
- Neutral threshold: `score < 0.6` → neutral

**Future Work:**
- Fine-tune regression model for true continuous scores
- Implement proper calibration

---

## Segment-Level Processing

### Decision: True Per-Segment Emotion Detection

**Rationale:**
- **Temporal accuracy**: Each conversation segment may have different emotions
- **Better analysis**: Enables tracking emotion changes over time
- **Real-world relevance**: Call center conversations have varying emotions

**Implementation:**
- Extract segment-specific mel-spectrograms from full audio
- Run model inference on each segment independently
- No copying of base emotion (previous bug fixed)

**Previous Issue:**
- Old implementation copied base emotion to all segments
- This lost temporal information and emotion transitions

**Trade-offs:**
- ✅ Accurate per-segment emotion detection
- ✅ Enables temporal emotion tracking
- ⚠️ More computation (one inference per segment)
- ⚠️ Requires proper segment time alignment

**Future Work:**
- Batch processing for efficiency
- Emotion transition analysis
- Correlation with sentiment drift

---

## Model Architecture Choices

### Decision: LayerNorm Instead of BatchNorm in LSTM Output

**Rationale:**
- **Variable-length sequences**: BatchNorm assumes fixed batch statistics
- **Speaker-independent**: LayerNorm normalizes per-sample, better for speaker-independent SER
- **Small batch sizes**: BatchNorm less effective with small batches

**Implementation:**
- LayerNorm applied after LSTM output (before FC layers)
- BatchNorm kept in CNN layers (where it's more effective)

**Trade-offs:**
- ✅ Better for variable-length sequences
- ✅ More stable with small batches
- ⚠️ Slightly different normalization behavior
- ⚠️ May need different learning rates

### Decision: Temporal Attention vs Mean Pooling

**Rationale:**
- **Attention mechanism**: Learns to weight important time steps
- **Better than averaging**: Simple averaging treats all time steps equally
- **Alternative**: Mean+Max pooling (simpler, still better than mean alone)

**Implementation:**
- Temporal attention by default
- Can switch to mean+max pooling via `use_attention` flag

**Trade-offs:**
- ✅ Learns important temporal patterns
- ✅ Better than simple averaging
- ⚠️ Adds model complexity
- ⚠️ Requires more training data

### Decision: LSTM Length Masking

**Rationale:**
- **Variable-length sequences**: Audio segments have different lengths
- **Proper masking**: Prevents model from processing padding as real data
- **Better gradients**: pack_padded_sequence ensures gradients only flow through real data

**Implementation:**
- Dataset returns actual sequence lengths
- Model uses `pack_padded_sequence` for LSTM input
- Padding handled properly in forward pass

**Trade-offs:**
- ✅ Proper handling of variable-length sequences
- ✅ More efficient computation
- ⚠️ Requires length tracking in dataset
- ⚠️ Slightly more complex training loop

---

## Training Strategy

### Decision: Actor-Independent Split

**Rationale:**
- **Speaker-independent SER**: Goal is to recognize emotions regardless of speaker
- **Prevents data leakage**: No actor appears in both train and validation
- **Better generalization**: Tests true speaker-independent performance

**Implementation:**
- Train: Actors 01-18
- Validation: Actors 19-24
- Ensures no speaker overlap

**Trade-offs:**
- ✅ True speaker-independent evaluation
- ✅ Better generalization test
- ⚠️ May have class imbalance if actors have different emotion distributions
- ⚠️ Smaller validation set

### Decision: Label Smoothing Cross-Entropy Loss

**Rationale:**
- **Prevents overconfidence**: Model learns to be less certain
- **Better generalization**: Reduces overfitting
- **Smoother gradients**: Helps training stability

**Implementation:**
- Smoothing factor: 0.1
- Applied to all classes equally

**Trade-offs:**
- ✅ Better generalization
- ✅ Prevents overconfidence
- ⚠️ Slightly lower training accuracy (expected)
- ⚠️ Requires tuning smoothing factor

### Decision: AdamW Optimizer with ReduceLROnPlateau

**Rationale:**
- **AdamW**: Better than Adam (decoupled weight decay)
- **ReduceLROnPlateau**: Adaptive learning rate based on validation loss
- **Proven effectiveness**: Standard in modern deep learning

**Parameters:**
- Learning rate: 0.001 (default)
- Weight decay: 5e-4
- Betas: (0.9, 0.999)

**Trade-offs:**
- ✅ Good default optimizer
- ✅ Adaptive learning rate
- ⚠️ Requires tuning for different datasets
- ⚠️ May need different schedules for different models

---

## Limitations and Future Work

### Current Limitations

1. **Sentiment Score Approximation**
   - Binary classification probability mapped to continuous score
   - Not calibrated or validated
   - **Solution**: Fine-tune regression model

2. **Emotion Mapping**
   - 5-class mapping from 8 RAVDESS emotions
   - Some information loss (e.g., surprise→happiness)
   - **Solution**: Consider 6-class system or context-aware mapping

3. **Normalization**
   - CMVN is default but not compared extensively
   - No ablation study on normalization methods
   - **Solution**: Run normalization comparison experiments

4. **Model Complexity**
   - CNN+LSTM is good but not state-of-the-art
   - Could benefit from transformer architectures
   - **Solution**: Experiment with transformer-based models

5. **Data Augmentation**
   - SpecAugment parameters not extensively tuned
   - No waveform-level augmentation
   - **Solution**: Tune augmentation parameters, add waveform augmentation

6. **Evaluation**
   - Limited to RAVDESS dataset
   - No cross-dataset validation
   - **Solution**: Test on IEMOCAP, EmoDB, etc.

### Future Work

1. **Model Improvements**
   - Experiment with transformer-based architectures
   - Multi-task learning (emotion + sentiment)
   - Attention visualization for interpretability

2. **Data Improvements**
   - Cross-dataset training
   - More diverse emotion datasets
   - Real call center data (if available)

3. **Evaluation Improvements**
   - Cross-dataset evaluation
   - Per-speaker analysis
   - Emotion transition analysis

4. **Production Readiness**
   - Model quantization for faster inference
   - Batch processing optimization
   - Real-time inference pipeline

5. **Interpretability**
   - Attention visualization
   - Feature importance analysis
   - Error analysis and failure cases

---

## References

1. Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161-1178.

2. Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). *PLoS ONE*, 13(5), e0196391.

3. Busso, C., et al. (2008). IEMOCAP: Interactive emotional dyadic motion capture database. *Language Resources and Evaluation*, 42(4), 335-359.

4. Park, D. S., et al. (2019). SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition. *Interspeech 2019*.

5. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT 2019*.

6. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR 2019*.

---

## Version History

- **v1.0** (Current): Initial design decisions documented
- Future versions will track changes and new decisions

