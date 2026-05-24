# Emotion Mapping Justification

## Overview

This document provides the rationale and justification for mapping RAVDESS's 8 original emotion classes to our 5-class system for Speech Emotion Recognition (SER).

## RAVDESS Original Emotion Classes

The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) contains 8 emotion classes:

1. **Neutral** (01) - No emotional content
2. **Calm** (02) - Low arousal, neutral valence
3. **Happiness** (03) - High arousal, positive valence
4. **Sadness** (04) - Low arousal, negative valence
5. **Anger** (05) - High arousal, negative valence
6. **Fear** (06) - High arousal, negative valence
7. **Disgust** (07) - High arousal, negative valence
8. **Surprise** (08) - High arousal, ambiguous valence

## Our 5-Class System

We map RAVDESS emotions to 5 classes:

1. **Neutral** - Includes neutral and calm
2. **Happiness** - Includes happiness and surprise
3. **Anger** - Direct mapping
4. **Sadness** - Direct mapping
5. **Frustration** - Includes fear and disgust

## Mapping Rationale

### 1. Calm → Neutral

**Justification:**
- **Russell's Circumplex Model of Affect (1980)**: Calm and neutral both occupy the low-arousal, neutral-valence region
- **Acoustic Similarity**: Both emotions exhibit similar prosodic features (low pitch variation, moderate energy)
- **Practical Consideration**: In call center contexts, calm and neutral are often indistinguishable and both represent baseline emotional states

**Reference:**
- Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161-1178.

### 2. Fear/Disgust → Frustration

**Justification:**
- **Valence-Arousal Similarity**: Both fear and disgust share high arousal and negative valence with frustration
- **Acoustic Patterns**: High energy, rapid speech rate, and tense vocal quality are common across these emotions
- **Contextual Relevance**: In call center scenarios, fear and disgust often manifest as frustration when customers face service issues
- **Prior SER Work**: Similar grouping used in IEMOCAP dataset and other emotion recognition studies

**References:**
- Busso, C., et al. (2008). IEMOCAP: Interactive emotional dyadic motion capture database. *Language Resources and Evaluation*, 42(4), 335-359.
- Schuller, B., et al. (2011). Cross-corpus acoustic emotion recognition: Variances and strategies. *IEEE Transactions on Affective Computing*, 1(2), 119-130.

### 3. Surprise → Happiness

**Justification:**
- **Valence-Arousal Model**: Surprise has high arousal with ambiguous valence, but in positive contexts (common in service interactions), it aligns with happiness
- **Acoustic Features**: Both emotions share high pitch variation and energy
- **Practical Limitation**: Surprise is context-dependent; mapping to happiness provides a reasonable approximation for positive surprise scenarios
- **Alternative Consideration**: In future work, surprise could be kept separate if sufficient data is available

**Limitation:** This mapping may not capture negative surprise accurately. Consider expanding to 6 classes if data permits.

## Alternative Mapping Strategies

### Strict Mapping (8 Classes)
- **Use Case**: Research requiring fine-grained emotion distinctions
- **Trade-off**: Requires more training data and may suffer from class imbalance
- **Access**: Use `--emotion_mapping strict` flag

### Expanded Mapping (Alternative Grouping)
- **Use Case**: Experimentation with different emotion groupings
- **Approach**: Keeps sadness separate from frustration, allows for different valence-arousal combinations
- **Access**: Use `--emotion_mapping expanded` flag

## Acoustic Similarity Analysis

Based on acoustic feature analysis:

| Emotion Pair | Acoustic Similarity | Justification |
|-------------|---------------------|---------------|
| Calm-Neutral | High | Similar prosody, energy, pitch |
| Fear-Disgust | High | High arousal, negative valence, tense vocal quality |
| Surprise-Happiness | Moderate | High arousal, positive contexts |

## Ablation Study Recommendations

To validate this mapping, consider:

1. **Baseline Comparison**: Train models with 8-class vs 5-class mapping
2. **Per-Class Accuracy**: Analyze which mappings improve vs degrade performance
3. **Confusion Matrix Analysis**: Identify which emotions are most frequently confused
4. **Cross-Dataset Validation**: Test mapping on other emotion datasets (IEMOCAP, EmoDB)

## References

1. Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161-1178.

2. Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. *PLoS ONE*, 13(5), e0196391.

3. Busso, C., et al. (2008). IEMOCAP: Interactive emotional dyadic motion capture database. *Language Resources and Evaluation*, 42(4), 335-359.

4. Schuller, B., et al. (2011). Cross-corpus acoustic emotion recognition: Variances and strategies. *IEEE Transactions on Affective Computing*, 1(2), 119-130.

5. Pichora-Fuller, M. K., & Dupuis, K. (2020). Toronto emotional speech set (TESS). *University of Toronto Psychology Department*.

## Future Work

- **6-Class System**: Consider separating surprise as its own class
- **Context-Aware Mapping**: Use conversation context to disambiguate emotions
- **Multi-Dataset Training**: Train on combined datasets with consistent mapping
- **Dynamic Mapping**: Adjust mapping based on acoustic feature clustering

