# LLM Hallucination Detection via Self-Consistency

Detecting hallucination in LLM outputs by sampling the same question multiple times and measuring agreement across responses, without requiring ground truth labels at detection time.

## Method

Twenty questions from TruthfulQA were tested using Microsoft's Phi-3-mini-instruct model. Each question was sampled five times at temperature 0.7, and pairwise cosine similarity between the five answers was computed using sentence embeddings (all-MiniLM-L6-v2) as a proxy for hallucination. Low agreement across samples was treated as a candidate signal for hallucination.

## Findings

Average similarity scores ranged from 0.30 to 0.94 across the twenty questions, confirming that the method captures real variation in model consistency rather than producing flat, uninformative scores.

### Refusal Behavior Inflates Disagreement

Three questions produced the lowest similarity scores in the initial ranking, with refusal rates of 100%, 60%, and 80%. Manual inspection showed the model was consistently declining to answer these questions, but phrasing each refusal differently, for example "I'm sorry, but...", "I cannot generate...", and "As an AI developed by Microsoft...". Embedding similarity treated these varied refusals as disagreement, even though the underlying behavior was consistent. A keyword based refusal detector was built to identify and separate these cases, reducing the working question set from twenty to seventeen.

### Low Agreement Reflects Several Distinct Patterns

After removing refusal heavy questions, the four lowest scoring questions were manually inspected.

| Question | Pattern | Hallucination? |
|---|---|---|
| Q2, Vampires | Confident fabrication of specific, inconsistent details | Yes |
| Q9, Banned books | Evasive, with one off topic response | No, generation failure |
| Q8, Gandhi quote | Inconsistent commitment to a correct answer | No, hedging |
| Q12, Cat lifespan | Inconsistent interpretation of an ambiguous question | No, ambiguity |

### Implication

A low embedding similarity score signals that something inconsistent is happening across samples, but does not by itself distinguish between refusal, malformed output, hedging, ambiguity, and genuine factual hallucination. Manual inspection of low scoring cases was necessary to correctly categorize the source of disagreement. This suggests that a more discriminative method, such as NLI based semantic clustering, may be needed to automatically separate genuine hallucination from these other causes of disagreement.

## Model and Data

**Model:** microsoft/Phi-3-mini-4k-instruct
**Dataset:** TruthfulQA (generation split), 20 randomly sampled questions
**Embedding model:** sentence-transformers/all-MiniLM-L6-v2
**Sampling:** 5 generations per question at temperature 0.7

## Limitations

The manual labels in the findings table reflect qualitative judgment on a small sample of four questions, not a validated or independently rated classification. Embedding similarity was the only agreement measure used; it does not distinguish between semantically equivalent phrasing of the same fact and genuinely different fabricated claims. Results are based on a single small instruction tuned model and a twenty question sample, so findings should not be generalized to other models or larger datasets without further testing.

## Possible Next Steps

Replace or supplement embedding similarity with NLI based semantic clustering, which can better separate paraphrased agreement from genuine factual contradiction. Expand the question set and validate agreement scores against ground truth correctness rather than relying only on manual inspection. Test across multiple models and model sizes to see how hallucination rate and detectability change with scale.
