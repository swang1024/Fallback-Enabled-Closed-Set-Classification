What to measure
My method has a natural decomposition into three inference variants: visual-only (V) (one api call with main_1_direct_prompt.py), textual-only (T) (two api calls with main_2_summary_gen.py and main_3_summary_pred.py), and cross-modal (V+T) (all three calls in main_1_direct_prompt.py, main_2_summary_gen.py and main_3_summary_pred.py). 
In this project, the full pipeline uses three calls per image in its generic form: one direct classification call, one image-summary call, and one summary-based classification call; by contrast, (V) uses one call and (T) uses two calls.

For each model, report:

1. Number of API/model calls per image.

2. Mean latency per image, plus standard deviation or median/IQR.

3. Throughput, e.g. images/sec or sec/100 images.

4. Estimated monetary cost per image for commercial APIs, and optionally per 1,000 images for readability.

5. Relative overhead of (V+T) to (V)

write a script for the above analysis


