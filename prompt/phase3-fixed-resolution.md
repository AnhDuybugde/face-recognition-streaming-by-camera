Read `.agents/long_memory.md`, `.agents/short_memory.md`,
and `.agents/rule_base.md` first.

The YuNet resolution diagnostic is complete.

Observed results:

DE190384.jpg:
- original 1181x1772 -> 0 faces, 105.20 ms
- longest side 640 -> 1 face, confidence 0.9458, 8.48 ms
- longest side 320 -> 1 face, confidence 0.9318, 2.23 ms

DE190469.jpg:
- original 151x226 -> 1 face, confidence 0.9270, 1.46 ms
- longest side 640 -> 1 face, confidence 0.9455, 7.64 ms
- longest side 320 -> 1 face, confidence 0.9279, 2.38 ms

DE190692.jpg:
- original 1200x1600 -> 0 faces, 101.83 ms
- longest side 640 -> 1 face, confidence 0.9467, 10.42 ms
- longest side 320 -> 1 face, confidence 0.9310, 2.44 ms

Conclusion:
YuNet is sensitive to very large face/input scale in these enrollment images.
Downscaling restores detection and greatly reduces latency.

Implement the smallest production fix.

Requirements:

1. Do not change the recognition architecture.
2. Do not change YuNet confidence threshold.
3. Do not add fallback retries.
4. Do not upscale small images.

5. Before enrollment face detection, normalize only oversized images:

   - define max detection side = 320 pixels
   - if max(width, height) > 320:
       resize while preserving aspect ratio so the longest side becomes 320
   - otherwise:
       keep the original image unchanged

Example:

1181x1772
→ approximately 213x320

151x226
→ remains 151x226

6. After preprocessing, always call:

   h, w = image.shape[:2]
   detector.setInputSize((w, h))

7. Keep this preprocessing reusable and small.
Do not create unnecessary classes or frameworks.

8. Ensure landmarks/bounding boxes operate correctly on the resized image.

For the current enrollment pipeline, downstream alignment may use the resized detection image directly if that is already compatible with the existing SFace alignment implementation.

Do not add coordinate remapping to the original image unless it is actually required.

9. Rebuild the entire enrollment gallery.

10. Report:

- total enrollment images
- successful identities
- failures
- average detection/alignment latency
- average encoding latency
- total enrollment time
- embedding dimension
- inter-identity cosine statistics:
  mean
  median
  min
  max
  max pair

11. Compare the new results with the previous baseline:

Previous:
- 51 images
- 49 successful
- detection average: 2.61 ms/image
- encoding average: 7.06 ms/image
- total: 734.72 ms
- mean inter-identity similarity: 0.5631
- max: 0.8152

12. Verify specifically whether:
- DE190384 now enrolls successfully
- DE190692 now enrolls successfully

13. Do not proceed to webcam recognition yet.

14. Update short memory.
Update long memory only if the max-side preprocessing policy is accepted as a durable architecture decision.

Stop after rebuilding and validating the gallery.