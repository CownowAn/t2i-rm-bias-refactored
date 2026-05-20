# python analysis/per_prompt_r2.py \
#     --manifest /nfs/data/sohyun/projects/t2i-rm-bias/data/baselines/mjhq/topic_0/black-forest-labs-FLUX.1-dev/manifest.json \
#     --cache outputs/detection_cache/mjhq/black-forest-labs-FLUX.1-dev.json \
#     --model_key "Qwen/Qwen3.5-9B::auto" \
#     --reward_name imagereward \
#     --N 2


python analysis/per_prompt_r2.py \
    --manifest /nfs/data/sohyun/projects/t2i-rm-bias/data/baselines/mjhq/topic_0/black-forest-labs-FLUX.1-dev/manifest.json \
    --cache outputs/detection_cache/mjhq/black-forest-labs-FLUX.1-dev.json \
    --model_key "Qwen/Qwen3.5-9B::auto" \
    --reward_name imagereward \
    --attrs \
      "Eyes show exaggerated brightness (strong catchlights or unnatural glow)." \
      "High global contrast/HDR-like tonemapping with very bright highlights and deep shadows." \
      "Fur/skin reads as synthetic ‘plush/velvet’ material with overly uniform fiber direction/clumping, rather than irregular natural hair/skin variation." \
      "Internal pattern boundaries on the subject (e.g., stripes/patch edges) have unnaturally crisp, vector-like edges with no natural feathering/texture transition." \
      "Over-saturated, neon/iridescent-looking coloration on the subject." \
      "The animal/character is depicted with its mouth/beak open showing a saturated red mouth interior and/or tongue in a posed “smile” rather than a neutral closed mouth." \
      "Aggressive micro-detail and sharpening visible in textures (fur/skin/foliage) with little or no film grain." \
      "Heavy vignette where the corners/edges are noticeably darker than the center." \
      "Single dominant subject is centered and fills most of the frame." \
      "Strong shallow depth of field with the subject sharp and background heavily blurred." \
      "Pronounced rim/back lighting that creates a bright outline glow on the subject's edges." \
      "Strong cool-warm complementary color grading (warm highlights against cool blue/teal shadows)." \
      "The subject is sharply focused with crisp visible details." \
      "Bright soft circular bokeh highlights are visible in the background." \
      "Lighting clearly separates the subject from the background with a high subject-to-background brightness ratio." \
      "Composition is balanced and visually pleasing with the main subject framed naturally."
    --N 2
