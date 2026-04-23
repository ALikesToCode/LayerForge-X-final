# Literature Review Structure

This is the compact version of the related-work review. The long-form version lives in `final_report_pack/01_LITERATURE_REVIEW_ADVANCED.md`.

## Layered representations

Layered Depth Images are the natural ancestor here: they store multiple samples along a camera ray instead of a single visible surface, which is exactly what disocclusion-aware editing or rendering needs. The more recent "3D photo" line of work extends this idea by combining single-image depth with inpainting to synthesise parallax views.

## Segmentation

Panoptic segmentation is the appropriate closed-set starting point because the project requires both object instances and stuff regions. A thing-only detector is insufficient for scenes such as roads, interiors, or street environments. Mask2Former is a reasonable universal baseline. For user control and open-vocabulary coverage, GroundingDINO plus SAM2 is stronger because it lets users define layer groups beyond the fixed COCO vocabulary using natural-language prompts.

## Monocular geometry

Depth Pro, Depth Anything V2, Marigold, and MoGe-style geometry models are all plausible single-image depth backends. Depth is used in three places here: for near/far ordering, for splitting large "stuff" planes into depth bins, and for driving the parallax preview.

## Matting

Hard binary masks are where most segmentation-for-editing pipelines fall over. Hair, fur, motion blur, glass, and even good antialiasing all need fractional alpha. Image matting and Matte Anything-style approaches motivate keeping soft alpha as a first-class output rather than a cosmetic afterthought.

## Amodal segmentation and inpainting

Amodal segmentation predicts hidden object extent, while inpainting predicts hidden appearance. Together they address the hidden-scene question that editing operations eventually expose. LaMa is a solid baseline for background completion, while newer object-removal diffusion methods are plausible upgrades when additional compute is available.

## Intrinsic decomposition

Albedo/shading decomposition is an advanced extension. It matters for practical reasons: recolouring or relighting without baking the original illumination into the texture requires some notion of factored appearance. The IIW/WHDR benchmark remains the standard evaluation path for comparison against prior work.
