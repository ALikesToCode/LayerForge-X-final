# Literature Review Structure

This is the compact version of the related work — the long-form version lives in `final_report_pack/01_LITERATURE_REVIEW_ADVANCED.md`.

## Layered representations

Layered Depth Images are the natural ancestor here: they store multiple samples along a camera ray instead of a single visible surface, which is exactly what disocclusion-aware editing or rendering needs. The more recent "3D photo" line of work extends this idea by combining single-image depth with inpainting to synthesise parallax views.

## Segmentation

Panoptic segmentation is the right closed-set starting point, since the project genuinely needs both object instances *and* stuff regions — you can't decompose a street scene with a thing-only detector. Mask2Former is a reasonable universal baseline. For novelty and user control, open-vocabulary detection plus SAM2 is the stronger story: it lets users define layer groups outside the COCO vocabulary using plain-language prompts.

## Monocular geometry

Depth Pro, Depth Anything V2, Marigold, and MoGe-style geometry models are all plausible single-image depth backends. Depth is used in three places here: for near/far ordering, for splitting large "stuff" planes into depth bins, and for driving the parallax preview.

## Matting

Hard binary masks are where most segmentation-for-editing pipelines fall over. Hair, fur, motion blur, glass, and even good antialiasing all need fractional alpha. Image matting and Matte Anything-style approaches motivate keeping soft alpha as a first-class output rather than a cosmetic afterthought.

## Amodal segmentation and inpainting

Amodal segmentation predicts hidden object extent; inpainting predicts hidden appearance. Together they cover the "what's behind this object?" question that editing always eventually raises. LaMa is a solid baseline for background completion; newer object-removal diffusion methods are obvious upgrades when compute is available.

## Intrinsic decomposition

Albedo/shading decomposition is the stretch module. The reason it matters is practical — if you want to recolour or relight without baking the original illumination into the texture, you need some notion of factored appearance. The IIW/WHDR benchmark is the traditional evaluation path when you want to compare against prior work.
