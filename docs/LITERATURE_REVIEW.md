# Literature Review Structure

## Layered representations

Layered Depth Images store multiple samples along a camera ray and motivate disocclusion-aware editing/rendering. 3D photo methods later combine single-image depth with inpainting to synthesize parallax views.

## Segmentation

Panoptic segmentation is the correct closed-set baseline because the project needs both object instances and stuff regions. Mask2Former is a strong universal segmentation baseline. Open-vocabulary detection plus SAM2 strengthens novelty because user prompts can define layer groups outside COCO categories.

## Monocular geometry

Depth Pro, Depth Anything V2, Marigold, and MoGe-style geometry models are modern single-image geometry backends. The project uses depth for ordering, stuff-plane splitting, and parallax.

## Matting

Hard masks are not enough for editing. Image matting and Matte Anything-style approaches motivate soft boundaries for hair, fur, transparency, and blur.

## Amodal segmentation and inpainting

Amodal segmentation predicts hidden object extent; inpainting predicts hidden appearance. LaMa and newer object-removal methods are relevant for completed background layers.

## Intrinsic decomposition

Albedo/shading decomposition enables recoloring and relighting without baking lighting changes into texture. IIW/WHDR is the classic evaluation pathway.
