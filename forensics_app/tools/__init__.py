"""Register course functionality here so it appears in the sidebar."""

from .blur import BlurTool
from .canny_edges import CannyEdgeTool
from .channel_split import ChannelSplitTool
from .channel_swap import ChannelSwapTool
from .contrast_stretch import ContrastStretchTool
from .convolution import ConvolutionTool
from .edge_visualisation import EdgeVisualisationTool
from .grayscale import GrayscaleTool
from .histogram import HistogramTool
from .image_info import ImageInfoTool
from .masking import MaskingTool
from .registry import ToolRegistry
from .sharpening import SharpeningTool
from .skimage_filters import SkimageFiltersTool
from .template_matching import NormalizedCrossCorrTool, TemplateMatchingTool
from .hog_feature import HOGFeatureTool
from .hog_detection import HOGPersonDetectionTool
from .corners import CornerDetectionTool
from .feature_descriptors import FeatureDescriptorTool
from .feature_matching import FeatureMatchingTool
from .image_lookup import ImageLookupTool


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            # Starter tools
            ImageInfoTool(),
            GrayscaleTool(),
            # Set 2
            BlurTool(),
            ChannelSplitTool(),
            ChannelSwapTool(),
            MaskingTool(),
            HistogramTool(),
            ContrastStretchTool(),
            # Set 3
            ConvolutionTool(),
            SkimageFiltersTool(),
            CannyEdgeTool(),
            EdgeVisualisationTool(),
            SharpeningTool(),
            # Set 4: Feature & HOG Detection
            NormalizedCrossCorrTool(),
            TemplateMatchingTool(),
            HOGFeatureTool(),
            HOGPersonDetectionTool(),
            # Set 5: Features & Matching
            CornerDetectionTool(),
            FeatureDescriptorTool(),
            FeatureMatchingTool(),
            ImageLookupTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
