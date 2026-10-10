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
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
