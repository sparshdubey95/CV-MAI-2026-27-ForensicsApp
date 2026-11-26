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
from .face_detection import FaceDetectionTool
from .eigenface_builder import EigenfaceBuilderTool
from .face_recognition import FaceRecognitionTool
from .bovw_tools import BoVWCodebookTool, BoVWHistogramTool, BoVWClassifierTool
from .cnn_tools import (
    CNNFeatureProjectionTool,
    CNNInferenceTool,
    CNNTrainingTool,
)
from .yolo_tools import (
    YOLOComparisonTool,
    YOLODataConversionTool,
    YOLOInferenceTool,
    YOLOTrainingTool,
)
from .clip_tools import CLIPZeroShotTool
from .unet_tools import UNetSegmentationTool, UNetTrainingTool


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
            # Set 6: Face Detection & Recognition
            FaceDetectionTool(),
            EigenfaceBuilderTool(),
            FaceRecognitionTool(),
            # Set 7: Bag of Visual Words & Classification
            BoVWCodebookTool(),
            BoVWHistogramTool(),
            BoVWClassifierTool(),
            # Set 8: Image Classification using CNN
            CNNTrainingTool(),
            CNNInferenceTool(),
            CNNFeatureProjectionTool(),
            # Set 9: Object Detection with YOLO
            YOLOInferenceTool(),
            YOLODataConversionTool(),
            YOLOTrainingTool(),
            YOLOComparisonTool(),
            # Set 10: CLIP Zero-Shot Classification
            CLIPZeroShotTool(),
            # Set 11: U-Net Segmentation
            UNetTrainingTool(),
            UNetSegmentationTool(),
        ]
    )


__all__ = [
    "ToolRegistry",
    "build_tool_registry",
    "CNNTrainingTool",
    "CNNInferenceTool",
    "CNNFeatureProjectionTool",
    "YOLOInferenceTool",
    "YOLODataConversionTool",
    "YOLOTrainingTool",
    "YOLOComparisonTool",
    "CLIPZeroShotTool",
    "UNetTrainingTool",
    "UNetSegmentationTool",
]
