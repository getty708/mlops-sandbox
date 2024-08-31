import torch
import torchvision.transforms as T
from opentelemetry import trace
from services.pose_estimation.utils import get_max_preds

TRANSPOSE_INPUT_SIZE_HW = (256, 192)
TRANSPOSE_HEATMAP_SIZE_HW = (64, 48)

tracer = trace.get_tracer_provider().get_tracer(__name__)


class TransPoseWrapper:
    def __init__(self, device: str = "cuda"):
        self.device = device

        self.transpose_preproc = T.Compose(
            [
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.Resize((256, 192)),
            ],
        )
        self.pose_estimator = torch.hub.load(
            "yangsenius/TransPose:main", "tph_a4_256x192", pretrained=True
        ).to(device)
        self.pose_estimator.eval()

    def __call__(self, img_tensor: torch.Tensor, bboxes_xyxy: torch.Tensor) -> torch.Tensor:
        with tracer.start_as_current_span("preprocess"):
            cropped_imgs, scales_xy = self.preproc(img_tensor, bboxes_xyxy)
        with tracer.start_as_current_span("inference"):
            heatmaps = self.pose_estimator(cropped_imgs)
        with tracer.start_as_current_span("postprocess"):
            keypoints = self.postproc(heatmaps, bboxes_xyxy, scales_xy)
        return keypoints

    def preproc(
        self, img_tensor: torch.Tensor, bboxes_xyxy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Crop BBOX
        cropped_imgs = []
        scales_xy = []
        for xyxy in bboxes_xyxy:
            _cropped_img = img_tensor[:, :, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]].clone()
            _cropped_img = self.transpose_preproc(_cropped_img)
            cropped_imgs.append(_cropped_img)
            scales_xy.append(
                [
                    (xyxy[2] - xyxy[0]) / TRANSPOSE_HEATMAP_SIZE_HW[1],
                    (xyxy[3] - xyxy[1]) / TRANSPOSE_HEATMAP_SIZE_HW[0],
                ]
            )
        cropped_imgs = torch.cat(cropped_imgs, dim=0)

        scales_xy = torch.tensor(scales_xy)
        return cropped_imgs, scales_xy

    def postproc(
        self, heatmaps: torch.Tensor, bboxes_xyxy: torch.Tensor, scales_xy: torch.Tensor
    ) -> torch.Tensor:
        coords, _ = get_max_preds(heatmaps.detach().cpu().numpy())
        keypoints = torch.from_numpy(coords).permute(1, 0, 2)  # (Joints, Batch, 2)
        keypoints = keypoints * scales_xy + bboxes_xyxy[:, :2]
        keypoints = keypoints.permute(1, 0, 2)  # (Batch, Joints, 2)

        return keypoints
