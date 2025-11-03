import io
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import sys, os, traceback
import inspect
from scipy.ndimage import zoom  # optional (we'll prefer torch interpolate)
from torch.special import logsumexp
from deepgaze_pytorch import DeepGazeIII, DeepGazeIIE, DeepGazeI  # keep others if you need them

class SaliencyService:
    def __init__(self, cb_path: str, input_size=(224, 224)):
        """
        cb_path: path to center bias .npy (e.g., mit1003 log-density prior)
        input_size: spatial size (H, W) the model sees after preprocessing
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = tuple(input_size)  # (H, W)

        # Model
        self.model = DeepGazeIII(pretrained=True).to(self.device).eval()

        # Preprocess must output exactly input_size
        self.preprocess = T.Compose([
            T.Resize(self.input_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Load center bias once (numpy log-density)
        cb_np = np.load('D:/dragonfly/codebase/deepgaze_api/models/centerbias_mit1003.npy')  # shape (H0, W0), log-density prior
        # Keep as torch on device for fast resize per request
        self.cb_base = torch.from_numpy(cb_np).float().unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,H0,W0)

        print(f"sal ready on device: {self.device}")



    def _call_deepgaze(self, x, cb):
        """
        Calls self.model with the right signature.
        - If the model expects x_hist/y_hist, we provide a dummy center fixation history.
        - Otherwise we call (x, cb) as DeepGaze I expects.
        """
        # Ensure cb has batch dim (B,H,W)
        if cb.ndim == 2:
            cb = cb.unsqueeze(0)
        B, H, W = cb.shape

        # Detect whether the model.forward wants x_hist/y_hist
        try:
            fwd = self.model.forward
        except AttributeError:
            fwd = self.model.__call__  # fallback

        params = inspect.signature(fwd).parameters
        needs_hist = ("x_hist" in params) and ("y_hist" in params)

        if needs_hist:
            # Provide a center fixation history with T=1 (repeat per batch)
            T = 1
            x_hist = torch.full((B, T), W // 2, dtype=torch.long, device=x.device)
            y_hist = torch.full((B, T), H // 2, dtype=torch.long, device=x.device)
            logits = self.model(x, cb, x_hist, y_hist)
        else:
            logits = self.model(x, cb)

        return logits

    def _centerbias_for_input(self, H, W):
        """
        Resize the cached center bias to (H, W) and normalize so log-sum-exp=0.
        Returns shape (1, H, W) on the correct device.
        """
        cb = F.interpolate(self.cb_base, size=(H, W), mode='bilinear', align_corners=False)  # (1,1,H,W)
        cb = cb.squeeze(1)  # (1,H,W)
        # Normalize: subtract logsumexp over spatial dims so that sum(exp(cb)) == 1
        cb = cb - torch.logsumexp(cb, dim=(1, 2), keepdim=True)
        return cb  # (1,H,W)

    def predict(self, img_bytes: bytes) -> bytes:
        """
        img_bytes: raw bytes of an RGB image file
        returns: PNG bytes of the grayscale saliency map (uint8 0..255)
        """
        try:
            # Load and remember original size
            pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            orig_w, orig_h = pil_img.size

            # Preprocess for model
            x = self.preprocess(pil_img).unsqueeze(0).to(self.device)  # (1,3,H,W), H,W=self.input_size

            # Center bias to match model spatial size
            H, W = self.input_size
            cb = self._centerbias_for_input(H, W)  # (1,H,W)

            with torch.no_grad():
                logits = self._call_deepgaze(x, cb)  # model expects image and log-prior
                sal = self.model.saliency(logits).squeeze(0).squeeze(0).cpu().numpy()  # (H,W) in [0,1]
            sal_resized = cv2.resize(sal, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            # Convert to 8-bit grayscale PNG in-memory
            sal_u8 = np.clip(sal_resized * 255.0, 0, 255).astype(np.uint8)
            ok, buf = cv2.imencode('.png', sal_u8)
            if not ok:
                raise RuntimeError('Failed to encode saliency PNG')

            return buf.tobytes()

        except Exception as e:
            # Log the error and re-raise or return None; here we re-raise for visibility
            tb = traceback.extract_tb(sys.exc_info()[2])[-1]
            print(f"[ERROR] {type(e).__name__} at {tb.filename}:{tb.lineno} -> {tb.line}")
            raise
