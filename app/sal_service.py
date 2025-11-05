import io
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import sys, os, traceback
import inspect
from scipy.ndimage import zoom
from torch.special import logsumexp
from deepgaze_pytorch import DeepGazeIII, DeepGazeIIE, DeepGazeI


class SaliencyService:
    def __init__(self, cb_path: str, input_size=(224, 224), model_type='DeepGazeIII'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = tuple(input_size)
        self.model_type = model_type

        if model_type == 'DeepGazeIIE':
            self.model = DeepGazeIIE(pretrained=True).to(self.device).eval()
        elif model_type == 'DeepGazeI':
            self.model = DeepGazeI(pretrained=True).to(self.device).eval()
        else:
            self.model = DeepGazeIII(pretrained=True).to(self.device).eval()

        self.preprocess = T.Compose([
            T.Resize(self.input_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        cb_np = np.load(cb_path)
        self.cb_base = torch.from_numpy(cb_np).float().unsqueeze(0).unsqueeze(0).to(self.device)

        print(f"sal ready on device: {self.device}")

    def _call_deepgaze(self, x, cb):
        model_class_name = self.model.__class__.__name__
        is_deepgaze_iii = "III" in model_class_name or "3" in model_class_name
        is_deepgaze_iie = "IIE" in model_class_name or "IIE" in self.model_type
        is_deepgaze_i = "I" in model_class_name and "III" not in model_class_name and "IIE" not in model_class_name

        if is_deepgaze_iie or is_deepgaze_i:
            if cb.ndim == 2:
                cb = cb.unsqueeze(0)
            elif cb.ndim == 4:
                cb = cb.squeeze(1)
            print(f"[DEBUG] Calling {model_class_name} with: model(image, centerbias)")
            print(f"[DEBUG] Shapes - x: {x.shape}, cb: {cb.shape}")
            logits = self.model(x, cb)
            return logits

        if is_deepgaze_iii:
            print(f"[DEBUG] Detected {model_class_name}, calling with image only")
            try:
                logits = self.model(x)
                return logits
            except Exception as e:
                print(f"[DEBUG] Single-arg call failed: {e}, trying with center bias...")

        if cb.ndim == 2:
            cb = cb.unsqueeze(0)
            B, H, W = cb.shape
            cb_3d = cb
            cb_4d = cb.unsqueeze(1)
        elif cb.ndim == 4:
            B, _, H, W = cb.shape
            cb_4d = cb
            cb_3d = cb.squeeze(1)
        else:
            B, H, W = cb.shape
            cb_3d = cb
            cb_4d = cb.unsqueeze(1)

        try:
            fwd = self.model.forward
        except AttributeError:
            fwd = self.model.__call__
        params = inspect.signature(fwd).parameters
        param_list = [(p, params[p]) for p in params.keys() if p != 'self']
        param_names = [p[0] for p in param_list]

        needs_hist = ("x_hist" in param_names) and ("y_hist" in param_names)
        has_center_bias = any(cb_name in param_names for cb_name in ['center_bias', 'centerbias', 'cb'])
        center_bias_param_name = next(
            (cb_name for cb_name in ['centerbias', 'center_bias', 'cb'] if cb_name in param_names), None)
        num_params = len(param_names)

        first_param_is_cb = len(param_names) > 0 and (param_names[0] in ['center_bias', 'centerbias', 'cb'])

        print(f"[DEBUG] Model: {model_class_name}, signature: {param_names}, num_params: {num_params}")
        print(
            f"[DEBUG] needs_hist: {needs_hist}, has_center_bias: {has_center_bias}, center_bias_param_name: {center_bias_param_name}")

        if num_params == 1:
            print(f"[DEBUG] Calling model with only image (single parameter)")
            try:
                logits = self.model(x)
                return logits
            except Exception as e:
                print(f"[DEBUG] Single-arg call failed: {e}, trying with center bias...")

        if needs_hist:
            T = 1
            x_hist = torch.full((B, T), W // 2, dtype=torch.long, device=x.device)
            y_hist = torch.full((B, T), H // 2, dtype=torch.long, device=x.device)
            if has_center_bias and center_bias_param_name:

                print(
                    f"[DEBUG] Shape check - x: {x.shape}, cb_4d: {cb_4d.shape}, cb_3d: {cb_3d.shape}, x_hist: {x_hist.shape}, y_hist: {y_hist.shape}")

                print(f"[DEBUG] Trying with centerbias=None (checking if optional)")
                try:
                    logits = self.model(x, centerbias=None, x_hist=x_hist, y_hist=y_hist)
                    return logits
                except (TypeError, RuntimeError) as e:
                    print(f"[DEBUG] None failed: {e}")

                print(f"[DEBUG] Trying 3D center bias format: {cb_3d.shape}")
                try:
                    logits = self.model(x, **{center_bias_param_name: cb_3d, 'x_hist': x_hist, 'y_hist': y_hist})
                    return logits
                except RuntimeError as e:
                    if "channels" in str(e) or "expected input" in str(e):
                        print(f"[DEBUG] 3D format failed: {e}")
                        feat_h, feat_w = H // 8, W // 8
                        cb_feat = self._centerbias_for_input(feat_h, feat_w, keep_4d=False)
                        print(f"[DEBUG] Trying centerbias at feature map size (28x28): {cb_feat.shape}")
                        try:
                            logits = self.model(x,
                                                **{center_bias_param_name: cb_feat, 'x_hist': x_hist, 'y_hist': y_hist})
                            return logits
                        except RuntimeError as e2:
                            print(f"[DEBUG] Feature map size failed: {e2}")
                            cb_feat_3ch_4d = cb_feat.unsqueeze(1).repeat(1, 3, 1, 1)
                            print(
                                f"[DEBUG] Trying centerbias with 3 channels at feature map size (4D): {cb_feat_3ch_4d.shape}")
                            try:
                                logits = self.model(x, **{center_bias_param_name: cb_feat_3ch_4d, 'x_hist': x_hist,
                                                          'y_hist': y_hist})
                                return logits
                            except RuntimeError as e3:
                                print(f"[DEBUG] 4D with 3 channels failed: {e3}")
                                cb_feat_3ch = cb_feat.squeeze(0).repeat(3, 1, 1)
                                print(f"[DEBUG] Trying 3D format with 3 channels: {cb_feat_3ch.shape}")
                                try:
                                    logits = self.model(x, **{center_bias_param_name: cb_feat_3ch, 'x_hist': x_hist,
                                                              'y_hist': y_hist})
                                    return logits
                                except RuntimeError as e4:
                                    print(f"[DEBUG] All formats failed. Error: {e4}")
                                    raise e  # Re-raise original error with full context
                    else:
                        raise
            else:
                print(f"[DEBUG] Calling model with positional args: model(x, centerbias, x_hist, y_hist)")
                logits = self.model(x, cb, x_hist, y_hist)
        elif has_center_bias and first_param_is_cb:
            print(f"[DEBUG] Calling model with swapped order: model(cb, x)")
            logits = self.model(cb, x)
        elif has_center_bias and center_bias_param_name:
            print(f"[DEBUG] Calling model with center bias as keyword: model(x, {center_bias_param_name}=cb)")
            logits = self.model(x, **{center_bias_param_name: cb})
        else:
            print(f"[DEBUG] Calling model with positional args: model(x, cb)")
            try:
                logits = self.model(x, cb)
            except RuntimeError as e:
                if "expected input" in str(e) and "channels" in str(e):
                    print(f"[DEBUG] Channel mismatch error, trying with image only")
                    logits = self.model(x)
                else:
                    raise

        return logits

    def _centerbias_for_input(self, H, W, keep_4d=False):
        cb = F.interpolate(self.cb_base, size=(H, W), mode='bilinear', align_corners=False)
        if keep_4d:
            cb = cb - torch.logsumexp(cb, dim=(2, 3), keepdim=True)
            return cb
        else:
            cb = cb.squeeze(1)
            cb = cb - torch.logsumexp(cb, dim=(1, 2), keepdim=True)
            return cb

    def predict(self, img_bytes: bytes) -> bytes:

        try:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            orig_w, orig_h = pil_img.size

            x = self.preprocess(pil_img).unsqueeze(0).to(self.device)

            H, W = self.input_size

            if self.model_type in ['DeepGazeIIE', 'DeepGazeI']:
                cb = self._centerbias_for_input(H, W, keep_4d=False)
            else:
                cb = self._centerbias_for_input(H, W, keep_4d=True)

            with torch.no_grad():
                logits = self._call_deepgaze(x, cb)

                if self.model_type in ['DeepGazeIIE', 'DeepGazeI']:
                    if logits.ndim == 4:
                        logits = logits.squeeze(1)
                    sal = torch.exp(logits)
                    sal = sal / sal.sum(dim=(1, 2), keepdim=True)
                    sal = sal.squeeze(0).cpu().numpy()
                    print(
                        f"[DEBUG] DeepGazeIIE/I saliency - shape: {sal.shape}, min: {sal.min():.6f}, max: {sal.max():.6f}, mean: {sal.mean():.6f}")
                else:
                    sal = self.model.saliency(logits).squeeze(0).squeeze(0).cpu().numpy()
                    print(
                        f"[DEBUG] DeepGazeIII saliency - shape: {sal.shape}, min: {sal.min():.6f}, max: {sal.max():.6f}, mean: {sal.mean():.6f}")

            if sal.max() == 0 or sal.min() == sal.max():
                print(f"[WARNING] Saliency map appears to be flat or all zeros!")

            sal_resized = cv2.resize(sal, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            if sal_resized.max() > sal_resized.min():
                sal_resized = (sal_resized - sal_resized.min()) / (sal_resized.max() - sal_resized.min())
            else:
                sal_resized = np.ones_like(sal_resized) * 0.5

            sal_u8 = np.clip(sal_resized * 255.0, 0, 255).astype(np.uint8)
            print(
                f"[DEBUG] Final saliency map - shape: {sal_u8.shape}, min: {sal_u8.min()}, max: {sal_u8.max()}, mean: {sal_u8.mean():.1f}")
            ok, buf = cv2.imencode('.png', sal_u8)
            if not ok:
                raise RuntimeError('Failed to encode saliency PNG')

            return buf.tobytes()

        except Exception as e:
            tb = traceback.extract_tb(sys.exc_info()[2])[-1]
            print(f"[ERROR] {type(e).__name__} at {tb.filename}:{tb.lineno} -> {tb.line}")
            raise
