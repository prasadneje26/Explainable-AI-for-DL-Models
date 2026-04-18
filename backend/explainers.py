"""
explainers.py — XAI explanations for all 4 model types.

Multiple explanation methods:
  Image   → SHAP DeepExplainer, LIME, Grad-CAM
  Text    → SHAP GradientExplainer, LIME
  Tabular → SHAP KernelExplainer
  Audio   → SHAP GradientExplainer
"""

import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from PIL import Image
import base64
import tensorflow as tf
from lime import lime_image
from lime.lime_text import LimeTextExplainer


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _arr_to_base64(arr_uint8: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr_uint8).save(buf, format='PNG')
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── Model 1: Image SHAP ───────────────────────────────────────────────────────

class ImageSHAP:
    """SHAP DeepExplainer for Image CNN."""

    def __init__(self, model, background: np.ndarray):
        print("[SHAP-Image] Initializing DeepExplainer...")
        self.model = model
        self.explainer = shap.DeepExplainer(model, background)

    def explain(self, image_input: np.ndarray, class_idx: int):
        """
        image_input: (1, 32, 32, 3)
        Returns: (overlay_base64, heatmap_base64)
        """
        try:
            shap_vals = self.explainer.shap_values(image_input)
            if isinstance(shap_vals, list):
                sv = shap_vals[class_idx][0]   # (H, W, C)
            else:
                sv = shap_vals[0, :, :, :, class_idx]

            heatmap = np.sum(np.abs(sv), axis=-1)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

            # Overlay
            img_disp = image_input[0].copy()
            img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)
            img_uint8 = (img_disp * 255).astype(np.uint8)

            cmap = plt.get_cmap('RdYlGn')
            hmap_color = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
            overlay = (img_uint8 * 0.5 + hmap_color * 0.5).astype(np.uint8)

            # Matplotlib plot
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_uint8); axes[0].set_title('Original'); axes[0].axis('off')
            axes[1].imshow(heatmap, cmap='hot'); axes[1].set_title('SHAP Heatmap'); axes[1].axis('off')
            axes[2].imshow(overlay); axes[2].set_title('Overlay'); axes[2].axis('off')
            plt.suptitle('SHAP DeepExplainer — Image CNN', fontsize=13)
            plt.tight_layout()

            return _arr_to_base64(overlay), _fig_to_base64(fig)
        except Exception as e:
            print(f"[SHAP-Image] Error: {e}")
            return None, None


class ImageLIME:
    """LIME for Image CNN."""

    def __init__(self, model):
        print("[LIME-Image] Initializing LimeImageExplainer...")
        self.model = model
        self.explainer = lime_image.LimeImageExplainer()

    def explain(self, image_input: np.ndarray, class_idx: int):
        """
        image_input: (1, 32, 32, 3)
        Returns: overlay_base64
        """
        try:
            def predict_fn(images):
                return self.model.predict(images, verbose=0)

            img = image_input[0]  # (32, 32, 3)
            explanation = self.explainer.explain_instance(
                img.astype('double'), predict_fn, top_labels=5, hide_color=0, num_samples=1000
            )
            temp, mask = explanation.get_image_and_mask(
                class_idx, positive_only=True, num_features=10, hide_rest=True
            )
            overlay = np.uint8(mask * 255)
            return _arr_to_base64(overlay)
        except Exception as e:
            print(f"[LIME-Image] Error: {e}")
            return None


class ImageGradCAM:
    """Grad-CAM for Image CNN."""

    def __init__(self, model):
        print("[GradCAM-Image] Initializing...")
        self.model = model

    def explain(self, image_input: np.ndarray, class_idx: int):
        """
        image_input: (1, 32, 32, 3)
        Returns: heatmap_base64
        """
        try:
            img = image_input[0]  # (32, 32, 3)
            img_tensor = tf.convert_to_tensor(img[None, ...], dtype=tf.float32)

            with tf.GradientTape() as tape:
                tape.watch(img_tensor)
                conv_outputs, predictions = self.grad_model(img_tensor)
                loss = predictions[:, class_idx]

            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            heatmap = heatmap.numpy()

            # Resize to original size
            heatmap = np.uint8(255 * heatmap)
            heatmap = Image.fromarray(heatmap).resize((32, 32), Image.BILINEAR)
            heatmap = np.array(heatmap)

            return _arr_to_base64(heatmap)
        except Exception as e:
            print(f"[GradCAM-Image] Error: {e}")
            return None

    @property
    def grad_model(self):
        if not hasattr(self, '_grad_model'):
            # Assuming the model has conv layers, find the last conv layer
            last_conv_layer = None
            for layer in reversed(self.model.layers):
                if 'conv' in layer.name.lower():
                    last_conv_layer = layer
                    break
            if last_conv_layer is None:
                raise ValueError("No convolutional layer found in model")

            self._grad_model = tf.keras.Model(
                [self.model.inputs], [last_conv_layer.output, self.model.output]
            )
        return self._grad_model


# ── Model 2: Text SHAP ────────────────────────────────────────────────────────

class TextSHAP:
    """SHAP GradientExplainer for Text LSTM."""

    def __init__(self, model, background: np.ndarray):
        print("[SHAP-Text] Initializing GradientExplainer...")
        self.model = model
        self.explainer = shap.GradientExplainer(model, background)

    def explain(self, text_input: np.ndarray, class_idx: int,
                original_text: str, word_index: dict):
        """
        text_input: (1, MAX_SEQ_LEN) int32
        Returns: plot_base64
        """
        try:
            inp = text_input.astype(np.float32)
            shap_vals = self.explainer.shap_values(inp)

            if isinstance(shap_vals, list):
                sv = shap_vals[class_idx][0]   # (MAX_SEQ_LEN,) or (MAX_SEQ_LEN, embed)
            else:
                sv = shap_vals[0, :, class_idx]

            if sv.ndim > 1:
                sv = np.sum(np.abs(sv), axis=-1)

            # Map back to words
            rev_index = {v+3: k for k, v in word_index.items()}
            words = original_text.lower().split()[:20]
            tokens = text_input[0][:len(words)]
            word_labels = [rev_index.get(int(t), '?') for t in tokens]
            word_shap   = sv[:len(words)]

            # Bar chart
            fig, ax = plt.subplots(figsize=(max(8, len(words)*0.6), 4))
            colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in word_shap]
            ax.bar(range(len(word_labels)), word_shap, color=colors)
            ax.set_xticks(range(len(word_labels)))
            ax.set_xticklabels(word_labels, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('SHAP Value')
            ax.set_title('SHAP Word Importance — Text LSTM\n(Green=Positive contribution, Red=Negative)')
            ax.axhline(0, color='black', linewidth=0.8)
            plt.tight_layout()

            return _fig_to_base64(fig)
        except Exception as e:
            print(f"[SHAP-Text] Error: {e}")
            return None


class TextLIME:
    """LIME for Text LSTM."""

    def __init__(self, model):
        print("[LIME-Text] Initializing LimeTextExplainer...")
        self.model = model
        self.explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])

    def explain(self, text: str, class_idx: int):
        """
        text: raw text string
        Returns: explanation dict
        """
        try:
            def predict_fn(texts):
                # Preprocess texts
                from data import preprocess_text
                inputs = np.array([preprocess_text(t)[0] for t in texts])
                probs = self.model.predict(inputs, verbose=0)
                return probs

            explanation = self.explainer.explain_instance(
                text, predict_fn, num_features=10, labels=[class_idx]
            )
            return explanation.as_list(label=class_idx)
        except Exception as e:
            print(f"[LIME-Text] Error: {e}")
            return None


# ── Model 3: Tabular SHAP ─────────────────────────────────────────────────────

class TabularSHAP:
    """SHAP KernelExplainer for Tabular DNN."""

    FEATURE_NAMES = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

    def __init__(self, model, background: np.ndarray):
        print("[SHAP-Tabular] Initializing KernelExplainer...")
        self.model = model
        # KernelExplainer works with any model via predict function
        self.explainer = shap.KernelExplainer(
            lambda x: model.predict(x, verbose=0),
            background[:20]   # small background for speed
        )

    def explain(self, tabular_input: np.ndarray, class_idx: int, class_name: str):
        """
        tabular_input: (1, 4)
        Returns: plot_base64
        """
        try:
            shap_vals = self.explainer.shap_values(tabular_input, nsamples=100)

            if isinstance(shap_vals, list):
                sv = shap_vals[class_idx][0]   # (4,)
            else:
                sv = shap_vals[0]

            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in sv]
            bars = ax.barh(self.FEATURE_NAMES, sv, color=colors)
            ax.set_xlabel('SHAP Value (contribution to prediction)')
            ax.set_title(f'SHAP Feature Importance — Tabular DNN\nPredicted: {class_name}')
            ax.axvline(0, color='black', linewidth=0.8)

            # Add value labels
            for bar, val in zip(bars, sv):
                ax.text(val + (0.001 if val >= 0 else -0.001),
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.4f}', va='center',
                        ha='left' if val >= 0 else 'right', fontsize=9)
            plt.tight_layout()

            return _fig_to_base64(fig)
        except Exception as e:
            print(f"[SHAP-Tabular] Error: {e}")
            return None


# ── Model 4: Audio SHAP ───────────────────────────────────────────────────────

class AudioSHAP:
    """SHAP GradientExplainer for Audio 1D-CNN."""

    def __init__(self, model, background: np.ndarray):
        print("[SHAP-Audio] Initializing GradientExplainer...")
        self.model = model
        self.explainer = shap.GradientExplainer(model, background)

    def explain(self, audio_input: np.ndarray, class_idx: int, class_name: str):
        """
        audio_input: (1, AUDIO_LEN, 1)
        Returns: plot_base64
        """
        try:
            shap_vals = self.explainer.shap_values(audio_input)

            if isinstance(shap_vals, list):
                sv = shap_vals[class_idx][0, :, 0]   # (AUDIO_LEN,)
            else:
                sv = shap_vals[0, :, 0, class_idx]

            signal = audio_input[0, :, 0]
            t = np.arange(len(signal))

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

            # Original signal
            ax1.plot(t, signal, color='#3498db', linewidth=0.8)
            ax1.set_ylabel('Amplitude')
            ax1.set_title(f'Input Signal — Predicted: {class_name}')
            ax1.grid(True, alpha=0.3)

            # SHAP values
            sv_norm = (sv - sv.min()) / (sv.max() - sv.min() + 1e-8)
            ax2.fill_between(t, sv, 0,
                             where=(sv >= 0), color='#2ecc71', alpha=0.7, label='Positive')
            ax2.fill_between(t, sv, 0,
                             where=(sv < 0),  color='#e74c3c', alpha=0.7, label='Negative')
            ax2.set_ylabel('SHAP Value')
            ax2.set_xlabel('Time Step')
            ax2.set_title('SHAP Time-Step Importance')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)

            plt.suptitle('SHAP GradientExplainer — Audio 1D-CNN', fontsize=13)
            plt.tight_layout()

            return _fig_to_base64(fig)
        except Exception as e:
            print(f"[SHAP-Audio] Error: {e}")
            return None
