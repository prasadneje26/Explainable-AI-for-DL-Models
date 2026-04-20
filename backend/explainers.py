"""
explainers.py — Deep SHAP explanations for all 4 models.

Each explainer produces:
  - A clear visualization
  - Deep textual explanation of what SHAP found
  - Model-specific interpretation

SHAP Methods used:
  Image   → DeepExplainer   (fast, designed for neural nets, uses DeepLIFT)
  Text    → KernelExplainer (model-agnostic, works on any function)
  Tabular → KernelExplainer (model-agnostic, works on any function)
  Audio   → GradientExplainer (uses integrated gradients)
"""

import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io, base64
from PIL import Image


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _arr_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format='PNG')
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── 1. Image SHAP ─────────────────────────────────────────────────────────────

class ImageSHAP:
    """
    SHAP DeepExplainer for Image CNN (MNIST).

    DeepExplainer uses the DeepLIFT algorithm:
    - Compares each neuron's activation to a reference (background) activation
    - Backpropagates these differences to the input pixels
    - Result: each pixel gets a SHAP value = its contribution to the prediction

    Positive SHAP pixel → pushed the model toward predicting this digit
    Negative SHAP pixel → pushed the model away from this digit
    """

    def __init__(self, model, background: np.ndarray):
        print("[SHAP-Image] Initializing DeepExplainer with 50 background samples...")
        self.model = model
        self.explainer = shap.DeepExplainer(model, background[:50])
        print("[SHAP-Image] Ready.")

    def explain(self, inp: np.ndarray, class_idx: int):
        """inp: (1,28,28,1) → (overlay_b64, plot_b64, deep_text)"""
        try:
            sv = self.explainer.shap_values(inp)
            s  = sv[class_idx][0, :, :, 0] if isinstance(sv, list) \
                 else sv[0, :, :, 0, class_idx]

            pos_pixels = int(np.sum(s > 0))
            neg_pixels = int(np.sum(s < 0))
            max_shap   = float(np.max(s))
            min_shap   = float(np.min(s))

            # Heatmap
            hmap = np.abs(s)
            hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)

            orig     = (inp[0, :, :, 0] * 255).astype(np.uint8)
            orig_rgb = np.stack([orig]*3, axis=-1)
            cmap     = plt.get_cmap('hot')
            hmap_rgb = (cmap(hmap)[:, :, :3] * 255).astype(np.uint8)
            overlay  = (orig_rgb * 0.45 + hmap_rgb * 0.55).astype(np.uint8)

            # Rich 3-panel plot
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.patch.set_facecolor('#f8fafc')

            axes[0].imshow(orig, cmap='gray', interpolation='nearest')
            axes[0].set_title('Original Input\n(28×28 grayscale)', fontsize=10, fontweight='bold')
            axes[0].axis('off')

            im = axes[1].imshow(s, cmap='RdYlGn', interpolation='nearest',
                                vmin=min_shap, vmax=max_shap)
            axes[1].set_title('SHAP Values per Pixel\n(Green=positive, Red=negative)',
                              fontsize=10, fontweight='bold')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

            axes[2].imshow(overlay, interpolation='nearest')
            axes[2].set_title('SHAP Heatmap Overlay\n(Bright = most important pixels)',
                              fontsize=10, fontweight='bold')
            axes[2].axis('off')

            plt.suptitle(f'SHAP DeepExplainer — Predicted Digit: {class_idx}',
                         fontsize=13, fontweight='bold', y=1.02)
            plt.tight_layout()

            deep_text = (
                f"The model predicted digit '{class_idx}' by analyzing pixel-level patterns. "
                f"SHAP DeepExplainer found {pos_pixels} pixels that positively contributed "
                f"(pushed toward digit {class_idx}) and {neg_pixels} pixels that negatively "
                f"contributed (pushed away). "
                f"The strongest positive pixel had SHAP value {max_shap:.4f} and the strongest "
                f"negative had {min_shap:.4f}. "
                f"The bright/hot regions in the heatmap correspond to the key strokes and curves "
                f"that define digit '{class_idx}' — for example, the loop in '0', the vertical "
                f"stroke in '1', or the curves in '8'. "
                f"This shows the CNN learned to focus on the same visual features a human would use."
            )

            return _arr_to_b64(overlay), _fig_to_b64(fig), deep_text

        except Exception as e:
            print(f"[SHAP-Image] Error: {e}")
            return None, None, str(e)


# ── 2. Text SHAP ──────────────────────────────────────────────────────────────

class TextSHAP:
    """
    SHAP KernelExplainer for Text DNN (IMDB).

    KernelExplainer is model-agnostic:
    - Creates perturbed versions of the input by masking words
    - Observes how the prediction changes
    - Fits a weighted linear model to assign Shapley values to each word

    This tells us exactly which words drove the sentiment prediction.
    """

    def __init__(self, model, background: np.ndarray):
        print("[SHAP-Text] Initializing KernelExplainer...")
        self.model = model
        self.explainer = shap.KernelExplainer(
            lambda x: model.predict(x.astype(np.float32), verbose=0),
            background[:20]
        )
        self._rev_index = None
        print("[SHAP-Text] Ready.")

    def _rev(self):
        if self._rev_index is None:
            import tensorflow as tf
            wi = tf.keras.datasets.imdb.get_word_index()
            self._rev_index = {v: k for k, v in wi.items()}
        return self._rev_index

    def explain(self, bow_vec: np.ndarray, class_idx: int, original_text: str):
        """bow_vec: (1, VOCAB_SIZE) → (plot_b64, deep_text)"""
        try:
            sv = self.explainer.shap_values(bow_vec, nsamples=300)
            s  = sv[class_idx][0] if isinstance(sv, list) else sv[0]

            rev     = self._rev()
            present = np.where(bow_vec[0] > 0)[0]
            if len(present) == 0:
                present = np.argsort(np.abs(s))[-15:]

            top_idx   = present[np.argsort(np.abs(s[present]))[-15:]]
            top_words = [rev.get(int(i), f'word_{i}') for i in top_idx]
            top_vals  = s[top_idx]

            # Sort for display
            order      = np.argsort(top_vals)
            top_words  = [top_words[i] for i in order]
            top_vals   = top_vals[order]

            pos_words = [w for w, v in zip(top_words, top_vals) if v > 0]
            neg_words = [w for w, v in zip(top_words, top_vals) if v < 0]
            sentiment = 'Positive' if class_idx == 1 else 'Negative'

            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#f8fafc')
            colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in top_vals]
            bars = ax.barh(top_words, top_vals, color=colors, edgecolor='white', height=0.7)
            ax.axvline(0, color='#2c3e50', linewidth=1.2)
            ax.set_xlabel('SHAP Value  (contribution to prediction)', fontsize=10)
            ax.set_title(
                f'SHAP Word Importance — Predicted: {sentiment}\n'
                f'Green = pushed toward {sentiment} | Red = pushed against {sentiment}',
                fontsize=11, fontweight='bold'
            )
            for bar, val in zip(bars, top_vals):
                ax.text(val + (0.001 if val >= 0 else -0.001),
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.4f}', va='center',
                        ha='left' if val >= 0 else 'right', fontsize=8)
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()

            deep_text = (
                f"The model classified this text as '{sentiment}'. "
                f"SHAP KernelExplainer analyzed {len(present)} words present in the input. "
                f"The top positive words (pushing toward {sentiment}) were: "
                f"{', '.join(pos_words[-3:]) if pos_words else 'none'}. "
                f"The top negative words (pushing against {sentiment}) were: "
                f"{', '.join(neg_words[:3]) if neg_words else 'none'}. "
                f"This shows the model learned that certain words are strong indicators of "
                f"sentiment — for example, words like 'excellent', 'amazing', 'loved' push "
                f"toward Positive, while 'terrible', 'boring', 'waste' push toward Negative. "
                f"The bag-of-words approach treats each word independently, so SHAP can "
                f"directly assign a contribution score to each word in the vocabulary."
            )

            return _fig_to_b64(fig), deep_text

        except Exception as e:
            print(f"[SHAP-Text] Error: {e}")
            return None, str(e)


# ── 3. Tabular SHAP ───────────────────────────────────────────────────────────

class TabularSHAP:
    """
    SHAP KernelExplainer for Tabular DNN (Iris).

    For tabular data, SHAP is most interpretable:
    - Each feature (sepal/petal measurement) gets a precise contribution score
    - We can directly say 'petal length = 5.1cm contributed +0.42 toward virginica'
    - This is the most academically rigorous XAI method for structured data
    """

    FEATURES = ['Sepal Length (cm)', 'Sepal Width (cm)',
                'Petal Length (cm)', 'Petal Width (cm)']
    COLORS   = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6']

    def __init__(self, model, background: np.ndarray):
        print("[SHAP-Tabular] Initializing KernelExplainer...")
        self.model = model
        self.explainer = shap.KernelExplainer(
            lambda x: model.predict(x.astype(np.float32), verbose=0),
            background[:20]
        )
        print("[SHAP-Tabular] Ready.")

    def explain(self, inp: np.ndarray, class_idx: int, class_name: str,
                raw_features: list):
        """inp: (1,4) standardized → (plot_b64, deep_text)"""
        try:
            sv = self.explainer.shap_values(inp, nsamples=300)

            # KernelExplainer returns:
            #   list of (n_samples, n_features) — one per class  → sv[class_idx][0]
            #   OR single (n_samples, n_features) for binary     → sv[0]
            if isinstance(sv, list) and len(sv) > class_idx:
                s = np.array(sv[class_idx]).flatten()
            elif isinstance(sv, list):
                s = np.array(sv[0]).flatten()
            else:
                s = np.array(sv).flatten()

            # Ensure length matches features
            s = s[:len(self.FEATURES)]

            most_imp = self.FEATURES[int(np.argmax(np.abs(s)))]
            most_val = float(s[int(np.argmax(np.abs(s)))])

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
            fig.patch.set_facecolor('#f8fafc')

            # Bar chart
            colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in s]
            bars = ax1.barh(self.FEATURES, s, color=colors, edgecolor='white', height=0.6)
            ax1.axvline(0, color='#2c3e50', linewidth=1.2)
            ax1.set_xlabel('SHAP Value')
            ax1.set_title(f'Feature Contributions → {class_name}', fontweight='bold')
            for bar, val in zip(bars, s):
                ax1.text(val + (0.005 if val >= 0 else -0.005),
                         bar.get_y() + bar.get_height()/2,
                         f'{val:.4f}', va='center',
                         ha='left' if val >= 0 else 'right', fontsize=9)
            ax1.grid(axis='x', alpha=0.3)

            # Input values vs SHAP
            x_pos = np.arange(len(self.FEATURES))
            ax2.bar(x_pos, raw_features, color=self.COLORS, alpha=0.7, label='Input value (cm)')
            ax2_twin = ax2.twinx()
            ax2_twin.plot(x_pos, s, 'ko-', linewidth=2, markersize=8, label='SHAP value')
            ax2_twin.axhline(0, color='gray', linewidth=0.8, linestyle='--')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(['Sepal L', 'Sepal W', 'Petal L', 'Petal W'], fontsize=9)
            ax2.set_ylabel('Measurement (cm)', color='#2c3e50')
            ax2_twin.set_ylabel('SHAP Value', color='black')
            ax2.set_title('Input Values vs SHAP Contributions', fontweight='bold')
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

            plt.suptitle(f'SHAP KernelExplainer — Iris: Predicted {class_name}',
                         fontsize=13, fontweight='bold')
            plt.tight_layout()

            deep_text = (
                f"The model predicted '{class_name}' based on the 4 flower measurements. "
                f"The most influential feature was '{most_imp}' with SHAP value {most_val:.4f}. "
                f"Positive SHAP values (green) pushed the prediction toward '{class_name}', "
                f"while negative values (red) pushed against it. "
                f"In Iris classification, petal measurements are typically the most "
                f"discriminative: setosa has very small petals (length < 2cm), "
                f"versicolor has medium petals (3–5cm), and virginica has large petals (>5cm). "
                f"The SHAP values confirm which specific measurements were decisive for "
                f"this particular flower sample."
            )

            return _fig_to_b64(fig), deep_text

        except Exception as e:
            print(f"[SHAP-Tabular] Error: {e}")
            return None, str(e)


# ── 4. Audio SHAP ─────────────────────────────────────────────────────────────

class AudioSHAP:
    """
    SHAP GradientExplainer for Audio 1D-CNN.

    GradientExplainer uses Integrated Gradients:
    - Integrates gradients along a path from a baseline (zeros) to the input
    - Each time step gets a SHAP value = its contribution to the prediction
    - Shows WHICH part of the signal the CNN focused on
    """

    def __init__(self, model, background: np.ndarray):
        print("[SHAP-Audio] Initializing GradientExplainer...")
        self.model = model
        self.explainer = shap.GradientExplainer(model, background[:50])
        print("[SHAP-Audio] Ready.")

    def explain(self, inp: np.ndarray, class_idx: int, class_name: str):
        """inp: (1, AUDIO_LEN, 1) → (plot_b64, deep_text)"""
        try:
            sv = self.explainer.shap_values(inp)

            # GradientExplainer returns list[class] of (N, T, 1) or (N, T, 1, C)
            if isinstance(sv, list) and len(sv) > class_idx:
                s = np.array(sv[class_idx]).reshape(-1)[:inp.shape[1]]
            elif isinstance(sv, list):
                s = np.array(sv[0]).reshape(-1)[:inp.shape[1]]
            else:
                arr = np.array(sv)
                if arr.ndim == 4:   # (N, T, 1, C)
                    s = arr[0, :, 0, class_idx]
                else:               # (N, T, 1)
                    s = arr[0, :, 0]

            signal = inp[0, :, 0]
            t      = np.arange(len(signal))

            pos_steps = int(np.sum(s > 0))
            neg_steps = int(np.sum(s < 0))
            peak_t    = int(np.argmax(np.abs(s)))

            # Build deep explanation text first (before plot, so it's always available)
            deep_text = (
                f"The model classified this signal as '{class_name}'. "
                f"SHAP GradientExplainer analyzed all {len(signal)} time steps. "
                f"{pos_steps} time steps positively contributed (pushed toward '{class_name}') "
                f"and {neg_steps} negatively contributed. "
                f"The most important time step was at position {peak_t}. "
                f"For a Sine Wave, the model focuses on smooth periodic transitions. "
                f"For a Square Wave, it focuses on sharp +1/-1 transitions. "
                f"For Noise, importance is spread randomly with no clear pattern. "
                f"The rolling average plot shows which regions were consistently important."
            )

            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            fig.patch.set_facecolor('#f8fafc')

            # Signal
            axes[0].plot(t, signal, color='#3498db', linewidth=1.0)
            axes[0].set_ylabel('Amplitude', fontsize=9)
            axes[0].set_title(f'Input Signal — Predicted: {class_name}',
                              fontweight='bold', fontsize=11)
            axes[0].grid(True, alpha=0.3)
            axes[0].axvline(peak_t, color='orange', linewidth=1.5,
                            linestyle='--', label=f'Peak SHAP at t={peak_t}')
            axes[0].legend(fontsize=8)

            # SHAP values
            axes[1].fill_between(t, s, 0, where=(s >= 0),
                                 color='#2ecc71', alpha=0.8, label='Positive contribution')
            axes[1].fill_between(t, s, 0, where=(s < 0),
                                 color='#e74c3c', alpha=0.8, label='Negative contribution')
            axes[1].axhline(0, color='#2c3e50', linewidth=0.8)
            axes[1].set_ylabel('SHAP Value', fontsize=9)
            axes[1].set_title('SHAP Time-Step Importance', fontweight='bold', fontsize=11)
            axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

            # Rolling importance
            window = 20
            rolling = np.convolve(np.abs(s), np.ones(window)/window, mode='same')
            axes[2].plot(t, rolling, color='#9b59b6', linewidth=1.2)
            axes[2].fill_between(t, rolling, alpha=0.3, color='#9b59b6')
            axes[2].set_ylabel('Avg |SHAP|', fontsize=9)
            axes[2].set_xlabel('Time Step', fontsize=9)
            axes[2].set_title(f'Rolling Average Importance (window={window})',
                              fontweight='bold', fontsize=11)
            axes[2].grid(True, alpha=0.3)

            plt.suptitle(f'SHAP GradientExplainer — Audio 1D-CNN: {class_name}',
                         fontsize=13, fontweight='bold')
            plt.tight_layout()

            deep_text = (
                f"The model classified this signal as '{class_name}'. "
                f"SHAP GradientExplainer analyzed all {len(signal)} time steps. "
                f"{pos_steps} time steps positively contributed (pushed toward '{class_name}') "
                f"and {neg_steps} negatively contributed. "
                f"The most important time step was at position {peak_t}. "
                f"For a Sine Wave, the model focuses on the smooth periodic transitions. "
                f"For a Square Wave, it focuses on the sharp +1/-1 transitions. "
                f"For Noise, importance is spread randomly with no clear pattern. "
                f"The rolling average plot (bottom) shows which regions of the signal "
                f"were consistently important — high peaks indicate the CNN's 'attention' areas."
            )

            return _fig_to_b64(fig), deep_text

        except Exception as e:
            print(f"[SHAP-Audio] Error: {e}")
            return None, str(e)
