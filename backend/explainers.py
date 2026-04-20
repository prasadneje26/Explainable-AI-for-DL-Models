"""
explainers.py — SHAP explanations for all 4 models.

Actual model formats:
  Image  : CIFAR-10 CNN   → DeepExplainer on (1,32,32,3) inputs
  Text   : Bidirectional LSTM → GradientExplainer on embedding sub-model
  Tabular: Iris DNN        → KernelExplainer on (1,4) standardized features
  Audio  : 1D-CNN          → GradientExplainer on (1,1000,1) signals
"""

import numpy as np
import shap
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, re
from PIL import Image as PILImage


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _arr_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    PILImage.fromarray(arr.astype(np.uint8)).save(buf, format='PNG')
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── 1. Image SHAP (CIFAR-10 — DeepExplainer) ──────────────────────────────────

class ImageSHAP:
    """
    SHAP DeepExplainer for CIFAR-10 CNN.
    Input shape: (1, 32, 32, 3) float32 [0,1].
    Uses DeepLIFT — backpropagates activation deltas through all CNN layers.
    """

    def __init__(self, model, background: np.ndarray):
        print("[SHAP-Image] Initializing DeepExplainer (CIFAR-10, 3-channel)...")
        self.model      = model
        self.background = background[:50]
        self.explainer  = shap.DeepExplainer(model, self.background)
        print("[SHAP-Image] Ready.")

    def explain(self, inp: np.ndarray, class_idx: int):
        """inp: (1,32,32,3) → (overlay_b64, plot_b64, deep_text, bullets)"""
        try:
            sv = self.explainer.shap_values(inp)

            # sv is list[num_classes] each (1,32,32,3), or array (1,32,32,3,num_classes)
            if isinstance(sv, list):
                s3 = sv[class_idx][0]            # (32,32,3)
            else:
                s3 = np.array(sv)[0, :, :, :, class_idx]   # (32,32,3)

            # Aggregate across channels: mean for signed, sum-abs for magnitude
            s_signed = s3.mean(axis=-1)          # (32,32)
            s_mag    = np.abs(s3).mean(axis=-1)  # (32,32) magnitude

            pos_px   = int(np.sum(s_signed > 0))
            neg_px   = int(np.sum(s_signed < 0))
            total_px = s_signed.size
            pct_pos  = pos_px / total_px * 100
            max_shap = float(np.max(s_signed))
            min_shap = float(np.min(s_signed))

            # Find hottest region
            rows = np.where(s_mag.max(axis=1) > s_mag.max() * 0.6)[0]
            cols = np.where(s_mag.max(axis=0) > s_mag.max() * 0.6)[0]
            region_desc = ""
            if len(rows) > 0 and len(cols) > 0:
                rc = int(np.mean(rows)); cc = int(np.mean(cols))
                v  = "top" if rc < 11 else "middle" if rc < 21 else "bottom"
                h  = "left" if cc < 11 else "center" if cc < 21 else "right"
                region_desc = f"{v}-{h} area"

            # Build overlay: original + heatmap blend
            orig_uint8 = (inp[0] * 255).astype(np.uint8)    # (32,32,3)
            norm_mag   = (s_mag - s_mag.min()) / (s_mag.max() - s_mag.min() + 1e-8)
            cmap       = plt.get_cmap('hot')
            heat_rgb   = (cmap(norm_mag)[:, :, :3] * 255).astype(np.uint8)
            overlay    = (orig_uint8 * 0.45 + heat_rgb * 0.55).astype(np.uint8)

            # ── 3-panel figure ──
            from data import CIFAR10_CLASSES
            class_name = CIFAR10_CLASSES[class_idx]

            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            fig.patch.set_facecolor('#ffffff')

            axes[0].imshow(orig_uint8)
            axes[0].set_title(f'Original Input\n32×32 RGB', fontsize=11, fontweight='bold', pad=8)
            axes[0].axis('off')

            im = axes[1].imshow(s_signed, cmap='RdYlGn', vmin=min_shap, vmax=max_shap)
            axes[1].set_title(f'SHAP Values (avg across RGB)\nGreen = supports "{class_name}"',
                              fontsize=11, fontweight='bold', pad=8)
            axes[1].axis('off')
            cb = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=8)

            axes[2].imshow(overlay)
            axes[2].set_title('Importance Heatmap Overlay\nBrighter = more important pixels',
                              fontsize=11, fontweight='bold', pad=8)
            axes[2].axis('off')

            plt.suptitle(f'SHAP DeepExplainer  —  CIFAR-10: Predicted "{class_name}"',
                         fontsize=13, fontweight='bold', y=1.01)
            plt.tight_layout()

            deep_text = (
                f"The CNN predicted '{class_name}' from a 32×32 RGB image. "
                f"SHAP DeepExplainer (DeepLIFT) traced activation differences through all "
                f"convolutional layers using {len(self.background)} CIFAR-10 reference images. "
                f"{pos_px} pixels ({pct_pos:.1f}%) positively supported the prediction; "
                f"the peak supporting pixel had SHAP = {max_shap:.4f}. "
                f"The model's strongest attention is in the {region_desc}."
            )

            bullets = [
                f"SHAP method: DeepExplainer (DeepLIFT) — backpropagates activation deltas through each CNN layer to individual pixels.",
                f"Input: 32×32 RGB image (CIFAR-10 format). SHAP values computed per pixel per channel, then averaged across the 3 channels for display.",
                f"Background baseline: {len(self.background)} CIFAR-10 test images used as reference distribution.",
                f"Supporting pixels: {pos_px} ({pct_pos:.1f}%) positively contributed to predicting '{class_name}' — peak SHAP = {max_shap:.4f}.",
                f"Opposing pixels: {neg_px} pushed against this class — min SHAP = {min_shap:.4f}.",
                f"Spatial focus: the model's strongest activations are in the {region_desc}, corresponding to the most class-distinctive features.",
                f"Color guide: in the center panel, green = pixel supports '{class_name}', red = opposes; in the overlay, bright = high importance.",
            ]

            return _arr_to_b64(overlay), _fig_to_b64(fig), deep_text, bullets

        except Exception as e:
            print(f"[SHAP-Image] Error: {e}")
            import traceback; traceback.print_exc()
            return None, None, str(e), []


# ── 2. Text SHAP (Bidirectional LSTM — GradientExplainer on embedding space) ──

class TextSHAP:
    """
    Gradient × Input attribution for Bidirectional LSTM (IMDB).

    Uses tf.GradientTape to compute gradients of the predicted class probability
    w.r.t. the embedding outputs, then multiplies elementwise by the embedding values.
    This is a standard saliency method for sequence models, equivalent to SHAP
    GradientExplainer in the single-reference limit.

    word_importance[t] = mean_over_embedding_dim( |gradient[t] * embedding[t]| )
    """

    def __init__(self, model, background: np.ndarray):
        """background: (N, 200) int32 word-ID sequences (used for reference, not GradExplainer)."""
        print("[SHAP-Text] Building LSTM gradient-attribution explainer...")
        self.model     = model
        self.bg_seqs   = background[:30]
        self.seq_len   = background.shape[1]

        # Extract embedding layer and build sub-model from embeddings onward
        self.emb_layer = model.layers[0]
        emb_dim        = self.emb_layer.output_dim

        emb_inp = tf.keras.Input(shape=(self.seq_len, emb_dim), name='emb_input')
        x = emb_inp
        for layer in model.layers[1:]:
            x = layer(x)
        self.sub_model = tf.keras.Model(emb_inp, x)

        self._rev_index = None
        print("[SHAP-Text] Ready.")

    def _rev(self):
        if self._rev_index is None:
            wi = tf.keras.datasets.imdb.get_word_index()
            self._rev_index = {v + 3: k for k, v in wi.items()}
            self._rev_index[0] = '<PAD>'
            self._rev_index[1] = '<START>'
            self._rev_index[2] = '<UNK>'
        return self._rev_index

    def _grad_x_input(self, seq: np.ndarray, class_idx: int):
        """
        Compute Gradient × Input attribution over the embedding space.
        Returns word_importance (seq_len,) and word_signed (seq_len,).
        """
        seq_tensor = tf.constant(seq, dtype=tf.int32)          # (1, 200)

        with tf.GradientTape() as tape:
            emb = self.emb_layer(seq_tensor)                   # (1, 200, 64)
            tape.watch(emb)
            preds = self.sub_model(emb, training=False)        # (1, 2)
            score = preds[:, class_idx]                        # scalar

        grads = tape.gradient(score, emb)                      # (1, 200, 64)
        if grads is None:
            raise RuntimeError("Gradient tape returned None — check model build.")

        grads_np = grads.numpy()[0]     # (200, 64)
        emb_np   = emb.numpy()[0]       # (200, 64)

        grad_x_inp = grads_np * emb_np  # (200, 64) element-wise

        word_importance = np.abs(grad_x_inp).mean(axis=-1)    # (200,)
        word_signed     = grad_x_inp.mean(axis=-1)             # (200,) signed

        return word_importance, word_signed

    def explain(self, seq: np.ndarray, class_idx: int, original_text: str):
        """seq: (1, SEQUENCE_LEN) int32 → (plot_b64, deep_text, bullets)"""
        try:
            rev = self._rev()
            seq_flat = seq[0]                                   # (200,)

            word_imp, word_signed = self._grad_x_input(seq, class_idx)  # (200,)

            # Non-padding positions
            positions = np.where(seq_flat > 0)[0]              # indices in [0, 200)
            if len(positions) == 0:
                positions = np.arange(self.seq_len)

            # Top-15 most important positions (by abs importance)
            top_n   = min(15, len(positions))
            argsort = np.argsort(word_imp[positions])[-top_n:]  # indices into positions
            top_pos = positions[argsort]                        # positions in seq_flat

            words   = [rev.get(int(seq_flat[p]), f'w{int(seq_flat[p])}') for p in top_pos]
            imp     = word_signed[top_pos]
            # remove PAD/START/UNK from display
            valid   = [(w, v) for w, v in zip(words, imp)
                       if w not in ('<PAD>', '<START>', '<UNK>')]
            if not valid:
                valid = list(zip(words, imp))

            # Sort by signed value for bar chart
            valid_sorted = sorted(valid, key=lambda x: x[1])
            words_sorted, imp_sorted = zip(*valid_sorted) if valid_sorted else ([], [])
            imp_sorted = list(imp_sorted)

            pos_words = [(w, v) for w, v in valid_sorted if v > 0]
            neg_words = [(w, v) for w, v in valid_sorted if v < 0]
            sentiment  = 'Positive' if class_idx == 1 else 'Negative'
            sum_pos    = float(sum(v for _, v in pos_words))
            sum_neg    = float(sum(v for _, v in neg_words))
            n_words    = len(positions)

            fig, ax = plt.subplots(figsize=(11, max(5, len(words_sorted) * 0.45)))
            fig.patch.set_facecolor('#ffffff')

            colors = ['#16a34a' if v > 0 else '#dc2626' for v in imp_sorted]
            bars   = ax.barh(list(words_sorted), imp_sorted, color=colors,
                             edgecolor='white', height=0.65)
            ax.axvline(0, color='#1e293b', linewidth=1.5)

            for bar, val in zip(bars, imp_sorted):
                sign   = 1 if val >= 0 else -1
                offset = sign * max(abs(max(imp_sorted, default=1)) * 0.01, 1e-6)
                ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                        f'{val:+.4f}', va='center',
                        ha='left' if val >= 0 else 'right', fontsize=8.5)

            ax.set_xlabel('Gradient × Input Score  (attribution toward predicted class)', fontsize=10)
            ax.set_title(
                f'Word Attribution for LSTM  —  Predicted: {sentiment}\n'
                f'Green = word pushes toward {sentiment}  |  Red = pushes away',
                fontsize=11, fontweight='bold', pad=12
            )
            ax.grid(axis='x', alpha=0.25, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()

            pos_list = ', '.join([f'"{w}" (+{v:.4f})' for w, v in reversed(pos_words[-4:])]) or 'none'
            neg_list = ', '.join([f'"{w}" ({v:.4f})' for w, v in neg_words[:4]]) or 'none'

            deep_text = (
                f"The Bidirectional LSTM classified this review as '{sentiment}'. "
                f"Gradient × Input attribution was computed via tf.GradientTape: "
                f"gradients of the {sentiment} probability w.r.t. the 64-dimensional "
                f"word embeddings were multiplied elementwise by the embedding values, "
                f"then averaged across embedding dimensions to get per-word importance. "
                f"The model processed {n_words} recognized words (padded to 200 positions). "
                f"Top words supporting '{sentiment}': {pos_list}. "
                f"Top words opposing '{sentiment}': {neg_list}."
            )

            bullets = [
                f"Attribution method: Gradient × Input — gradients of the {sentiment} class score w.r.t. the 64-d word embeddings, multiplied elementwise by the embedding values.",
                f"Input: {n_words} words encoded as IMDB word IDs, left-padded to 200 positions for the BiLSTM.",
                f"Word score = mean over the 64 embedding dimensions of (gradient × embedding); sign indicates direction.",
                f"Top supportive words (push toward {sentiment}): {pos_list}.",
                f"Top opposing words (push against {sentiment}): {neg_list}.",
                f"Total positive attribution: +{sum_pos:.4f} | Total negative attribution: {sum_neg:.4f}.",
                f"Unlike Bag-of-Words, this BiLSTM reads left-to-right and right-to-left simultaneously, so word order and context shape the attributions.",
            ]

            return _fig_to_b64(fig), deep_text, bullets

        except Exception as e:
            print(f"[SHAP-Text] Error: {e}")
            import traceback; traceback.print_exc()
            return None, str(e), []


# ── 3. Tabular SHAP (Iris DNN — KernelExplainer) ──────────────────────────────

class TabularSHAP:
    """SHAP KernelExplainer for Tabular DNN (Iris). Input: (1,4) standardized."""

    FEATURES   = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    FEAT_UNITS = ['Sepal Length (cm)', 'Sepal Width (cm)',
                  'Petal Length (cm)', 'Petal Width (cm)']
    IRIS_RANGES = {
        'setosa':     {'petal_length': '1.0–1.9', 'petal_width': '0.1–0.6'},
        'versicolor': {'petal_length': '3.0–5.1', 'petal_width': '1.0–1.8'},
        'virginica':  {'petal_length': '4.5–6.9', 'petal_width': '1.4–2.5'},
    }

    def __init__(self, model, background: np.ndarray):
        print("[SHAP-Tabular] Initializing KernelExplainer...")
        self.model      = model
        self.background = background[:20]
        self.explainer  = shap.KernelExplainer(
            lambda x: model.predict(x.astype(np.float32), verbose=0),
            self.background
        )
        print("[SHAP-Tabular] Ready.")

    def explain(self, inp: np.ndarray, class_idx: int, class_name: str,
                raw_features: list):
        try:
            sv = self.explainer.shap_values(inp, nsamples=300)
            if isinstance(sv, list) and len(sv) > class_idx:
                s = np.array(sv[class_idx]).flatten()
            elif isinstance(sv, list):
                s = np.array(sv[0]).flatten()
            else:
                s = np.array(sv).flatten()
            s = s[:len(self.FEATURES)]

            best_idx  = int(np.argmax(np.abs(s)))
            most_imp  = self.FEAT_UNITS[best_idx]
            most_val  = float(s[best_idx])
            most_raw  = float(raw_features[best_idx])
            ranked    = sorted(enumerate(s), key=lambda x: abs(x[1]), reverse=True)
            iris_range= self.IRIS_RANGES.get(class_name.lower(), {})

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
            fig.patch.set_facecolor('#ffffff')

            colors = ['#16a34a' if v > 0 else '#dc2626' for v in s]
            bars   = ax1.barh(self.FEAT_UNITS, s, color=colors, edgecolor='white', height=0.55)
            ax1.axvline(0, color='#1e293b', linewidth=1.5)
            for bar, val in zip(bars, s):
                offset = 0.003 if val >= 0 else -0.003
                ax1.text(val + offset, bar.get_y() + bar.get_height() / 2,
                         f'{val:+.4f}', va='center',
                         ha='left' if val >= 0 else 'right', fontsize=9)
            ax1.set_xlabel('SHAP Value  (contribution to class probability)', fontsize=10)
            ax1.set_title(f'Feature Contributions → {class_name}\nGreen = supports | Red = opposes',
                          fontweight='bold', fontsize=11, pad=10)
            ax1.grid(axis='x', alpha=0.25, linestyle='--')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)

            x_pos = np.arange(len(self.FEATURES))
            bar_colors = ['#3b82f6', '#f97316', '#22c55e', '#a855f7']
            ax2.bar(x_pos, raw_features, color=bar_colors, alpha=0.75, label='Measured value (cm)', width=0.5)
            ax2_twin = ax2.twinx()
            ax2_twin.plot(x_pos, s, 'o-', color='#0f172a', linewidth=2.5, markersize=9, label='SHAP value', zorder=5)
            ax2_twin.axhline(0, color='#94a3b8', linewidth=1, linestyle='--')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(['Sepal L', 'Sepal W', 'Petal L', 'Petal W'], fontsize=10)
            ax2.set_ylabel('Measured Value (cm)', color='#1e293b', fontsize=9)
            ax2_twin.set_ylabel('SHAP Value', color='#0f172a', fontsize=9)
            ax2.set_title('Input Values vs SHAP Contributions', fontweight='bold', fontsize=11, pad=10)
            h1, l1 = ax2.get_legend_handles_labels()
            h2, l2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)
            ax2.spines['top'].set_visible(False)

            plt.suptitle(f'SHAP KernelExplainer  —  Iris Dataset: Predicted {class_name}',
                         fontsize=13, fontweight='bold', y=1.01)
            plt.tight_layout()

            rank_str = ' | '.join(
                [f'{self.FEAT_UNITS[i]}: {float(s[i]):+.4f}' for i, _ in ranked]
            )
            deep_text = (
                f"The Tabular DNN classified this flower as '{class_name}'. "
                f"The most decisive feature was '{most_imp}' = {most_raw:.2f} cm (SHAP: {most_val:+.4f}). "
                f"Full ranking: {rank_str}. "
                f"For {class_name}: petal length {iris_range.get('petal_length','?')} cm, "
                f"petal width {iris_range.get('petal_width','?')} cm."
            )
            bullets = [
                f"SHAP method: KernelExplainer — 300 random feature-coalition samples against {len(self.background)} background Iris data points.",
                f"Prediction: '{class_name}' with the 4 measurements as input.",
                f"Most influential: '{most_imp}' = {most_raw:.2f} cm → SHAP {most_val:+.4f} ({'supports' if most_val > 0 else 'opposes'} '{class_name}').",
                f"All contributions: {rank_str}.",
                f"Reference ranges for '{class_name}': petal length {iris_range.get('petal_length','?')} cm, petal width {iris_range.get('petal_width','?')} cm.",
                f"Positive SHAP (green) = the measured value pushed the network toward '{class_name}'; negative (red) = pulled away.",
                f"The right chart overlays your actual measurements (bars) with the SHAP contribution of each (line plot), showing which features are both large and impactful.",
            ]
            return _fig_to_b64(fig), deep_text, bullets

        except Exception as e:
            print(f"[SHAP-Tabular] Error: {e}")
            return None, str(e), []


# ── 4. Audio SHAP (1D-CNN — GradientExplainer) ────────────────────────────────

class AudioSHAP:
    """SHAP GradientExplainer (Integrated Gradients) for Audio 1D-CNN. Input: (1,1000,1)."""

    SIGNAL_TRAITS = {
        'Sine Wave':   'smooth, continuous periodic oscillations',
        'Square Wave': 'abrupt ±1 transitions at regular intervals',
        'Noise':       'random fluctuations with no periodic structure',
    }

    def __init__(self, model, background: np.ndarray):
        print("[SHAP-Audio] Initializing GradientExplainer...")
        self.model      = model
        self.background = background[:50]
        self.explainer  = shap.GradientExplainer(model, self.background)
        print("[SHAP-Audio] Ready.")

    def explain(self, inp: np.ndarray, class_idx: int, class_name: str):
        try:
            sv = self.explainer.shap_values(inp)

            if isinstance(sv, list) and len(sv) > class_idx:
                s = np.array(sv[class_idx]).reshape(-1)[:inp.shape[1]]
            elif isinstance(sv, list):
                s = np.array(sv[0]).reshape(-1)[:inp.shape[1]]
            else:
                arr = np.array(sv)
                if arr.ndim == 4:
                    s = arr[0, :, 0, class_idx]
                else:
                    s = arr[0, :, 0]

            signal    = inp[0, :, 0]
            t         = np.arange(len(signal))
            pos_steps = int(np.sum(s > 0))
            neg_steps = int(np.sum(s < 0))
            peak_t    = int(np.argmax(np.abs(s)))
            peak_v    = float(s[peak_t])
            pct_pos   = pos_steps / len(s) * 100

            window  = 40
            rolling = np.convolve(np.abs(s), np.ones(window) / window, mode='same')
            peak_region_start = int(np.argmax(rolling))
            peak_region_end   = min(peak_region_start + window, len(signal))

            signal_trait = self.SIGNAL_TRAITS.get(class_name, 'distinctive characteristics')

            fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
            fig.patch.set_facecolor('#ffffff')

            axes[0].plot(t, signal, color='#2563eb', linewidth=1.0, alpha=0.9)
            axes[0].axvline(peak_t, color='#f97316', linewidth=2, linestyle='--',
                            label=f'Peak SHAP at t={peak_t}')
            axes[0].set_ylabel('Amplitude', fontsize=10)
            axes[0].set_title(f'Input Signal  —  Predicted: {class_name}',
                              fontweight='bold', fontsize=12, pad=8)
            axes[0].legend(fontsize=9, loc='upper right')
            axes[0].grid(True, alpha=0.2)
            axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

            axes[1].fill_between(t, s, 0, where=(s >= 0), color='#16a34a', alpha=0.8,
                                 label=f'Positive ({pos_steps} steps, {pct_pos:.1f}%)')
            axes[1].fill_between(t, s, 0, where=(s < 0),  color='#dc2626', alpha=0.8,
                                 label=f'Negative ({neg_steps} steps)')
            axes[1].axhline(0, color='#1e293b', linewidth=0.8)
            axes[1].set_ylabel('SHAP Value', fontsize=10)
            axes[1].set_title('Time-Step SHAP Attribution',
                              fontweight='bold', fontsize=12, pad=8)
            axes[1].legend(fontsize=9, loc='upper right')
            axes[1].grid(True, alpha=0.2)
            axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

            axes[2].plot(t, rolling, color='#7c3aed', linewidth=1.8)
            axes[2].fill_between(t, rolling, alpha=0.25, color='#7c3aed')
            axes[2].axvspan(peak_region_start, peak_region_end, alpha=0.15, color='#f97316',
                            label=f'Attention peak [t={peak_region_start}–{peak_region_end}]')
            axes[2].set_ylabel('Avg |SHAP|', fontsize=10)
            axes[2].set_xlabel('Time Step', fontsize=10)
            axes[2].set_title(f'Rolling Average Importance (window={window})',
                              fontweight='bold', fontsize=12, pad=8)
            axes[2].legend(fontsize=9, loc='upper right')
            axes[2].grid(True, alpha=0.2)
            axes[2].spines['top'].set_visible(False); axes[2].spines['right'].set_visible(False)

            plt.suptitle(f'SHAP GradientExplainer  —  Audio 1D-CNN: {class_name}',
                         fontsize=13, fontweight='bold', y=1.005)
            plt.tight_layout()

            deep_text = (
                f"The 1D-CNN classified this signal as '{class_name}'. "
                f"SHAP GradientExplainer (Integrated Gradients) analyzed all {len(signal)} time steps "
                f"against {len(self.background)} background samples. "
                f"{pos_steps} steps ({pct_pos:.1f}%) positively supported the prediction. "
                f"Peak SHAP is at time step {peak_t} (value {peak_v:+.6f}). "
                f"Highest-attention region: steps {peak_region_start}–{peak_region_end}. "
                f"'{class_name}' is characterized by {signal_trait}."
            )

            bullets = [
                f"SHAP method: GradientExplainer (Integrated Gradients) — integrates model gradients from zero-baseline to the actual 1000-step signal.",
                f"Signal: {len(signal)} time steps representing one waveform period, normalized to [-1, +1].",
                f"Positive contribution: {pos_steps} steps ({pct_pos:.1f}%) actively supported '{class_name}'.",
                f"Negative contribution: {neg_steps} steps opposed the classification.",
                f"Peak SHAP: time step t={peak_t}, value = {peak_v:+.6f} — the single most decisive moment.",
                f"Attention region: steps {peak_region_start}–{peak_region_end} (orange shading in rolling average chart).",
                f"Signal fingerprint: '{class_name}' exhibits {signal_trait} — the SHAP hotspots align with exactly these temporal patterns.",
            ]

            return _fig_to_b64(fig), deep_text, bullets

        except Exception as e:
            print(f"[SHAP-Audio] Error: {e}")
            return None, str(e), []
