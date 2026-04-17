"""
eval/attention_heatmap.py
--------------------------
Asmi's Component — MarianMT Attention Heatmap

Extracts encoder-decoder cross-attention from Helsinki-NLP MarianMT
and renders it as an interactive heatmap showing which source words
align to which target words.

Why MarianMT and not Claude:
  Claude is a closed API — its internal attention weights are inaccessible.
  MarianMT is open-source (HuggingFace) and exposes all attention matrices.
  We use MarianMT ONLY for the visualization. The actual translation still
  comes from Claude. We explicitly state this in the paper.

Usage:
  from eval.attention_heatmap import render_attention_heatmap
  html = render_attention_heatmap("Let's break the ice", "ja")
  # Returns HTML string — embed in Streamlit with st.components.v1.html(html)

  # Or standalone:
  python3 eval/attention_heatmap.py --text "Let's break the ice" --target ja
"""

import os, sys, re, logging
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logger = logging.getLogger(__name__)

# Helsinki-NLP MarianMT model names for each language pair
MARIAN_MODELS = {
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "es": "Helsinki-NLP/opus-mt-en-es",
    "de": "Helsinki-NLP/opus-mt-en-de",
    "ja": "Helsinki-NLP/opus-mt-en-jap",
    "zh": "Helsinki-NLP/opus-mt-en-zh",
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "ar": "Helsinki-NLP/opus-mt-en-ar",
    "ko": "Helsinki-NLP/opus-mt-en-ko",
    "pt": "Helsinki-NLP/opus-mt-en-ROMANCE",
    "sw": "Helsinki-NLP/opus-mt-en-swc",
}

# Cache loaded models so we don't reload every call
_model_cache = {}
_tokenizer_cache = {}


def _load_marian(target_lang: str):
    """Load and cache a MarianMT model for the given target language."""
    if target_lang in _model_cache:
        return _model_cache[target_lang], _tokenizer_cache[target_lang]

    model_name = MARIAN_MODELS.get(target_lang)
    if not model_name:
        raise ValueError(f"No MarianMT model for language: {target_lang}")

    try:
        from transformers import MarianMTModel, MarianTokenizer
        logger.info(f"Loading MarianMT: {model_name} ...")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model     = MarianMTModel.from_pretrained(model_name)
        model.eval()
        _model_cache[target_lang]    = model
        _tokenizer_cache[target_lang] = tokenizer
        logger.info("MarianMT loaded")
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Could not load MarianMT {model_name}: {e}")


def extract_attention(
    source_text: str,
    target_lang: str,
) -> Optional[dict]:
    """
    Run a forward pass through MarianMT and extract cross-attention.

    Returns:
        {
          "source_tokens": list of str,
          "target_tokens": list of str,
          "attention":     2D list [target_len][source_len] of floats,
          "translation":   str (MarianMT's translation, not Claude's)
        }
    Or None if MarianMT is not available.
    """
    try:
        import torch
        model, tokenizer = _load_marian(target_lang)
    except Exception as e:
        logger.warning(f"MarianMT unavailable: {e}")
        return None

    try:
        # Tokenise source
        inputs = tokenizer(
            [source_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        # Forward pass — output attentions
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                output_attentions=True,
                return_dict_in_generate=True,
                max_new_tokens=128,
            )

        # Decode translation
        translation = tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True
        )

        # Extract cross-attention from last decoder layer, average across heads
        # cross_attentions is a tuple of (n_steps,) each being
        # (n_layers, batch, n_heads, 1, src_len)
        cross_attentions = outputs.cross_attentions
        if not cross_attentions:
            return None

        # Stack: shape [n_steps, n_layers, n_heads, src_len]
        # Use last layer, average over heads
        last_layer_idx = -1
        attn_matrix = []
        for step_attn in cross_attentions:
            # step_attn: tuple of tensors, one per layer
            # each tensor: (batch=1, n_heads, 1, src_len)
            last_layer = step_attn[last_layer_idx]  # (1, n_heads, 1, src_len)
            avg_heads  = last_layer[0].mean(dim=0).squeeze()  # (src_len,)
            attn_matrix.append(avg_heads.tolist())

        # Get readable token strings
        src_tokens = tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0].tolist(),
            skip_special_tokens=True
        )
        tgt_tokens = tokenizer.convert_ids_to_tokens(
            outputs.sequences[0].tolist(),
            skip_special_tokens=True
        )

        # Trim attention matrix to match token lengths
        n_tgt = min(len(tgt_tokens), len(attn_matrix))
        n_src = len(src_tokens)
        matrix = [row[:n_src] for row in attn_matrix[:n_tgt]]

        return {
            "source_tokens": src_tokens,
            "target_tokens": tgt_tokens[:n_tgt],
            "attention":     matrix,
            "translation":   translation,
        }

    except Exception as e:
        logger.warning(f"Attention extraction failed: {e}")
        return None


def render_attention_heatmap(
    source_text: str,
    target_lang: str,
    height: int = 400,
) -> str:
    """
    Render an interactive attention heatmap as an HTML string.

    Uses Plotly.js loaded from CDN — no additional Python dependencies.
    Embed in Streamlit: st.components.v1.html(html, height=height+50)

    Returns an HTML string, or a plain error message string if MarianMT
    is not available.
    """
    data = extract_attention(source_text, target_lang)

    if data is None:
        return (
            "<div style='padding:16px;color:#a0aec0;background:#1a202c;"
            "border-radius:8px;font-size:13px;'>"
            "⚠️ Attention heatmap unavailable. "
            "Install transformers and download MarianMT models: "
            "<code>pip install transformers sentencepiece</code></div>"
        )

    src_tokens = data["source_tokens"]
    tgt_tokens = data["target_tokens"]
    matrix     = data["attention"]
    translation = data["translation"]

    # Build Plotly heatmap data
    import json
    matrix_json     = json.dumps(matrix)
    src_tokens_json = json.dumps(src_tokens)
    tgt_tokens_json = json.dumps(tgt_tokens)

    html = f"""
<div style="font-family:sans-serif;background:#1a202c;padding:12px;border-radius:8px;">
  <div style="color:#a0aec0;font-size:11px;margin-bottom:4px;">
    MarianMT alignment (approximation) — Claude's actual translation may differ
  </div>
  <div style="color:#e2e8f0;font-size:12px;margin-bottom:8px;">
    MarianMT: <em>{translation}</em>
  </div>
  <div id="heatmap_{hash(source_text) & 0xFFFFFF}"></div>
</div>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<script>
(function() {{
  var matrix = {matrix_json};
  var srcTokens = {src_tokens_json};
  var tgtTokens = {tgt_tokens_json};
  var divId = "heatmap_{hash(source_text) & 0xFFFFFF}";

  var data = [{{
    z: matrix,
    x: srcTokens,
    y: tgtTokens,
    type: 'heatmap',
    colorscale: [
      [0, '#1a202c'], [0.3, '#2d3748'], [0.6, '#4a5568'],
      [0.8, '#3182ce'], [1.0, '#90cdf4']
    ],
    showscale: true,
    hoverongaps: false,
    hovertemplate: 'Source: %{{x}}<br>Target: %{{y}}<br>Attention: %{{z:.3f}}<extra></extra>'
  }}];

  var layout = {{
    paper_bgcolor: '#1a202c',
    plot_bgcolor: '#2d3748',
    font: {{ color: '#e2e8f0', size: 11 }},
    margin: {{ t: 10, b: 60, l: 80, r: 10 }},
    height: {height},
    xaxis: {{ title: 'Source tokens', tickangle: -30, color: '#a0aec0' }},
    yaxis: {{ title: 'Target tokens', color: '#a0aec0' }},
  }};

  var config = {{ responsive: true, displayModeBar: false }};
  Plotly.newPlot(divId, data, layout, config);
}})();
</script>
"""
    return html


def main():
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",   default="Let's break the ice at the meeting")
    parser.add_argument("--target", default="ja")
    parser.add_argument("--output", default="eval/results/attention_heatmap.html")
    args = parser.parse_args()

    os.makedirs("eval/results", exist_ok=True)
    html = render_attention_heatmap(args.text, args.target)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"<html><body style='background:#0d1117'>{html}</body></html>")

    print(f"Heatmap saved to {args.output}")
    print(f"Open in browser to view.")


if __name__ == "__main__":
    main()