README — CulturaTrans
Culturally-Aware Multilingual Translator with Explanation
PES University — GenAI Project | Team 15
Anshul Banda · Anvita Agarwal · Asmi Vishal Kapadnis

What This Project Does
CulturaTrans is a translation system that goes beyond word-for-word conversion. It detects the register (formality level) of the source text, recommends the culturally appropriate tone for the target language, generates three style variants (formal, casual, literal), provides romanized pronunciation, detects gender-ambiguous sentences and produces male/female translation variants, and explains every translation decision in plain language. It also includes a Learning Mode with word breakdown, flashcard generation, and CEFR difficulty gauging.
This is not a wrapper around Google Translate. The cultural intelligence layer, register detection, gender variant generation, and explainability features are original contributions with no equivalent in existing tools.

Supported Languages
English → French, Spanish, Japanese, Hindi, Arabic, Mandarin Chinese, German, Swahili, Brazilian Portuguese, Korean

File Structure
culturally_aware_multilingual_translator/
│
├── app.py                          Main Streamlit UI
├── train_formality.py              Training script for XLM-RoBERTa formality classifier
├── requirements.txt                All Python dependencies
├── .env                            API keys (never commit this)
├── .env.template                   Template showing which keys are needed
│
├── pipeline/
│   ├── __init__.py                 Package exports
│   ├── interfaces.py               All TypedDict contracts (Handoff A/B/C/D)
│   ├── text_preprocessor.py        Unicode normalisation, language detection
│   ├── formality_classifier.py     Heuristic + XLM-RoBERTa formality detection
│   ├── prompt_builder.py           BM25 few-shot retrieval, CoT prompt assembly
│   ├── llm_engine.py               GPT/Gemini/Groq fallback chain, gender variants
│   └── variant_formatter.py        Pipeline orchestrator — calls all modules in order
│
├── eval/
│   ├── __init__.py
│   ├── bleu_comet.py               BLEU + COMET evaluation harness
│   └── ablation.py                 Ablation study (4 conditions)
│
└── data/
    ├── README.md                   Instructions for downloading all datasets
    ├── build_opus_index.py         Downloads OPUS-100 and builds BM25 index
    └── generate_formality_labels.py  Generates LLM formality labels for 7 languages

Tech Stack
ComponentTechnologyUIStreamlitLLM (primary)Google Gemini 1.5 Flash (free)LLM (fallback 1)OpenAI GPT-4o-miniLLM (fallback 2)Groq / Llama 3.3 70B (free)Formality classificationXLM-RoBERTa (fine-tuned) / heuristic fallbackLanguage detectionfastText + langdetectEvaluationsacrebleu (BLEU), unbabel-comet (COMET)TrainingHuggingFace Transformers + Trainer

Setup
bash# 1. Clone the repository
git clone https://github.com/your-username/culturally_aware_multilingual_translator.git
cd culturally_aware_multilingual_translator

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
pip install langdetect google-generativeai

# 4. Set up API keys
cp .env.template .env
# Open .env and add at least one key (Gemini recommended — it's free)

# 5. Run the app
streamlit run app.py
Minimum requirement: One API key. Gemini is free with no credit card — get it at https://aistudio.google.com/app/apikey

API Keys
KeyWhere to getCostGEMINI_API_KEYhttps://aistudio.google.com/app/apikeyFreeOPENAI_API_KEYhttps://platform.openai.com/api-keysPay-as-you-goGROQ_API_KEYhttps://console.groq.comFreeANTHROPIC_API_KEYhttps://console.anthropic.comPay-as-you-go

Running Evaluations (Week 3)
bash# BLEU + COMET on all 10 languages
python eval/bleu_comet.py --all --n 100

# Ablation study
python eval/ablation.py --lang ja --n 50

# Build OPUS retrieval index (run once)
python data/build_opus_index.py

# Generate formality labels for training
python data/generate_formality_labels.py

# Train formality classifier (after downloading GYAFC)
python train_formality.py

What Each Team Member Built
Anvita Agarwal — LLM engine, formality classifier, prompt builder, full pipeline orchestrator, Streamlit UI, evaluation harness, gender detection
Asmi Vishal Kapadnis — CSI annotation pipeline, mDeBERTa classifier, sensitivity flagging lexicon, context memory, explainability UI, attention heatmap
Anshul Banda — Speech pipeline (Whisper ASR), SeamlessM4T TTS, learning layer (word breakdown, flashcards, CEFR difficulty gauge)