# Medical Study Assistant -- Hugging Face Docker Space (also works on any Docker host).
FROM python:3.11-slim

# Hugging Face runs the container as user 1000; Chroma needs to WRITE to its database folder.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# CPU-only PyTorch (the default wheel is several GB bigger because it ships CUDA).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements-full.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

USER user
ENV HOME=/home/user HF_HOME=/home/user/.cache/huggingface PATH=/home/user/.local/bin:$PATH

# Download the embedding model INTO the image, so the first visitor does not wait for 2 GB.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

COPY --chown=user app.py rag.py study.py ./
COPY --chown=user db ./db

# Secrets (OPENROUTER_API_KEY, OPENROUTER_MODEL, APP_PASSWORD) come from the host's environment, never from the image.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT} --server.address 0.0.0.0 --browser.gatherUsageStats false"]
