# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python:3.11

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./backend_requirements.txt backend_requirements.txt
RUN pip install --no-cache-dir --upgrade -r backend_requirements.txt

COPY --chown=user . /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
