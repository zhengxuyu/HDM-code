# The browser half of the demo: a browser, MiniWoB, and a client. No GPU.
#
# The model half runs behind scripts/serve_decision.sbatch, which holds the
# engine and the head together and answers with a decision. An earlier version
# of this image carried the engine too, because the head was thought to need
# SGLang's in-process Engine API; the HTTP server returns hidden states as well,
# but returning them costs 14.3s per step against 0.7s in-process, all of it
# spent encoding every prompt position as JSON. So the head stays with the
# engine, and this image only drives the browser and draws the clock.
#
#   docker build -t hdm-cua -f demo/Dockerfile .
#   docker run --rm --shm-size=1g -v "$PWD/out":/out hdm-cua \
#     --decision-url http://<host>:30001 --output /out/run1 \
#     --task browsergym/miniwob.click-checkboxes-large --headless
#
# Headed, to watch it: pass -e DISPLAY and mount the X socket, or record the
# container's display with x11vnc. --headless is for a machine with no screen.
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/opt/venv/bin:${PATH}" VIRTUAL_ENV=/opt/venv
RUN uv venv /opt/venv --python 3.11

RUN uv pip install browsergym-core browsergym-miniwob gymnasium playwright \
                   safetensors pydantic numpy
RUN playwright install chromium --with-deps

# browsergym-miniwob registers the tasks but ships none of the pages, so the
# demo has no network and no HTML unless the upstream repo is cloned. Pinned:
# the tasks are identified by file name and the repo renames them.
RUN git clone https://github.com/Farama-Foundation/miniwob-plusplus.git /opt/miniwob \
 && git -C /opt/miniwob checkout 33c3b4ddef8c6eb67c57a29663d844b1eda7e614 \
 && test -f /opt/miniwob/miniwob/html/miniwob/click-button.html
ENV MINIWOB_URL="file:///opt/miniwob/miniwob/html/miniwob/"

COPY pyproject.toml README.md /src/
COPY src /src/src
RUN uv pip install -e /src --no-deps
COPY demo/cua_demo.py /src/demo/cua_demo.py
WORKDIR /src
ENTRYPOINT ["python", "demo/cua_demo.py"]
