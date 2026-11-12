# The browser half of the gate: a browser, the three venues' task packages, and a
# client. No GPU, and no model.
#
# The decision side runs behind scripts/serve_decision.sbatch, which holds the
# engine and the head together. The head stays with the engine because returning
# hidden states over HTTP costs 14.3 s per step against 0.7 s in-process, all of
# it spent encoding every prompt position as JSON (experiments/browsergym). What
# crosses this container's socket is a decision, a few hundred bytes. The gate
# needs no head at all -- it runs two constant policies -- and does not use one if
# the service has it loaded.
#
#   docker build -t htm-gate -f src/slime_htm/envs/Dockerfile \
#     --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
#
#   # Are the site URLs even set? No --groups, no --decision-url, no server.
#   docker run --rm htm-gate --venue webarena --check-env
#
#   # `--list` next. It runs nothing either, so it needs no decision service and
#   # no site, but it does need --groups, because it prints them.
#   docker run --rm --shm-size=1g htm-gate --list \
#     --venue webarena --groups gitlab,shopping,shopping_admin,map,reddit
#
#   # The ceiling measurement: no server, no API key, nothing external. Two
#   # groups and no pooled one -- `click` is the measurement, `beyond-click` is
#   # the control that is expected near zero because the agent cannot type.
#   docker run --rm --shm-size=1g -v "$PWD/out":/out htm-gate \
#     --decision-url http://127.0.0.1:30001 --venue miniwob \
#     --groups click,beyond-click \
#     --output /out/miniwob-gate.json --rows /out/miniwob-gate.rows.jsonl
#
#   # WebArena, one site. --output and --rows must land in the mount, or the
#   # run's only two artefacts die with the container.
#   docker run --rm --shm-size=1g -v "$PWD/out":/out \
#     -e WA_SHOPPING -e WA_SHOPPING_ADMIN -e WA_REDDIT -e WA_GITLAB \
#     -e WA_WIKIPEDIA -e WA_MAP -e WA_HOMEPAGE -e WA_FULL_RESET \
#     htm-gate --decision-url http://127.0.0.1:30001 \
#       --venue webarena --groups gitlab --tasks-per-group 60 \
#       --output /out/gate.json --rows /out/gate.rows.jsonl
#
# Expect `--list` to print, for webarena: gitlab 172, shopping 153,
# shopping_admin 156, map 64, reddit 104, multi 45 -- 694 of the 812 tasks, the
# 118 model-judged ones excluded. Anything else means the image did not get the
# pinned packages, and that is the whole finding: the registries have moved
# between releases, so a different count is a different benchmark.
#
# MiniWoB needs MINIWOB_URL, pointing at a pinned checkout of the pages; see
# scripts/build_browsergym.sbatch, which does the same thing on the cluster
# without a container.
#
# --shm-size is not optional: chromium in a container with the default 64 MB of
# /dev/shm crashes on pages the size of a shopping-admin form.
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/opt/venv/bin:${PATH}" VIRTUAL_ENV=/opt/venv
RUN uv venv /opt/venv --python 3.11

# All three venues in one image: they resolve together on browsergym-core 0.14.3,
# checked with `uv pip compile`, so there is no reason to build three.
#
# Every one is pinned and every one's version goes in the gate's header
# (VENUES[...]["packages"]). The registries have moved between releases, and the
# same task name in two releases is two different tasks. playwright is
# deliberately not pinned here: browsergym-core pins it exactly (==1.44), and a
# second pin would either agree redundantly or fight it.
#
# webarena's package is an unofficial port -- `libwebarena` says "use at your own
# risks" in its own description -- which is a reason to pin it harder, not softer.
# It pulls `openai`, because 118 of the 812 tasks are graded by a model; the gate
# excludes those, so no key is needed and none is passed in.
RUN uv pip install browsergym-core==0.14.3 \
                   browsergym-webarena==0.14.3 \
                   browsergym-workarena==0.5.3 \
                   browsergym-miniwob==0.14.3 \
                   gymnasium playwright safetensors pydantic numpy
RUN playwright install chromium --with-deps

# browsergym-miniwob registers the tasks and ships none of the pages, so without
# this clone `--venue miniwob` has no HTML and no network. Pinned, because the
# tasks are identified by file name and the repo renames them -- the same commit
# scripts/build_browsergym.sbatch uses, and the one the committed action split was
# computed from.
RUN git clone --quiet https://github.com/Farama-Foundation/miniwob-plusplus.git /opt/miniwob \
 && git -C /opt/miniwob checkout --quiet 33c3b4ddef8c6eb67c57a29663d844b1eda7e614 \
 && test -f /opt/miniwob/miniwob/html/miniwob/click-button.html
ENV MINIWOB_URL="file:///opt/miniwob/miniwob/html/miniwob/"

# WorkArena's setup step installs its tasks' data into a live ServiceNow instance,
# and WebArena's sites are seven containers of their own. Neither is in this
# image, and neither runs at build time:
#
#   python -m browsergym.workarena.install          # against the instance
#   # webarena: https://github.com/gasse/webarena-setup/tree/main/webarena
#
COPY pyproject.toml README.md /src/
COPY src /src/src
WORKDIR /src
RUN uv pip install --no-deps -e .

# Which code this is. The COPY above carries no .git, so `commit_now()` would fall
# back to a REVISION file, and without one every row in the run would be labelled
# "unknown" -- a verdict against pre-registered thresholds that cannot be tied to
# a reviewed commit. The `commit <sha>` prefix is the format `commit_in()` parses;
# a bare sha reads as no record at all, which is tested. An empty build arg fails
# the build here rather than producing an untraceable run seven hours later, and
# the gate refuses to start on "unknown" even if some other packaging route
# forgets this line.
ARG GIT_COMMIT
RUN test -n "$GIT_COMMIT" || (echo "build with --build-arg GIT_COMMIT=\$(git rev-parse HEAD)" >&2; exit 1)
RUN printf 'commit %s\n' "$GIT_COMMIT" > /src/REVISION

ENTRYPOINT ["python", "-m", "slime_htm.envs.gate"]
