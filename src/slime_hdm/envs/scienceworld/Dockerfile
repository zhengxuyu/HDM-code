# ScienceWorld in a sandbox, because the cluster cannot host it.
#
# The environment needs a JVM: scienceworld drives a Scala simulator over py4j.
# The Slurm cluster has no Java and no container runtime, and the GPUs are there,
# so the split is the one the browser demo already uses -- the environment runs
# here and asks a decision service on the cluster for each step. What crosses the
# network is a state, a list of admissible actions, and the decision.
#
# The JRE is copied from the Temurin image rather than installed with apt,
# because apt's signature check fails under qemu on an Apple Silicon host and
# this image has to build on a laptop as well as on the machine that runs it.
#
#   docker build --platform linux/amd64 -t hdm-scienceworld \
#     -f src/slime_hdm/envs/scienceworld/Dockerfile .
#   docker run --rm --network host -v "$PWD/out":/out hdm-scienceworld \
#     --decision-url http://localhost:30004 --episodes 30 --output /out/gate
FROM eclipse-temurin:17-jre-jammy AS jre

FROM python:3.11-slim
COPY --from=jre /opt/java/openjdk /opt/java/openjdk
ENV JAVA_HOME=/opt/java/openjdk PATH="/opt/java/openjdk/bin:${PATH}" PYTHONUNBUFFERED=1
RUN java -version

RUN pip install --no-cache-dir scienceworld numpy safetensors pydantic
# The simulator jar ships with the wheel; touching the environment at build time
# proves the JVM bridge works here rather than at the first step of a run.
RUN python -c "from scienceworld import ScienceWorldEnv; e = ScienceWorldEnv('', envStepLimit=10); print('tasks:', len(e.get_task_names())); e.close()"

COPY pyproject.toml README.md /src/
COPY src /src/src
RUN pip install --no-cache-dir -e /src --no-deps
WORKDIR /src
ENTRYPOINT ["python", "-m", "slime_hdm.envs.scienceworld.gate"]
