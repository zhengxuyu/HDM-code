"use strict";
const el = (id) => document.getElementById(id);
function seconds(value) { return Number.isFinite(value) ? `${value.toFixed(3)} s` : "—"; }
function ms(value) {
  if (!Number.isFinite(value)) return "—";
  return value < 1 ? `${(value * 1000).toFixed(0)} ms` : `${value.toFixed(3)} s`;
}

/** Where one request spent its time: transport, loading, and each server stage.
 *
 * Transport is the client round trip minus the total the server reports for
 * itself, so it holds TCP, the SSH tunnel, serialisation and reading the
 * response -- everything outside the server's own clock, not the network alone.
 *
 * The stages listed under the decision are the ones that call a model. Choosing
 * the policy and typing the answers are not listed, being zero to the
 * millisecond, so these rows sum to slightly less than the decision above them
 * rather than exactly to it.
 */
function renderTiming(roundtrip, timing) {
  const rows = [
    ["Round trip", roundtrip, "measured by this page", "total"],
    ["Transport and serialisation", roundtrip - timing.total_seconds,
     "the round trip minus the server's own total", "sub"],
    ["Loading the model", timing.model_setup_seconds,
     "non-zero on the first request only", "sub"],
    ["Server decision", timing.decision_seconds, "all of engine.evaluate", "total"],
    ["\u2003Building candidates", timing.schema_seconds,
     "a question that arrives without options gets them here; this is a model call", "sub"],
    ["\u2003Fast path", timing.fast_seconds,
     "one prefill, the readout, and the self-routing block", "sub"],
    ["\u2003Slow path", timing.slow_seconds, "escalated questions only", "sub"],
  ];
  const body = document.createElement("tbody");
  for (const [label, value, note, kind] of rows) {
    if (!Number.isFinite(value)) continue;
    const row = document.createElement("tr");
    row.className = kind;
    const name = document.createElement("td"); name.textContent = label;
    const number = document.createElement("td"); number.textContent = ms(value);
    const hint = document.createElement("td"); hint.textContent = note;
    row.append(name, number, hint); body.append(row);
  }
  el("timing").replaceChildren(body);
}
function pct(value) { return Number.isFinite(value) ? `${(value * 100).toFixed(0)}%` : "—"; }
// A typed answer carries its value under a key named after its type (choice,
// score, noul); there is no single `value` field. Reading `.value` showed every
// answer as undefined until a test checked the page against a real response.
function answerValue(answer) {
  if (!answer) return "—";
  const value = answer[answer.type];
  return value === undefined ? JSON.stringify(answer) : String(value);
}
const price = () => Number(el("lambda").value);

async function loadExample() {
  try {
    const response = await fetch("/example");
    if (!response.ok) throw new Error("Could not load the example");
    const example = await response.json();
    el("schema-input").value = JSON.stringify(example.questions, null, 2);
  } catch (error) { el("error").textContent = error.message; el("error").hidden = false; }
}

async function health() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("The service is unavailable");
    const status = await response.json();
    el("health").textContent = status.model_loaded ? "● model ready" : "● connected · model not loaded";
  } catch { el("health").textContent = "○ not connected"; }
}

/** One card per question: mode, answer, v against the price, the distribution. */
function renderAnswers(answers, diagnostics, lambda) {
  const container = el("answers");
  container.replaceChildren();
  for (const [name, field] of Object.entries(diagnostics.fields)) {
    const answer = answers[name];
    const card = document.createElement("article"); card.className = "answer";

    const head = document.createElement("div"); head.className = "answer-heading";
    const title = document.createElement("strong"); title.textContent = name;
    const right = document.createElement("span"); right.className = "answer-tags";
    const mode = document.createElement("span");
    mode.className = `mode mode-${field.system === "two" ? "slow" : "fast"}`;
    mode.textContent = field.system === "two"
      ? `slow · ${field.generated_tokens || 0} tokens` : "fast · 0 tokens";
    const winner = document.createElement("span");
    winner.className = "winner"; winner.textContent = answerValue(answer);
    right.append(mode, winner);
    head.append(title, right);
    card.append(head);

    const gain = field.escalation_gain;
    const stats = document.createElement("div"); stats.className = "routing-stats";
    for (const [label, value, explain] of [
      ["Value of thinking v", gain === undefined ? "no head" : gain.toFixed(4),
       "o3 - o2: the utility the slow mode is expected to add over the fast one"],
      ["Confidence c", Number.isFinite(field.concentration) ? field.concentration.toFixed(3) : "—",
       "the probability the fast answer is right; without a head, the distribution concentration"],
      ["Against the price", gain === undefined ? "—"
        : `${gain.toFixed(3)} ${gain > lambda ? ">" : "≤"} ${lambda}`,
       "escalate when v > lambda"],
    ]) {
      const cell = document.createElement("div"); cell.title = explain;
      const key = document.createElement("span"); key.textContent = label;
      const val = document.createElement("strong"); val.textContent = value;
      cell.append(key, val); stats.append(cell);
    }
    card.append(stats);

    const probabilities = answer && answer.probabilities;
    if (probabilities) {
      for (const [label, probability] of Object.entries(probabilities)) {
        const row = document.createElement("div"); row.className = "probability";
        const text = document.createElement("span"); text.textContent = label;
        const bar = document.createElement("progress"); bar.max = 1; bar.value = probability;
        const value = document.createElement("span");
        value.textContent = `${(probability * 100).toFixed(2)}%`;
        row.append(text, bar, value); card.append(row);
      }
    }
    if (field.reasoning) {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = "What the slow mode wrote (the answer is still assembled from logits, not parsed from this)";
      const content = document.createElement("pre"); content.textContent = field.reasoning;
      details.append(summary, content); card.append(details);
    }
    container.append(card);
  }
}

let running = false;
let shown = null;   // the price the result on screen belongs to

async function run() {
  if (running) return;
  let questions;
  try {
    questions = JSON.parse(el("schema-input").value);
    if (!questions || Array.isArray(questions) || typeof questions !== "object") {
      throw new Error("Questions must be a JSON object.");
    }
    if (Object.keys(questions).length < 1) throw new Error("Give at least one question.");
  } catch (error) {
    el("error").textContent = `Invalid request: ${error.message}`; el("error").hidden = false;
    el("result-status").textContent = "Invalid request"; return;
  }

  const lambda = price();
  running = true;
  el("error").hidden = true;
  el("raw-details").hidden = true;
  for (const id of ["m-escalated", "m-tokens", "m-seconds"]) el(id).textContent = "…";
  el("submit").disabled = true; el("example").disabled = true;
  el("lambda").disabled = true;
  el("output-panel").setAttribute("aria-busy", "true");
  el("result-status").textContent = `Running… λ = ${lambda}`;
  const started = performance.now();
  const timer = setInterval(() => {
    el("elapsed").textContent = `${((performance.now() - started) / 1000).toFixed(1)} s elapsed`;
  }, 100);

  try {
    const response = await fetch("/v1/systemtwo/debug", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        model: "litjev", questions, state: el("state-input").value,
        routing: {lambda, budget: Number(el("budget-input").value) || 512},
      }),
    });
    const text = await response.text();
    const roundtrip = (performance.now() - started) / 1000;
    let payload;
    try { payload = JSON.parse(text); }
    catch { throw new Error(`The service returned HTTP ${response.status} and not JSON. Check the server log.`); }
    if (!response.ok) {
      const detail = payload.detail;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    const {result: standard, diagnostics} = payload;
    const total = Object.keys(diagnostics.fields).length;
    const escalated = diagnostics.routing.escalated.length;
    el("m-escalated").textContent = `${escalated}/${total} · ${pct(escalated / total)}`;
    el("m-tokens").textContent = String(standard.usage.output_tokens);
    el("m-seconds").textContent = seconds(roundtrip);
    renderTiming(roundtrip, diagnostics.timing);
    renderAnswers(standard.answers, diagnostics, lambda);
    el("raw").textContent = JSON.stringify(payload, null, 2);
    el("raw-details").hidden = false;
    el("result-status").textContent = `λ = ${lambda} · ${total} questions · ${escalated} escalated`;
    shown = lambda;
    el("stale").hidden = true;
  } catch (error) {
    el("error").textContent = error.message; el("error").hidden = false;
    el("result-status").textContent = "Request failed";
    for (const id of ["m-escalated", "m-tokens", "m-seconds"]) el(id).textContent = "—";
    el("timing").replaceChildren();
  } finally {
    clearInterval(timer);
    el("elapsed").textContent = `Done · ${((performance.now() - started) / 1000).toFixed(3)} s`;
    el("submit").disabled = false; el("example").disabled = false;
    el("lambda").disabled = false;
    el("output-panel").setAttribute("aria-busy", "false");
    running = false;
    health();
  }
}

// The slider only sets the price; the button sends. Bound to `input` one drag
// would queue dozens of calls at a 27B, and bound to `change` -- which fires on
// release -- every adjustment would cost a round of inference before the next
// one could be tried.
el("lambda").addEventListener("input", () => {
  el("lambda-value").textContent = price().toFixed(2);
  // Say so when what is on screen belongs to another price, so it is not read
  // as this one's.
  if (shown !== null) {
    el("stale").hidden = price() === shown;
  }
});
el("routing-form").addEventListener("submit", (event) => { event.preventDefault(); run(); });
el("example").addEventListener("click", loadExample);

el("lambda-value").textContent = price().toFixed(2);
loadExample(); health();
