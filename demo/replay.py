"""Replay a recorded run as one video, three policies on a shared wall clock.

The demo's claim is about time, so the three policies have to be seen against
the same clock rather than one after another. Each policy's frames are held for
the time that step actually took, so a viewer watches `fast` stop early with the
task unfinished, `slow` still working, and `routed` stop early with it done.

Nothing here re-runs the model. It reads `trajectory.json` and the frames beside
it, which is what makes the video reproducible from a recorded run and honest
about where its numbers come from: they are the measured step times, not a
paced animation.

    uv run python demo/replay.py out/run1 --output out/run1/replay.mp4
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

BACKGROUND = (14, 14, 17)
TEXT = (245, 245, 247)
DIM = (150, 150, 158)
GOOD = (140, 233, 154)
BAD = (255, 107, 107)
THINKING = (255, 209, 102)
HEADER = 0
CAPTION = 70
GAP = 14


def font(size):
    """One face at one size. There is no bold: this system ships one mono face."""
    try:
        return ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


def content_box(images, pad=10, gap=60):
    """The task, without the browser around it or the clock beside it.

    MiniWoB draws in a 320x210 frame, so a viewport wide enough to keep the
    in-page clock off the task leaves most of the shot flat grey, with the task
    at one end and the clock at the other. Played back that is a grey band with
    something small in the corner.

    So the crop stops at the first wide empty column run: the task is kept, the
    gap and the clock beyond it are dropped. The clock is no loss here, because
    the panel header below draws the same numbers; it is drawn into the page for
    anyone recording the browser live, where there is no header.
    """
    columns = None
    for image in images:
        background = image.getpixel((image.width - 1, image.height - 1))
        difference = ImageChops.difference(
            image, Image.new("RGB", image.size, background)).convert("L")
        mask = Image.eval(difference, lambda value: 255 if value > 12 else 0)
        rows = [mask.crop((x, 0, x + 1, image.height)).getbbox() is not None
                for x in range(image.width)]
        columns = rows if columns is None else [a or b for a, b in zip(columns, rows)]
    if columns is None or not any(columns):
        return None
    left = columns.index(True)
    right, run = left, 0
    for x in range(left, len(columns)):
        if columns[x]:
            right, run = x, 0
        else:
            run += 1
            if run >= gap:
                break

    box = None
    for image in images:
        background = image.getpixel((image.width - 1, image.height - 1))
        strip = image.crop((left, 0, right + 1, image.height))
        difference = ImageChops.difference(
            strip, Image.new("RGB", strip.size, background)).convert("L")
        found = Image.eval(difference, lambda value: 255 if value > 12 else 0).getbbox()
        if found is None:
            continue
        box = found if box is None else (min(box[1 - 1], found[0]), min(box[1], found[1]),
                                         max(box[2], found[2]), max(box[3], found[3]))
    if box is None:
        return None
    height = images[0].height
    return (max(0, left + box[0] - pad), max(0, box[1] - pad),
            min(images[0].width, left + box[2] + pad), min(height, box[3] + pad))


def timeline(run, frames):
    """(start, end, image) per step, on the policy's own clock.

    A step's frame is the page the step was decided on, taken before the action
    ran. The run's `final_frame` is what the last action produced, and it is
    held from the end of the run onwards, so the video shows where the policy
    arrived rather than the page it left.
    """
    # Boundaries come from each step's recorded wall clock, not from summing the
    # model's time: the two differ by the browser's own work, and a video built
    # on the second shows a policy arriving seconds before its clock says it did.
    spans, clock = [], 0.0
    for step in run["steps"]:
        name = step.get("frame")
        path = frames / name if name else None
        image = Image.open(path).convert("RGB") if path and path.exists() else None
        ends = step.get("elapsed_seconds", clock + step["step_seconds"])
        spans.append((clock, ends, image, step))
        clock = ends
    clock = max(clock, run["total_seconds"])
    ending = run.get("final_frame")
    path = frames / ending if ending else None
    if path and path.exists():
        spans.append((clock, float("inf"), Image.open(path).convert("RGB"),
                      dict(run["steps"][-1], escalated=False, arrived=True)))
    return spans


def at(spans, moment):
    """The frame in force at `moment`, and how far into its step we are.

    The step's own progress is what makes the clock move rather than jump: a
    step lasts seconds and produces one screenshot, so without it the reading
    would only change when the picture does.
    """
    for start, end, image, step in spans:
        if start <= moment < end:
            return image, step, bool(step.get("arrived")), moment - start
    if spans:
        return spans[-1][2], spans[-1][3], True, 0.0
    return None, None, True, 0.0


def wrapped(draw, text, chosen, room):
    lines = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            lines.append("")
            continue
        current = ""
        for word in paragraph.split():
            trial = f"{current} {word}".strip()
            if current and draw.textlength(trial, font=chosen) > room:
                lines.append(current)
                current = word
            else:
                current = trial
        if current:
            lines.append(current)
    return lines


def reasoning_panel(overlay, draw, step, into, scale, width, height):
    """The reasoning, appearing while it is being produced rather than before.

    A step is photographed once, after it has been decided, so an overlay baked
    into that picture is on screen for the whole step: the first frame of a step
    showed reasoning the model had not written yet. Here the step's own timings
    place it. The fast pass runs first and produces nothing, so nothing is drawn
    for `fast_seconds`; after that the text arrives across the seconds the
    generation actually took.

    The reveal is evenly paced across that measured span. Per-token times are
    not recorded, so this says how long the writing took, not when each word of
    it landed.
    """
    reasoning = (step.get("reasoning") or "").replace("</think>", "").strip()
    if not reasoning:
        return
    started = step.get("fast_seconds", 0.0)
    span = max(step.get("think_seconds", 0.0), 1e-6)
    if into <= started:
        return
    shown = reasoning[:max(1, int(len(reasoning) * min((into - started) / span, 1.0)))]

    box = int(360 * scale)
    left = width - box - int(20 * scale)
    top = int(210 * scale)
    chosen = font(int(15 * scale))
    line_height = int(20 * scale)
    room = box - int(28 * scale)
    lines = wrapped(draw, shown, chosen, room)
    visible = int((height - top - int(40 * scale)) / line_height)
    lines = lines[-visible:] if len(lines) > visible else lines
    tall = int(34 * scale) + max(1, len(lines)) * line_height

    draw.rounded_rectangle((left, top, left + box, top + tall),
                           radius=int(12 * scale), fill=(10, 10, 14, 224))
    draw.text((left + int(14 * scale), top + int(10 * scale)), "THINKING",
              font=font(int(14 * scale)), fill=THINKING)
    for number, text in enumerate(lines):
        draw.text((left + int(14 * scale), top + int(30 * scale) + number * line_height),
                  text, font=chosen, fill=(235, 235, 240))


# What each policy is called on screen. `routed` is the method, so it carries
# the method's name; `instant` described how a step felt rather than which mode
# produced it.
SHOWN = {"routed": "HDM", "slow": "SLOW", "fast": "FAST"}


def tokens_by(spans, moment):
    """Generated tokens so far, counted as they would have been produced.

    The two savings this demo is about are seconds and tokens, and a total shown
    only at the end makes the second one an epilogue. A step that is thinking
    contributes in proportion to how far through its generation it is, on the
    same evenly paced schedule the reasoning text is revealed on.
    """
    spent = 0
    for start, end, _, step in spans:
        if step.get("arrived"):
            continue
        if moment >= end:
            spent += step.get("generated_tokens", 0)
        elif moment > start and step.get("escalated"):
            into = moment - start - step.get("fast_seconds", 0.0)
            share = min(max(into, 0.0) / max(step.get("think_seconds", 0.0), 1e-6), 1.0)
            spent += int(step.get("generated_tokens", 0) * share)
    return spent


def panel(size, run, spans, moment):
    """One policy's column: the page, a clock over it, and how it ended."""
    width, height = size
    scale = max(1.0, width / 900)
    caption = int(CAPTION * scale)
    canvas = Image.new("RGB", (width, height + caption), BACKGROUND)
    image, step, finished, into = at(spans, moment)
    if image is not None:
        canvas.paste(image.resize((width, height)), (0, 0))

    # The plate is sized to the text rather than to a constant: the fonts scale
    # with the panel and a fixed width let the status line run off the end of it.
    policy_font = font(int(21 * scale))
    time_font = font(int(52 * scale))
    status_font = font(int(23 * scale))
    elapsed = min(moment, run["total_seconds"])
    clock = f"{elapsed:.1f}s"
    if finished:
        status = "succeeded" if run["succeeded"] else "did not succeed"
        colour = GOOD if run["succeeded"] else BAD
    elif step["escalated"]:
        # The forward pass comes first and thinks nothing, so the counter starts
        # when the generation does. Counting from the top of the step credited
        # the thinking with seconds spent before it began.
        spent = min(max(into - step.get("fast_seconds", 0.0), 0.0), step["think_seconds"])
        status = (f"step {step['step']}  thinking +{spent:.1f}s" if spent > 0
                  else f"step {step['step']}  reading the page")
        colour = THINKING if spent > 0 else (205, 205, 214)
    else:
        status = f"step {step['step']}  fast mode"
        colour = GOOD

    spent = tokens_by(spans, moment)
    cost = f"{spent} tokens"
    name = SHOWN.get(run["policy"], run["policy"].upper())

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    pad = int(20 * scale)
    inset = int(20 * scale)
    plate = int(max(draw.textlength(name, font=policy_font),
                    draw.textlength(clock, font=time_font),
                    draw.textlength(status, font=status_font),
                    draw.textlength(cost, font=status_font)) + 2 * inset)
    tall = int(184 * scale)
    draw.rounded_rectangle((pad, pad, pad + plate, pad + tall),
                           radius=int(16 * scale), fill=(10, 10, 14, 224))

    left = pad + inset
    line = pad + int(14 * scale)
    draw.text((left, line), name, font=policy_font, fill=(205, 205, 214))
    draw.text((left, line + int(24 * scale)), clock, font=time_font, fill=TEXT)
    foot = line + int(88 * scale)
    draw.text((left, foot), status, font=status_font, fill=colour)
    # Seconds and tokens are the two things being saved, so the second one is on
    # the clock rather than in a closing caption.
    draw.text((left, foot + int(30 * scale)), cost, font=status_font,
              fill=(235, 209, 140) if spent else (150, 150, 158))
    if not finished and step is not None:
        reasoning_panel(overlay, draw, step, into, scale, width, height)
    if not finished:
        span = max(step["step_seconds"], 1e-6)
        bar = foot + int(62 * scale)
        draw.rectangle((left, bar, pad + plate - inset, bar + int(4 * scale)),
                       fill=(58, 58, 68, 255))
        draw.rectangle((left, bar, left + int((plate - 2 * inset) * min(into / span, 1.0)),
                        bar + int(4 * scale)), fill=colour)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(canvas)
    bottom = height + int(10 * scale)
    if finished:
        draw.text((int(12 * scale), bottom),
                  f"{run['total_seconds']:.1f}s   {run['generated_tokens']} tokens   "
                  f"{run['escalations']} of {len(run['steps'])} steps thought",
                  font=font(int(20 * scale)), fill=DIM)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="a directory holding trajectory.json and frames/")
    parser.add_argument("--output", help="default: replay.mp4 inside the run")
    parser.add_argument("--credit",
                        help="what ran it: backbone, server, GPU. Right of the task, because "
                             "seconds on a clock mean nothing without the hardware they were "
                             "spent on")
    parser.add_argument("--plan",
                        help="the steps the task asks for, separated by |, written across the "
                             "top so a viewer can follow along rather than infer the route")
    parser.add_argument("--only",
                        help="comma-separated policies to show, in this order. The three "
                             "panels are a run's whole record; a video may want two of them")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="1.0 is real time, which is the point; 2.0 halves the video")
    parser.add_argument("--scale", type=float, default=2.0,
                        help="how much to enlarge each panel")
    parser.add_argument("--no-crop", dest="crop", action="store_false",
                        help="keep the browser chrome, which is mostly flat grey")
    parser.add_argument("--tail", type=float, default=2.5, help="seconds held on the last frame")
    arguments = parser.parse_args()

    run_directory = Path(arguments.run)
    trajectory = json.loads((run_directory / "trajectory.json").read_text())
    frames = run_directory / "frames"
    runs = trajectory["runs"]
    if arguments.only:
        wanted = [name.strip() for name in arguments.only.split(",") if name.strip()]
        by_name = {run["policy"]: run for run in runs}
        missing = [name for name in wanted if name not in by_name]
        if missing:
            raise SystemExit(f"{missing} not in this run, which has "
                             f"{sorted(by_name)}")
        runs = [by_name[name] for name in wanted]
    spans = [timeline(run, frames) for run in runs]

    every = [image for column in spans for _, _, image, _ in column if image]
    if not every:
        raise SystemExit(f"no frames under {frames}")
    box = content_box(every) if arguments.crop else None
    if box is not None:
        spans = [[(start, end, image.crop(box) if image else None, step)
                  for start, end, image, step in column] for column in spans]
        print(f"cropped to {box[2] - box[0]}x{box[3] - box[1]} of "
              f"{every[0].width}x{every[0].height}")
    size = (box[2] - box[0], box[3] - box[1]) if box else every[0].size
    # The task is small on the page, so the video is upscaled rather than shown
    # at a size where the checkboxes cannot be read.
    size = (int(size[0] * arguments.scale), int(size[1] * arguments.scale))
    longest = max(run["total_seconds"] for run in runs)
    count = int((longest + arguments.tail) * arguments.fps / arguments.speed)

    # The band across the top carries the task and the route it asks for. Without
    # it a viewer has to infer both from two browsers moving at different speeds,
    # and spends the video working out what is being attempted instead of
    # watching how much each way of attempting it costs.
    scale = max(1.0, size[0] / 900)
    plan = [part.strip() for part in (arguments.plan or "").split("|") if part.strip()]
    goal_font = font(int(25 * scale))
    plan_font = font(int(21 * scale))
    credit_font = font(int(20 * scale))

    sheet_width = sum(int(size[0]) for _ in runs) + GAP * (len(runs) - 1)
    ruler = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    room = sheet_width - int(36 * scale)
    if arguments.credit:
        room -= int(ruler.textlength(arguments.credit, font=credit_font) + 40 * scale)

    # Wrapped, not shrunk and not cut: the task is the one thing a viewer has to
    # hold in mind for two minutes, and an ellipsis in the middle of it defeats
    # the point of putting it on screen at all.
    lines, current = [], ""
    for word in trajectory["goal"].split():
        trial = f"{current} {word}".strip()
        if current and ruler.textlength(trial, font=goal_font) > room:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    step_height = int(31 * scale)
    band = int(16 * scale) + len(lines) * step_height + (int(38 * scale) if plan else int(6 * scale))

    staging = Path(tempfile.mkdtemp(prefix="hdm-replay-"))
    try:
        for index in range(count):
            moment = index * arguments.speed / arguments.fps
            columns = [panel(size, run, column, moment) for run, column in zip(runs, spans)]
            width = sum(c.width for c in columns) + GAP * (len(columns) - 1)
            sheet = Image.new("RGB", (width, columns[0].height + band), BACKGROUND)
            offset = 0
            for column in columns:
                sheet.paste(column, (offset, band))
                offset += column.width + GAP
            draw = ImageDraw.Draw(sheet)
            if arguments.credit:
                span = draw.textlength(arguments.credit, font=credit_font)
                draw.text((width - int(18 * scale) - span, int(14 * scale)),
                          arguments.credit, font=credit_font, fill=DIM)
            for number, text in enumerate(lines):
                draw.text((int(18 * scale), int(12 * scale) + number * step_height),
                          text, font=goal_font, fill=TEXT)
            if plan:
                x = int(18 * scale)
                y = int(16 * scale) + len(lines) * step_height
                # Not `index`: that is the frame counter this loop sits inside,
                # and shadowing it wrote every frame over the same five files.
                for position, part in enumerate(plan, 1):
                    chip = f"{position}"
                    draw.rounded_rectangle(
                        (x, y, x + int(22 * scale), y + int(24 * scale)),
                        radius=int(6 * scale), fill=(44, 44, 54))
                    draw.text((x + int(7 * scale), y + int(3 * scale)), chip,
                              font=plan_font, fill=(215, 215, 224))
                    x += int(30 * scale)
                    draw.text((x, y + int(3 * scale)), part, font=plan_font, fill=DIM)
                    x += int(draw.textlength(part, font=plan_font) + 34 * scale)
            sheet.save(staging / f"{index:05d}.png")

        output = Path(arguments.output or run_directory / "replay.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(arguments.fps),
             "-i", str(staging / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(output)],
            check=True)
        print(f"{output}  {count} frames  {count / arguments.fps:.1f}s at {arguments.speed}x")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
