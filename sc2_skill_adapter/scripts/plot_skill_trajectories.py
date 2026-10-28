"""Standalone research figure using ReportLab's chart library (SVG output)."""
import json
from pathlib import Path
import sys

from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String, Rect, Circle, Line
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/native-5v5-adaptive-skill-n10-seed100"
OUT = RUN / "trajectory_analysis"
BLUE, RED, GRAY = colors.HexColor("#2166ac"), colors.HexColor("#c94741"), colors.HexColor("#718096")
INK = colors.HexColor("#152d43")
states = {}
for event in map(json.loads, (RUN / "events.jsonl").open(encoding="utf-8")):
    if event["event"] == "skill_observation":
        states[event["episode"], event["step"]] = event["payload"]["state"]
case = json.loads((ROOT / "analysis/case_study/case_data.json").read_text(encoding="utf-8"))["5"]
drawing = Drawing(1100, 740)
drawing.add(Rect(0, 0, 1100, 740, fillColor=colors.white, strokeColor=None))
drawing.add(String(40, 704, "Trajectory diagnosis | Protoss 5v5", fontName="Helvetica-Bold", fontSize=23, fillColor=INK))
drawing.add(String(40, 679, "Recorded skill decisions + frame-by-frame replay comparison. No counterfactual games.", fontSize=11, fillColor=GRAY))


def panel(x, y, title, series, xlow, xhigh, ylow, yhigh, xstep, ystep, xlabel, ylabel):
    drawing.add(String(x, y+231, title, fontName="Helvetica-Bold", fontSize=13, fillColor=INK))
    plot = LinePlot()
    plot.x, plot.y, plot.width, plot.height = x+46, y+34, 425, 174
    plot.data = series
    plot.xValueAxis.valueMin, plot.xValueAxis.valueMax, plot.xValueAxis.valueStep = xlow, xhigh, xstep
    plot.yValueAxis.valueMin, plot.yValueAxis.valueMax, plot.yValueAxis.valueStep = ylow, yhigh, ystep
    plot.xValueAxis.labels.fontSize = plot.yValueAxis.labels.fontSize = 9
    plot.xValueAxis.strokeColor = plot.yValueAxis.strokeColor = GRAY
    plot.yValueAxis.visibleGrid = True
    plot.yValueAxis.gridStrokeColor = colors.HexColor("#e6edf3")
    for i in range(len(series)):
        plot.lines[i].strokeColor = [BLUE, RED][i % 2]
        plot.lines[i].strokeWidth = 2
    drawing.add(plot)
    drawing.add(String(x+220, y+5, xlabel, fontSize=10, fillColor=GRAY))
    drawing.add(String(x+2, y+214, ylabel, fontSize=9, fillColor=GRAY))
    return plot


ys = [(s, states[2, s]["allies"]["u0"]["position"][1]) for s in range(70, 85)]
p = panel(40, 382, "A. Episode 2: chase / return-to-center oscillation",
          [ys], 70, 84, 14, 16, 2, .5, "Decision step (zero-based)", "Zealot u0: Y coordinate")
for step, y in ys:
    visible = bool(states[2, step]["visible_enemies"])
    drawing.add(Circle(p.x+(step-70)/14*p.width, p.y+(y-14)/2*p.height, 3.5,
                       fillColor=BLUE if visible else RED, strokeColor=colors.white))
drawing.add(String(90, 633, "Blue dots: enemy visible     Red dots: observation empty", fontSize=10, fillColor=GRAY))

hps = [(s, sum(states[2, s]["allies"]["u0"][k] for k in ("health", "shield"))) for s in range(70, 85)]
panel(575, 382, "B. Episode 2: health drains during oscillation",
      [hps], 70, 84, 0, 60, 2, 15, "Decision step (zero-based)", "Health + shield")
drawing.add(String(620, 633, "No attack orders in steps 70-84; final defeat follows step 84.", fontSize=10, fillColor=GRAY))

data = [[(r["step"], r["e3_ehp"]) for r in case[name]["timeline"] if r["step"] <= 40]
        for name in ("nearest", "skill")]
p = panel(40, 75, "C. Episode 5: enemy Stalker e3 survives longer",
          data, 0, 40, 0, 160, 10, 40, "Decision step (zero-based)", "Enemy e3: health + shield")
xx = p.x + 21/40*p.width
drawing.add(Line(xx, p.y, xx, p.y+p.height, strokeColor=GRAY, strokeDashArray=[3, 3]))
drawing.add(String(xx+5, p.y+148, "skill changes u0 target", fontSize=8, fillColor=GRAY))

data = [[(r["step"], sum(r[f"u{i}_ehp"] for i in range(5))) for r in case[name]["timeline"]]
        for name in ("nearest", "skill")]
panel(575, 75, "D. Episode 5: team durability diverges",
      data, 0, 90, 0, 1000, 15, 250, "Decision step (zero-based)", "Team health + shield")
for x, color, name in ((90, BLUE, "nearest (win)"), (250, RED, "skill (loss)")):
    drawing.add(Line(x, 325, x+23, 325, strokeColor=color, strokeWidth=2.5))
    drawing.add(String(x+30, 322, name, fontSize=11, fillColor=INK))
drawing.add(String(620, 325, "C/D use replay ground truth for retrospective analysis.", fontSize=10, fillColor=GRAY))
drawing.add(String(40, 34, "Changing target can be useful; this figure shows sequence and divergence, not isolated causal effect.", fontSize=10, fillColor=GRAY))
drawing.add(String(40, 17, "A/B use logged partial observations. Enemy disappearance is not counted as death.", fontSize=10, fillColor=GRAY))
from reportlab.graphics import renderSVG
renderSVG.drawToFile(drawing, str(OUT / "trajectory_diagnostics.svg"))
print(OUT / "trajectory_diagnostics.svg")
