"""Generate the survivorship-validation report PDF (Norgate point-in-time trial).

Standalone: build_report.py builds its document at module scope, so it cannot be
imported for its styles without side effects. Palette is kept in sync by hand.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                PageBreak, HRFlowable)

NAVY = HexColor("#0d2340"); BLUE = HexColor("#1f6feb"); GREY = HexColor("#5b6470")
LIGHT = HexColor("#eef2f7"); GREEN = HexColor("#0f7a3d"); RED = HexColor("#b3261e")
AMBER = HexColor("#8a6100"); LINE = HexColor("#c9d2de")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                    fontSize=15, textColor=NAVY, spaceBefore=16, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                    fontSize=11.5, textColor=BLUE, spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontName="Helvetica", fontSize=9.6,
                      leading=14, textColor=HexColor("#1a2330"), spaceAfter=6)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=8.2, textColor=GREY, leading=11)
CELL = ParagraphStyle("CELL", parent=BODY, fontSize=8.4, leading=11, spaceAfter=0)
CELLB = ParagraphStyle("CELLB", parent=CELL, fontName="Helvetica-Bold")
TITLE = ParagraphStyle("TITLE", parent=ss["Title"], fontName="Helvetica-Bold",
                       fontSize=24, textColor=NAVY, leading=28, spaceAfter=4)
SUB = ParagraphStyle("SUB", parent=BODY, fontSize=12, textColor=GREY, spaceAfter=2)
CALLOUT = ParagraphStyle("CALLOUT", parent=BODY, fontSize=10, leading=15,
                         textColor=NAVY, leftIndent=10, rightIndent=10,
                         spaceBefore=4, spaceAfter=4)


def P(t, s=BODY): return Paragraph(t, s)


def bullets(items, s=BODY):
    return [Paragraph("&bull;&nbsp;&nbsp;" + it,
                      ParagraphStyle("b", parent=s, leftIndent=12, spaceAfter=4))
            for it in items]


def table(data, widths, header=True, shade_rows=None):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    sty = [("VALIGN", (0, 0), (-1, -1), "TOP"),
           ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
           ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6)]
    if header:
        sty += [("BACKGROUND", (0, 0), (-1, 0), LIGHT),
                ("LINEBELOW", (0, 0), (-1, 0), 0.9, NAVY)]
    for r in (shade_rows or []):
        sty.append(("BACKGROUND", (0, r), (-1, r), HexColor("#fdf6e3")))
    t.setStyle(TableStyle(sty))
    return t


def box(par_text, color=LIGHT):
    t = Table([[Paragraph(par_text, CALLOUT)]], colWidths=[6.9 * inch], hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), color),
                           ("LEFTPADDING", (0, 0), (-1, -1), 12),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                           ("TOPPADDING", (0, 0), (-1, -1), 9),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                           ("LINEBEFORE", (0, 0), (0, -1), 3, BLUE)]))
    return t


S = []

# ---------------------------------------------------------------- cover
S += [Spacer(1, 0.35 * inch),
      P("Survivorship Validation", TITLE),
      P("Testing The Edge against real point-in-time Russell&nbsp;1000 data", SUB),
      HRFlowable(width="100%", thickness=1.1, color=NAVY, spaceBefore=10, spaceAfter=12),
      P("Prepared 22 July 2026 &nbsp;&bull;&nbsp; Norgate Data trial (2-year window) "
        "&nbsp;&bull;&nbsp; Internal research note", SMALL)]

S += [Spacer(1, 0.12 * inch),
      box("<b>Bottom line.</b> We built the tooling to test whether The Edge's track record "
          "depends on survivorship bias, and ran it. <b>The test came back inconclusive.</b> "
          "The two-year data trial is too short to answer the question: the result swings "
          "between &minus;2% and +55% a year depending purely on which day of the month you "
          "start, which is far larger than the effect we were trying to measure. "
          "An earlier version of this analysis reported a clean pass (+28.5%/yr). "
          "<b>That result was wrong and has been retracted</b> &mdash; a code review found a "
          "look-ahead error and a lucky start date. This report documents both the finding "
          "and the error, because the error is the more useful lesson.")]

# ---------------------------------------------------------------- 1
S += [P("1. &nbsp;The question", H1),
      P("<b>Survivorship bias</b> is the most common way a backtest lies to you. Our stock "
        "universe is built from companies that exist <i>today</i>. That quietly excludes every "
        "company that went bankrupt, got acquired, or was thrown out of the index along the "
        "way. If your strategy would have bought some of those losers, a backtest built only "
        "on today's survivors never shows the damage &mdash; so it looks better than reality.", BODY),
      P("Our universe is missing roughly 89% of delisted names. So the question was simple and "
        "important: <b>is The Edge's performance real, or is it an artifact of only ever seeing "
        "the winners?</b>", BODY),
      P("2. &nbsp;What we built", H1),
      P("Norgate Data sells point-in-time index data: it knows exactly who was in the "
        "Russell&nbsp;1000 on any given date, and it keeps pricing dead companies right through "
        "their delisting. We built two pieces of tooling against it:", BODY)]
S += bullets([
    "<b>norgate_ingest.py</b> &mdash; pulls every company that was in the Russell&nbsp;1000 at "
    "any point in the window, along with its total-return prices and its day-by-day index "
    "membership flag.",
    "<b>norgate_devalidate.py</b> &mdash; re-runs The Edge on that universe. At each rebalance "
    "it can only pick from companies that were <i>actually in the index that day</i>, and if a "
    "holding delists mid-period it is priced through the failure rather than quietly dropped.",
])
S += [Spacer(1, 0.06 * inch),
      P("What the data gave us:", H2),
      table([[P("Measure", CELLB), P("Value", CELLB), P("Note", CELLB)],
             [P("Companies pulled", CELL), P("1,135", CELL), P("Russell 1000 current <i>and</i> past", CELL)],
             [P("Of which delisted", CELL), P("50", CELL), P("the names a biased universe hides", CELL)],
             [P("Window", CELL), P("17 Jul 2024 &ndash; 16 Jul 2026", CELL), P("hard cap of the free trial", CELL)],
             [P("Trading days", CELL), P("501", CELL), P("~2 years", CELL)],
             [P("Rebalances per sleeve", CELL), P("9", CELL), P("2-month holding clock", CELL)]],
            [1.7 * inch, 1.5 * inch, 3.1 * inch]),
      Spacer(1, 0.05 * inch),
      P("That last row is the whole problem, and it is worth pausing on. Nine decisions is not "
        "a sample you can conclude anything from. We did not know how badly that would bite "
        "until we measured it.", SMALL)]

# ---------------------------------------------------------------- 3
S += [P("3. &nbsp;The first answer &mdash; and why it was wrong", H1),
      P("The first run looked like unambiguous good news. On the real, de-biased universe The "
        "Edge returned <b>28.5%/yr against the S&amp;P's 16.3%</b>, with a shallower drawdown "
        "than our normal backtest. Read at face value: removing the bias made the model look "
        "<i>better</i>, so survivorship was never flattering us.", BODY),
      P("A code review of that script found three separate errors, which compounded in the "
        "same flattering direction:", BODY),
      Spacer(1, 0.04 * inch),
      table([[P("Error", CELLB), P("What it did", CELLB), P("Effect", CELLB)],
             [P("<b>Look-ahead</b>", CELL),
              P("The model read a day's closing price to pick stocks, then bought at that "
                "<i>same</i> closing price. In reality you see the close and can only trade "
                "the next day.", CELL),
              P("Free profit on every trade", CELL)],
             [P("<b>Lucky start date</b>", CELL),
              P("The test started on one arbitrary day. We never checked whether a different "
                "start date gave a different answer.", CELL),
              P("Cherry-picked a good draw", CELL)],
             [P("<b>Drawdown mismeasured</b>", CELL),
              P("Losses were sampled once every 42 days, then compared against our normal "
                "backtest's daily measurement. Crashes that happened and recovered between "
                "samples were invisible.", CELL),
              P("Hid the real risk", CELL)]],
            [1.15 * inch, 4.0 * inch, 1.15 * inch]),
      Spacer(1, 0.08 * inch),
      box("<b>Why this matters more than the result.</b> None of these were exotic. Each one "
          "pushed the answer in the direction we were hoping for, which is exactly why none of "
          "them looked suspicious at the time. The number was only caught because the code was "
          "reviewed <i>after</i> it produced a pleasing answer &mdash; not before.", HexColor("#fdf6e3"))]

# ---------------------------------------------------------------- 4
S += [P("4. &nbsp;The corrected result", H1),
      P("We rebuilt the test to match how the product actually trades: buy the day <i>after</i> "
        "the signal, run two staggered sleeves, check the 200-day market-regime rule every day, "
        "and measure losses daily.", BODY),
      table([[P("", CELLB), P("Return/yr", CELLB), P("Sharpe", CELLB), P("Worst drop", CELLB)],
             [P("<b>The Edge</b> (de-biased, corrected)", CELL), P("<b>3.3%</b>", CELL),
              P("0.14", CELL), P("&minus;36.2%", CELL)],
             [P("S&amp;P 500 (total return)", CELL), P("18.9%", CELL), P("0.86", CELL), P("&minus;18.7%", CELL)],
             [P("<i>Original (retracted)</i>", CELL), P("<i>28.5%</i>", CELL),
              P("<i>0.95</i>", CELL), P("<i>&minus;11%</i>", CELL)]],
            [3.0 * inch, 1.3 * inch, 1.0 * inch, 1.3 * inch], shade_rows=[3]),
      Spacer(1, 0.08 * inch),
      P("Taken alone, that top row reads like a disaster. It isn't &mdash; and this is the "
        "genuinely important part of the report. That 3.3% is <b>also</b> just one draw. When we "
        "re-ran the same test starting on 14 different days of the month, we got this:", BODY),
      Spacer(1, 0.04 * inch),
      table([[P("Start offset (days)", CELLB), P("0", CELLB), P("3", CELLB), P("6", CELLB),
              P("9", CELLB), P("12", CELLB), P("15", CELLB), P("18", CELLB)],
             [P("Return/yr", CELL), P("7.6%", CELL), P("33.1%", CELL), P("6.7%", CELL),
              P("6.0%", CELL), P("5.3%", CELL), P("24.6%", CELL), P("&minus;2.4%", CELL)],
             [P("Start offset (days)", CELLB), P("21", CELLB), P("24", CELLB), P("27", CELLB),
              P("30", CELLB), P("33", CELLB), P("36", CELLB), P("39", CELLB)],
             [P("Return/yr", CELL), P("1.8%", CELL), P("13.6%", CELL), P("44.0%", CELL),
              P("51.6%", CELL), P("54.8%", CELL), P("21.0%", CELL), P("38.0%", CELL)]],
            [1.25 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch,
             0.8 * inch, 0.8 * inch], header=False),
      Spacer(1, 0.08 * inch),
      P("<b>Same strategy. Same data. Same two years. Returns from &minus;2.4% to +54.8%, "
        "purely from which day you happen to start.</b> The median is 17.3%, sitting right on "
        "top of the S&amp;P's 18.9%; exactly half the start dates (7 of 14) beat the market.", BODY),
      box("<b>The honest conclusion.</b> Over a two-year window with nine decisions, this test "
          "is a coin flip. The noise is many times larger than the survivorship effect we were "
          "trying to detect. It cannot confirm the bias is harmless, and it cannot prove it is "
          "fatal. <b>Survivorship stays an open question.</b> Anyone quoting either the 28.5% "
          "or the 3.3% as a finding is quoting noise.")]

# ---------------------------------------------------------------- 5
S += [P("5. &nbsp;A second correction: the rigor claims on the site", H1),
      P("While validating the above, we re-ran the overfitting statistics published on the "
        "<b>/model</b> page and found them stale and flattering. They have been corrected.", BODY),
      table([[P("Statistic", CELLB), P("Page claimed", CELLB), P("Actual", CELLB), P("Status", CELLB)],
             [P("Configurations screened", CELL), P("~150", CELL), P("120", CELL), P("corrected", CELL)],
             [P("Significance (Newey-West t)", CELL), P("~2.8", CELL), P("+2.17", CELL),
              P("still significant", CELL)],
             [P("Deflated Sharpe", CELL), P("&gt;0.95", CELL), P("~1.00", CELL), P("was correct", CELL)],
             [P("Overfitting probability (PBO)", CELL), P("~0.27", CELL), P("~0.5, unstable", CELL),
              P("<b>materially wrong</b>", CELL)]],
            [2.0 * inch, 1.35 * inch, 1.5 * inch, 1.45 * inch], shade_rows=[4]),
      Spacer(1, 0.08 * inch),
      P("<b>What PBO is, in plain language.</b> It answers: <i>when I picked my best settings, "
        "was I finding real signal or just picking noise?</i> The test splits the history in "
        "half every possible way, finds the settings that looked best on one half, then checks "
        "how those same settings actually ranked on the other half. If the winner routinely "
        "turns out mediocre later, you were fitting noise.", BODY)]
S += bullets([
    "<b>Near 0</b> &mdash; the winner keeps winning on fresh data. Healthy.",
    "<b>Near 0.5</b> &mdash; picking the winner is a coin flip. You are selecting noise.",
    "<b>Above 0.5</b> &mdash; the winner is reliably <i>worse</i> later. Actively misleading.",
])
S += [P("The page claimed 0.27 (comfortably healthy). The real figure is about 0.5 &mdash; and "
        "it is <b>not stable</b>: the identical test measured 0.67 on one data vintage and 0.49 "
        "a week later. Publishing the 0.49 alone would have been picking the flattering sample, "
        "which is precisely the bias PBO exists to detect. The page now reports it as "
        "inconclusive and shows both figures.", BODY),
      box("<b>Important nuance.</b> A PBO near 0.5 does <i>not</i> mean the acceleration signal "
          "is fake. It means the <i>dials</i> &mdash; basket size and growth mix &mdash; cannot "
          "be reliably optimised, because thirty near-identical settings all perform about the "
          "same, so there is no persistent winner to find. The real defence against dial-mining "
          "is that we never picked the winner: the shipped configuration ranks <b>59th of "
          "120</b>, chosen on economic reasoning, not on backtest rank.")]

# ---------------------------------------------------------------- 6
S += [P("6. &nbsp;Where this leaves the product", H1)]
S += bullets([
    "<b>Nothing about the live model changed.</b> No signal, weight, or rule was altered by "
    "this work. What changed is what we are entitled to claim about it.",
    "<b>Survivorship is unresolved</b>, not cleared. The /model page has been reverted to say "
    "so, and the earlier &ldquo;the edge survives&rdquo; language has been removed.",
    "<b>The tooling is built and works.</b> Point it at a full-history dataset and it answers "
    "the question properly &mdash; the blocker is data, not code.",
    "<b>Two published statistics were corrected</b> in the direction of being less flattering.",
    "<b>The strongest honest claim remains unchanged</b> from earlier work: sell the "
    "risk-adjusted shape and drawdown control, not secret alpha. Roughly 70% of the return is "
    "buyable factor premium.",
])
S += [P("7. &nbsp;What it would take to actually settle it", H1),
      P("The trial caps history at two years, which is the single binding constraint. A paid "
        "full-history point-in-time licence (Norgate Platinum) would give roughly twenty years "
        "&mdash; on the order of 120 rebalances per sleeve instead of 9. At that length the "
        "start-date noise averages out and the survivorship effect becomes measurable. The same "
        "two scripts run unchanged against it.", BODY),
      P("One further limitation to carry forward: the Norgate stock package contains no "
        "fundamentals, so this test necessarily ran the model in pure-momentum mode with the "
        "revenue-growth blend switched off. Even with full history, validating the shipped "
        "75% growth mix needs a fundamentals source joined to the same point-in-time universe.", BODY),
      P("8. &nbsp;Reproducibility", H1),
      P("Everything in this report regenerates from the repository:", BODY),
      table([[P("File", CELLB), P("Purpose", CELLB)],
             [P("norgate_ingest.py", CELL), P("Pulls the point-in-time universe; caches to data/cache/", CELL)],
             [P("norgate_devalidate.py", CELL), P("Runs the de-biased test + the start-date sweep", CELL)],
             [P("edge_rigor_honest.py", CELL), P("Regenerates Newey-West t, Deflated Sharpe, PBO", CELL)],
             [P("commit 19eb37c", CELL), P("The retraction and corrected basis", CELL)]],
            [1.9 * inch, 5.0 * inch]),
      Spacer(1, 0.1 * inch),
      P("Figures quoted: de-biased book 3.3%/yr, Sharpe 0.14, max drawdown &minus;36.2% over 374 "
        "trading days (9 rebalances per sleeve, two sleeves); S&amp;P 500 total return 18.9%, "
        "Sharpe 0.86, &minus;18.7% over the same span; start-offset sweep across 14 phases, "
        "median 17.3%, range &minus;2.4% to +54.8%. Universe 1,135 names including 50 delisted, "
        "17 Jul 2024 to 16 Jul 2026.", SMALL)]


def footer(canvas, d):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(0.85 * inch, 0.5 * inch,
                      "The Edge - Survivorship Validation  |  Internal research note  |  22 Jul 2026")
    canvas.drawRightString(7.65 * inch, 0.5 * inch, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


doc = SimpleDocTemplate("Edge_Survivorship_Validation.pdf", pagesize=letter,
                        leftMargin=0.85 * inch, rightMargin=0.85 * inch,
                        topMargin=0.7 * inch, bottomMargin=0.75 * inch,
                        title="The Edge - Survivorship Validation",
                        author="Quant Model Research")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote Edge_Survivorship_Validation.pdf")
