from flask import Flask, render_template, request, redirect, url_for, flash
import random, itertools
from collections import Counter

app = Flask(__name__)
app.secret_key = "change-me-in-prod"

# ---------- Card helpers ----------
SUITS = ['♠', '♥', '♦', '♣']
RANK_CHARS = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}

def rank(card: int) -> int:
    return (card % 13) + 2

def suit(card: int) -> int:
    return card // 13

def card_to_str(card: int) -> str:
    r = rank(card)
    rch = RANK_CHARS.get(r, 'T' if r == 10 else str(r if r <= 10 else r))
    if r == 10:
        rch = 'T'
    return f"{rch}{SUITS[suit(card)]}"

def cards_to_str(cards):
    return ' '.join(card_to_str(c) for c in cards)

def make_deck():
    return list(range(52))

# ---------- 5-card evaluator ----------
def is_straight(ranks_desc):
    uniq = sorted(set(ranks_desc), reverse=True)
    if len(uniq) < 5:
        return False, 0
    if uniq[0] - uniq[4] == 4 and len(uniq) == 5:
        return True, uniq[0]
    if 14 in uniq:
        wheel = {14, 5, 4, 3, 2}
        if set(uniq[:5]) == wheel or set(uniq) >= wheel:
            return True, 5
    return False, 0

def hand_rank_5(cards5):
    rs = [rank(c) for c in cards5]
    ss = [suit(c) for c in cards5]
    rs_sorted = sorted(rs, reverse=True)
    flush = len(set(ss)) == 1
    is_str, straight_high = is_straight(rs_sorted)
    counts = Counter(rs)
    counts_sorted = sorted(counts.values(), reverse=True)

    if is_str and flush:
        return 8, (straight_high,)
    if counts_sorted[0] == 4:
        quad = max(r for r, c in counts.items() if c == 4)
        kicker = max(r for r in rs if r != quad)
        return 7, (quad, kicker)
    if counts_sorted[0] == 3 and counts_sorted[1] == 2:
        trips = max(r for r, c in counts.items() if c == 3)
        pair = max(r for r, c in counts.items() if c == 2)
        return 6, (trips, pair)
    if flush:
        return 5, tuple(sorted(rs, reverse=True))
    if is_str:
        return 4, (straight_high,)
    if counts_sorted[0] == 3:
        trips = max(r for r, c in counts.items() if c == 3)
        kickers = sorted((r for r in rs if r != trips), reverse=True)
        return 3, (trips, *kickers)
    if counts_sorted[0] == 2 and counts_sorted[1] == 2:
        pairs = sorted((r for r, c in counts.items() if c == 2), reverse=True)
        kicker = max(r for r, c in counts.items() if c == 1)
        return 2, (*pairs, kicker)
    if counts_sorted[0] == 2:
        pair = max(r for r, c in counts.items() if c == 2)
        kickers = sorted((r for r in rs if r != pair), reverse=True)
        return 1, (pair, *kickers)
    return 0, tuple(sorted(rs, reverse=True))

def best_five_of_seven(cards7):
    best = None
    for combo in itertools.combinations(cards7, 5):
        hr = hand_rank_5(combo)
        if best is None or hr > best:
            best = hr
    return best

# ---------- Preflop heuristic ----------
BASE_VALUES = {14:10.0,13:8.0,12:7.0,11:6.0,10:5.0,9:4.5,8:4.0,7:3.5,6:3.0,5:2.5,4:2.0,3:1.5,2:1.0}

def card_rank_char(r:int)->str:
    return {14:'A',13:'K',12:'Q',11:'J',10:'T'}.get(r,str(r))

def preflop_score(hole):
    r1, r2 = sorted([rank(hole[0]), rank(hole[1])], reverse=True)
    s1, s2 = suit(hole[0]), suit(hole[1])
    suited = s1 == s2
    base = BASE_VALUES[r1]
    if r1 == r2:
        score = max(5.0, 2 * BASE_VALUES[r1])
    else:
        score = base
    if suited and r1 != r2:
        score += 2.0
    gap = (r1 - r2) - 1
    if r1 != r2:
        score += {0:1.0,1:0.0,2:-1.0,3:-2.0}.get(gap, -4.0)
    if r1 <= 12 and r2 <= 12 and gap >= 2 and r1 != r2:
        score -= 1.0
    score = round(score * 2)/2.0
    if score < 6: rec = "Fold"
    elif score < 8: rec = "Marginal"
    elif score < 11: rec = "Play"
    else: rec = "Strong"
    reason = (f"{card_rank_char(r1)}{card_rank_char(r2)} pair" if r1==r2
              else ("suited " if suited else "") + (["connected","1-gap","2-gap","3-gap"][gap] if 0<=gap<=3 else ">=4-gap"))
    return score, rec, reason

# ---------- Equity (Monte Carlo) ----------
def estimate_equity(hole, board, n_opponents, sims, rng):
    if n_opponents <= 0:
        return 100.0, 0.0
    known = set(hole) | set(board)
    deck = [c for c in make_deck() if c not in known]
    wins = ties = 0
    need = 5 - len(board)

    for _ in range(sims):
        rng.shuffle(deck)
        opp_holes = [deck[2*j:2*j+2] for j in range(n_opponents)]
        idx = 2 * n_opponents
        future_board = deck[idx: idx + need]
        final_board = list(board) + future_board
        hero_rank = best_five_of_seven(list(hole) + final_board)
        opp_ranks = [best_five_of_seven(o + final_board) for o in opp_holes]
        better = any(r > hero_rank for r in opp_ranks)
        equal = any(r == hero_rank for r in opp_ranks)
        if not better and not equal:
            wins += 1
        elif not better and equal:
            ties += 1
    total = float(sims)
    return wins*100/total, ties*100/total

def detect_draws(hole, board):
    all_cards = list(hole) + list(board)
    if len(board) < 5:
        suit_counts = Counter(suit(c) for c in all_cards)
        if any(cnt == 4 for cnt in suit_counts.values()):
            return "You have a flush draw (9 outs)"
        rs = sorted(set(rank(c) for c in all_cards))
        if 14 in rs: rs.append(1)
        for i in range(len(rs)-3):
            w = rs[i:i+4]
            if len(w)==4 and w[0]-w[3]==3:
                return "You have an open-ended straight draw (8 outs)"
        for i in range(len(rs)-3):
            w = rs[i:i+4]
            if w[0]-w[3]==4:
                return "You have a gutshot straight draw (4 outs)"
    return None

# ---------- Serialization helpers (no session needed) ----------
def ser(lst):      # list[int] -> "int,int,int"
    return ",".join(str(x) for x in lst)

def deser(s):      # "int,int" -> list[int]
    if not s:
        return []
    return [int(x) for x in s.split(",") if x != ""]

# ---------- Web routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            n_opp = int(request.form.get("opponents", "2"))
            sims = int(request.form.get("sims", "3000"))
            n_opp = min(max(n_opp, 1), 8)
            sims = min(max(sims, 1000), 20000)
        except ValueError:
            flash("Please enter valid numbers.")
            return redirect(url_for("index"))

        rng = random.Random()
        deck = make_deck()
        rng.shuffle(deck)
        hole = [deck.pop(), deck.pop()]
        board = []

        score, rec, reason = preflop_score(hole)

        return render_template(
            "hand.html",
            stage="preflop",
            hole_str=cards_to_str(hole),
            board_str="",
            preflop={"score": f"{score:.1f}", "rec": rec, "reason": reason},
            equity=None,
            hint=None,
            # hidden form state
            deck_ser=ser(deck),
            hole_ser=ser(hole),
            board_ser=ser(board),
            n_opp=n_opp,
            sims=sims
        )

    return render_template("index.html")

@app.route("/progress", methods=["POST"])
def progress():
    # pull state from hidden fields
    deck = deser(request.form.get("deck_ser", ""))
    hole = deser(request.form.get("hole_ser", ""))
    board = deser(request.form.get("board_ser", ""))
    try:
        n_opp = int(request.form.get("n_opp", "2"))
        sims = int(request.form.get("sims", "3000"))
    except ValueError:
        return redirect(url_for("index"))

    action = request.form.get("action", "continue")
    if action == "fold":
        return render_template("result.html", message="You folded. Hand ended.")

    # Advance a street
    if len(board) == 0:
        # flop
        if len(deck) < 3:
            return redirect(url_for("index"))
        board.extend([deck.pop(), deck.pop(), deck.pop()])
        stage = "flop"
    elif len(board) == 3:
        if len(deck) < 1:
            return redirect(url_for("index"))
        board.append(deck.pop())  # turn
        stage = "turn"
    elif len(board) == 4:
        if len(deck) < 1:
            return redirect(url_for("index"))
        board.append(deck.pop())  # river
        stage = "river"
    else:
        # showdown
        rng = random.Random()
        win, tie = estimate_equity(hole, board, n_opp, sims, rng)
        return render_template(
            "result.html",
            message=f"Final board: {cards_to_str(board)}. Equity vs {n_opp}: {win+tie:.1f}% (win {win:.1f}%, tie {tie:.1f}%)."
        )

    # Compute street equity
    rng = random.Random()
    win, tie = estimate_equity(hole, board, n_opp, sims, rng)
    hint = detect_draws(hole, board)

    return render_template(
        "hand.html",
        stage=stage,
        hole_str=cards_to_str(hole),
        board_str=cards_to_str(board),
        preflop=None,
        equity={"line": f"Equity vs {n_opp}: {win+tie:.1f}% (win {win:.1f}%, tie {tie:.1f}%)"},
        hint=hint,
        # updated hidden state for next click
        deck_ser=ser(deck),
        hole_ser=ser(hole),
        board_ser=ser(board),
        n_opp=n_opp,
        sims=sims
    )

@app.route("/reset")
def reset():
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
