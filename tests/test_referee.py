"""Test cases for the robust maths referee (paper, Appendix D). Run: python tests/test_referee.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fdrl"))
import checkers as ck

CASES = [  # (end of reply, gold, expected label)
    # format drift on a correct answer: right
    (r"\boxed{\boxed{\mathbf{110}}}", "110", 1),
    (r"### Final Answer: $$\boxed{\boxed{\mathbf{90}}}$$", "90", 1),
    (r"\boxed{60 \text{ square units}}", "60", 1),
    (r"\boxed{y = 28}", "28", 1),
    (r"\boxed{\text{The value is } 28}", "28", 1),
    (r"\boxed{973} \boxed{\text{as an integer}}", "973", 1),
    (r"\boxed{\boxed{\frac{50}{5}} = \boxed{10}}", "10", 1),
    (r"\boxed{45} ... Final Answer: \boxed{45} \boxed{45}", "45", 1),
    (r"\boxed{\textbf{36}}", "36", 1),
    (r"\boxed{}  so the answer is 52", "52", 1),
    (r"Final Answer: **12**", "12", 1),
    # exploits and wrong answers: wrong
    (r"so \boxed{21}. 12", "12", 0),
    (r"\boxed{12} \boxed{13}", "12", 0),
    (r"12 of them have ... so the answer is \boxed{21}", "12", 0),
    (r"\boxed{2\sqrt{3}}", "12", 0),
    (r"Area = 1/2 |32| = 16. Wait, let me re-check the sum: 0 + 8 + 24 + 0 = 3", "16", 0),
    (r"\boxed{120}", "12", 0),
    ("a reply with no number", "12", 0),
]

if __name__ == "__main__":
    bad = [(r, g, e, ck.gold_label(r, g)) for r, g, e in CASES if ck.gold_label(r, g) != e]
    for r, g, e, got in bad:
        print(f"FAIL gold={g} expected={e} got={got}: {r}")
    print(f"{len(CASES) - len(bad)}/{len(CASES)} passed")
    sys.exit(1 if bad else 0)
