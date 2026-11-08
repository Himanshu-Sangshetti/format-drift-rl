# Hand-labelling protocol

The label is the **committed final answer**: the value the reply last presents as its answer to the
question asked. The labeller decides this from the reply alone, before the reference answer is shown.
Correctness (right / wrong) is computed afterwards by comparing the label with the reference.

## Labels

- a value, e.g. `48`: the reply commits to this answer
- `C` (cut off): the reply hits the token limit before committing to any answer
- `N` (no answer): the reply ends (not cut off) without committing to any answer

## Rules

1. **What counts as a commitment.** A value stated as the result for the quantity the question asks
   for: `\boxed{X}`, "the answer is X", "Final answer: X", "so there are X ways", or a closing
   sentence that gives X as the result. Formatting does not matter (boxed, bold, plain text, another
   language all count).
2. **Last commitment wins.** If the reply commits to X and later commits to a different value Y,
   the label is Y.
3. **Repetition and loops do not cancel a commitment.** If, after committing to X, the reply repeats
   X, re-checks it, restates the problem, or loops, without committing to a different value, the label
   is X, even if the reply is then cut off.
4. **Hedges are not commitments.** "Maybe X", "perhaps X", "X?", "it could be X or Y", or listing
   candidates without choosing one does not commit. If the last statement about the answer is a hedge
   on a different value than an earlier commitment, the earlier commitment is withdrawn: label `C`
   (if cut off) or `N`.
5. **Intermediate values do not count.** A number computed along the way, or the value of a different
   quantity than the one asked, is not a commitment.
6. **A cut through the answer.** If the token limit cuts the reply while the answer is being written
   and the value may be incomplete, label `C`.
7. **Equivalent forms are the same value** (`3/4` and `0.75`; units and thousands separators are
   ignored).
8. **Wrong question.** If the reply answers a different question than the one asked, its committed
   value is still the label (it will usually be wrong against the reference).

## Consistency

- 50 items are re-labelled by the human after at least 4 days, blind to the first label.
