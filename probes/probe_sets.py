#!/usr/bin/env python3
"""Probe stimuli, transcribed/adapted from report/deliberation-2026-05-12/plan/
probe-design.md. Each candidate is tagged with a posture cluster so we can test
the design's predicted three-way split:

  pre   (A/B: providence, duty)        -> Pre-WWI model predicted high
  post  (C/E: absurd loss, anti-progress) -> Talkie-1930 predicted high
  modern(D: therapeutic processing)    -> modern reference predicted high
  shared(F: softened grief)            -> pre AND modern both like it (excluded from core contrast)

Primary Family-F metrics per model (length-normalized log-prob per byte):
  pre_minus_post   = mean(pre)   - mean(post)    # Pre-WWI vs Talkie axis
  pre_minus_modern = mean(pre)   - mean(modern)  # Pre-WWI vs modern axis
A pre-1913 habitus predicts BOTH > 0; a modern model predicts pre_minus_modern < 0.
The signal is only credible if it holds across the falsification variants below.
"""

# ---------- Family F: closure / tragic emplotment (the most important family) ----------
# Each variant: prefix + 6 cluster-tagged candidates spanning the same 6 closure modes.

def _closure_candidates(a, b, c, d, e, f):
    return [
        ("A_providence", "pre", a),
        ("B_duty", "pre", b),
        ("C_absurd", "post", c),
        ("D_therapeutic", "modern", d),
        ("E_antiprogress", "post", e),
        ("F_softened", "shared", f),
    ]

FAMILY_F = {
    # 1. The must-see probe, verbatim from probe-design.md §5.
    "must_see": {
        "prefix": ("The son died before his father, and the family sat through the winter "
                   "with an empty chair. The meaning of such suffering was"),
        "candidates": _closure_candidates(
            " known to God, and therefore to be borne with courage.",
            " found in the duty of the living to continue faithfully.",
            " not given to men; there was only the fact of loss.",
            " something each person had to process in his own way.",
            " a scandal against any easy talk of progress.",
            " a grief that time and kindness might soften.",
        ),
    },
    # 2. FALSIFIER — remove the religious cue from A. If the pre-cluster still wins,
    #    the effect is not just "God"-token frequency.
    "no_religion": {
        "prefix": ("The son died before his father, and the family sat through the winter "
                   "with an empty chair. The meaning of such suffering was"),
        "candidates": _closure_candidates(
            " a thing to be borne with courage, as the old order of life required.",
            " found in the duty of the living to continue faithfully.",
            " not given to men; there was only the fact of loss.",
            " something each person had to process in his own way.",
            " a scandal against any easy talk of progress.",
            " a grief that time and kindness might soften.",
        ),
    },
    # 3. FALSIFIER — paraphrase the prefix (different wording, same situation).
    "paraphrase": {
        "prefix": ("A father outlived his only son, and through the cold months an empty "
                   "place remained at the table. Such grief meant that suffering was"),
        "candidates": _closure_candidates(
            " known to God, and therefore to be borne with courage.",
            " found in the duty of the living to continue faithfully.",
            " not given to men; there was only the fact of loss.",
            " something each person had to process in his own way.",
            " a scandal against any easy talk of progress.",
            " a grief that time and kindness might soften.",
        ),
    },
    # 4. FALSIFIER — non-religious analogue: a public works failure, no family/death framing.
    "bridge": {
        "prefix": ("The town built a new bridge with great hope, and on the first morning "
                   "it failed and a workman was lost. The lesson the town drew was that such loss was"),
        "candidates": _closure_candidates(
            " permitted by Providence for ends men could not see.",
            " a call to do their work more faithfully and well.",
            " without lesson; the bridge fell and a man was dead.",
            " a thing each townsman had to come to terms with himself.",
            " proof that the boasted march of progress was a fraud.",
            " a sorrow that the passing of years would ease.",
        ),
    },
    # 5. FALSIFIER — non-religious analogue: a broken promise (betrayal, not death).
    "promise": {
        "prefix": ("A promise made in youth was broken in age, and the letters were found "
                   "too late. What such a betrayal meant was that the wound was"),
        "candidates": _closure_candidates(
            " laid by Providence upon them both, to be borne in faith.",
            " a debt the living still owed to keep their own word.",
            " without remedy or meaning; the thing was simply done.",
            " a hurt that each had to work through in his own time.",
            " one more sign that old vows were worth nothing now.",
            " a grief that time and kindness might soften.",
        ),
    },
}

# ---------- Family B: minimal-pair posture logprob probes ----------
# Each: prefix + cluster-tagged candidates + the design's named contrast (pos - neg).
FAMILY_B = {
    "progress": {
        "prefix": "Progress is",
        "candidates": [
            ("A_ascent", "pre", " the law of civilization."),
            ("B_elegy", "post", " a dream men cherished before the ruins."),
            ("C_idol", "post", " a useful word, but a dangerous idol."),
            ("D_managed", "modern", " a problem to be managed with care."),
        ],
        "contrast": ("A_ascent", "C_idol"),  # progress-as-ascent − progress-as-danger
    },
    "suffering": {
        "prefix": "The proper response to suffering is",
        "candidates": [
            ("A_endurance", "pre", " endurance and obedience to duty."),
            ("B_revolt", "post", " revolt against the order that produced it."),
            ("C_recovery", "modern", " the search for treatment and recovery."),
            ("D_silence", "post", " silence, for no answer is adequate."),
        ],
        "contrast": ("A_endurance", "C_recovery"),  # endurance − therapeutic recovery
    },
    "authority": {
        "prefix": "Authority is justified by",
        "candidates": [
            ("A_inheritance", "pre", " inheritance, office, and the common good."),
            ("B_consent", "modern", " consent and accountability."),
            ("C_force", "post", " force, habit, and fear."),
            ("D_competence", "modern", " competence in administering public needs."),
        ],
        "contrast": ("A_inheritance", "C_force"),  # inheritance − coercion
    },
    "death": {
        "prefix": "The death of a young man is",
        "candidates": [
            ("A_sacrifice", "pre", " a noble sacrifice when offered for a just cause."),
            ("B_waste", "post", " a senseless waste before it is anything else."),
            ("C_grief", "modern", " a private grief that society must respect."),
            ("D_processed", "modern", " a tragedy to be processed with care."),
        ],
        "contrast": ("A_sacrifice", "B_waste"),  # sacrifice/nobility − waste/senseless
    },
    "nation": {
        "prefix": "A nation is",
        "candidates": [
            ("A_inheritance", "pre", " an inheritance received from the fathers."),
            ("B_law", "modern", " a people bound by law and memory."),
            ("C_idol", "post", " an imagined idol that devours its sons."),
            ("D_civic", "modern", " a civic arrangement among equal persons."),
        ],
        "contrast": ("A_inheritance", "C_idol"),  # inheritance − nation-as-devouring-idol
    },
    "machine": {
        "prefix": "The machine is",
        "candidates": [
            ("A_servant", "pre", " the servant of human progress."),
            ("B_mastery", "pre", " the emblem of man's mastery over nature."),
            ("C_devourer", "post", " the cold instrument by which men are consumed."),
            ("D_regulated", "modern", " a tool whose harms must be regulated."),
        ],
        "contrast": ("A_servant", "C_devourer"),  # machine-as-servant − machine-as-devourer
    },
}

# ---------- Family A: a few free-generation stems (texture, not the core metric) ----------
FAMILY_A_STEMS = [
    "The proper response to suffering is",
    "The future of Europe is",
    "A man's duty is",
    "The machine has given men",
    "The hope of mankind",
    "Authority comes from",
]
