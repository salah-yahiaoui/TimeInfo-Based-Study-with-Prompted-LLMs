from __future__ import annotations

import json

from timetank_annotation.schemas import TemporalAnnotation

TIMEINFO_LABEL_GUIDE = """You are working with a TimeInfo-inspired operational annotation scheme.

Your task is to identify temporal expressions in one sentence and assign exactly one interval label to each extracted span.

Allowed labels:
- closed
- closed_duration
- left_open
- right_open

Core semantic principle:
Classify by interval meaning, not by cue word alone.

Operational meaning of each label:
- closed:
  a bounded temporal mention such as a single date, a calendar point, or a localized bounded period.
  Use this when the expression does NOT function as an explicit start-to-end span.
- closed_duration:
  an explicitly bounded temporal span with both a start boundary and an end boundary expressed in the text.
- left_open:
  a temporal expression with a known end boundary but an unknown, unspecified, or linguistically absent start boundary.
- right_open:
  a temporal expression with a known start boundary but an unknown, unspecified, or linguistically absent end boundary.

Decision procedure:
Apply these rules in order.

1. If the expression explicitly gives both a start and an end boundary, label it as closed_duration.
   Typical semantic patterns:
   - from X to Y
   - between X and Y
   - X-Y date span

2. If the expression gives only an end boundary, label it as left_open.
   Typical meanings:
   - until a known point
   - up to a known point
   - by a known point
   - as of a known point

3. If the expression gives only a start boundary, label it as right_open.
   Typical meanings:
   - since a known point
   - starting from a known point
   - after a known point

4. Otherwise, if the expression refers to a bounded temporal point or localized bounded period, label it as closed.

Important semantic and extraction rules:
- The label must reflect the temporal boundaries implied by the expression.
- Use lexical cues such as "on", "between", "from", "since", "as of", "by", "up to", "starting", and "after" only as clues.
- Do not classify by cue word alone if the semantic boundaries indicate a different label.
- Keep the cue word in the extracted span when removing it would change the interval meaning.
  For example:
  - keep "Since December, 2019"
  - keep "as of March 29th 2020"
  - keep "by January 22, 2020"
- Include modifiers such as "early", "mid", and "late" when they are part of the temporal meaning.
- Do not normalize dates.
- Do not paraphrase.
- Extract the exact span only.
- Do not include surrounding non-temporal content unless it is part of the temporal expression.
- Return only the four allowed labels.
- If no valid temporal expression matches the schema, return an empty list.

Scope note:
- This experiment operationalizes only the interval label component inspired by TimeInfo.
- It does not ask you to produce a full TimeInfo structure such as granularity, precision, tempClue, valType, startDuration, or endDuration.
- However, your judgment must still be semantically faithful to the interval interpretation intended by TimeInfo.
"""


TIMEINFO_FEW_SHOT_EXAMPLES = """Examples:
<examples>

<example>
<input>"The mortality of the 27 included patients infected by 2019-nCoV was 37%, which is much higher than that reported 2% on 4 Feb 2020 [3] ."</input>
<output>
[
  {
    "text": "on 4 Feb 2020",
    "label": "closed"
  }
]
</output>
</example>

<example>
<input>"Finally, the estimated number of total infected cases on Jan. 20th in five regions are all significantly larger than one, suggesting the COVID-19 has already spread out nationwide at that moment."</input>
<output>
[
  {
    "text": "on Jan. 20th",
    "label": "closed"
  }
]
</output>
</example>

<example>
<input>"The reproduction number R D 0 at three infectious durations: D = 7, 10.5, 14, for the 30 mainland provinces and 15 cities in Hubei province on February 10th."</input>
<output>
[
  {
    "text": "on February 10th",
    "label": "closed"
  }
]
</output>
</example>

<example>
<input>"In this study, we analyzed the clinical features in 47 patients with COVID-19 who were admitted to Renmin Hospital of Wuhan University between February 1 and February 1, 2020."</input>
<output>
[
  {
    "text": "between February 1 and February 1, 2020",
    "label": "closed_duration"
  }
]
</output>
</example>

<example>
<input>"From January 5 to February 7, 2020, 123 COVID-19 patients were enrolled in the study"</input>
<output>
[
  {
    "text": "From January 5 to February 7, 2020",
    "label": "closed_duration"
  }
]
</output>
</example>

<example>
<input>"From late December, 2019 to early January, 2020, before large scale isolation measures were implemented, many departments in the hospital experienced cross-infection among patients and to the medical staff, due to unrecognized infectious patients and lack of proper protection."</input>
<output>
[
  {
    "text": "From late December, 2019 to early January, 2020",
    "label": "closed_duration"
  }
]
</output>
</example>

<example>
<input>"Since December, 2019, an outbreak of pneumonia caused by a novel coronavirus, severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), has led to a serious epidemic in China and other countries, resulting in worldwide concern."</input>
<output>
[
  {
    "text": "Since December, 2019",
    "label": "right_open"
  }
]
</output>
</example>

<example>
<input>"A total of 791 age-, sex-, and ethnicity-matched adults were recruited as control subjects after the 2003 SARS epidemic, and none of these control subjects ever developed SARS ( Table 1) ."</input>
<output>
[
  {
    "text": "after the 2003",
    "label": "right_open"
  }
]
</output>
</example>

<example>
<input>"Starting March 16, 2020 , France closed all schools nationwide."</input>
<output>
[
  {
    "text": "Starting March 16, 2020",
    "label": "right_open"
  }
]
</output>
</example>

<example>
<input>"According to the public data provided by the Italian Civil Protection Agency 38 , we have estimated a 39% prevalence of COVID-19, as of March 29th 2020, among suspected patients (ratio between affected patients on number of RT-PCR) in the Lombardy region."</input>
<output>
[
  {
    "text": "as of March 29th 2020",
    "label": "left_open"
  }
]
</output>
</example>

<example>
<input>"The clinical outcomes were monitored up to March 13, 2020, the final date of follow-up."</input>
<output>
[
  {
    "text": "up to March 13, 2020",
    "label": "left_open"
  }
]
</output>
</example>

<example>
<input>"Considering the timing of exported COVID-19 cases reported outside of China, we estimate that only 8.95% (95% CrI 2.22% -28.72%) of cases infected in Wuhan by January 12 might have been confirmed by January 22, 2020."</input>
<output>
[
  {
    "text": "by January 22, 2020",
    "label": "left_open"
  }
]
</output>
</example>

</examples>"""


ANNOTATOR_SYSTEM_PROMPT = f"""{TIMEINFO_LABEL_GUIDE}

You are the annotation agent in a two-agent review loop.

Your job:
- read one sentence
- extract all valid temporal expressions that fit the current operational scheme
- assign exactly one allowed label to each extracted span
- return only the final structured answer

You must be exhaustive with respect to the sentence.
That means:
- do not miss a valid temporal expression that should be annotated under this scheme
- do not add spans that are not justified by the sentence
- do not split or truncate a span if the full expression is required for its interval meaning

Revision policy:
- On later attempts, you will receive reviewer feedback and possibly reviewer-suggested corrections.
- Re-evaluate the full sentence from scratch before answering.
- Use the feedback to fix the output, but do not assume the previous attempt was otherwise correct.
- If the reviewer indicates a missing span, wrong label, truncated span, extra span, duplicate span, or invalid span, correct the entire annotation list accordingly.

Output policy:
- Return JSON only.
- Return a JSON list.
- Each item must have:
  - "text": exact temporal span from the sentence
  - "label": one of the four allowed labels
- If no valid temporal expression exists, return [].

Reason silently before answering.
Do not reveal your reasoning.

{TIMEINFO_FEW_SHOT_EXAMPLES}
"""


VERIFIER_SYSTEM_PROMPT = f"""{TIMEINFO_LABEL_GUIDE}

You are the verification agent in a two-agent review loop.

Your job:
- inspect the sentence
- inspect the candidate annotation list produced by the annotation agent
- judge whether the candidate list is fully correct for this operational scheme

You must verify all of the following:
- every proposed span is an exact span from the sentence
- no proposed span is truncated in a way that changes interval meaning
- no proposed span includes unnecessary surrounding text
- every label is semantically correct
- no valid annotation is missing
- no extra invalid annotation is present
- duplicated annotations are rejected

Decision policy:
- Use "accept" only if the entire candidate annotation list is correct and complete.
- Use "revise" if there is any missing annotation, wrong label, invalid span, duplicate, extra annotation, truncation problem, or other semantic issue.

Feedback policy:
- Be explicit.
- Describe exactly what must be corrected.
- If revising, provide a full corrected annotation list in `suggested_annotations`, not a partial patch.
- If accepting, set `suggested_annotations` equal to the accepted annotation list.

Output policy:
- Return JSON only.
- Follow the provided schema exactly.
- Keep feedback concise but specific.

Reason silently before answering.
Do not reveal your reasoning.

{TIMEINFO_FEW_SHOT_EXAMPLES}
"""


def render_annotation_request(
    *,
    sentence: str,
    review_round: int,
    max_review_rounds: int,
    previous_annotations: list[TemporalAnnotation] | None = None,
    reviewer_feedback: str | None = None,
    reviewer_suggested_annotations: list[TemporalAnnotation] | None = None,
    controller_feedback: list[str] | None = None,
) -> str:
    payload: list[str] = [
        f"Annotation round {review_round} of {max_review_rounds}.",
        "Annotate the following sentence under the operational TimeInfo interval scheme.",
        "",
        "<input>",
        sentence,
        "</input>",
    ]

    if review_round > 1:
        payload.extend(
            [
                "",
                "Previous candidate annotations:",
                _annotations_json(previous_annotations or []),
                "",
                "Reviewer feedback:",
                reviewer_feedback or "No reviewer feedback provided.",
            ]
        )

        if reviewer_suggested_annotations is not None:
            payload.extend(
                [
                    "",
                    "Reviewer suggested full corrected annotation list:",
                    _annotations_json(reviewer_suggested_annotations),
                ]
            )

        if controller_feedback:
            payload.extend(
                [
                    "",
                    "Controller technical feedback:",
                    json.dumps(controller_feedback, ensure_ascii=False, indent=2),
                ]
            )

        payload.extend(
            [
                "",
                "Revise the annotation list accordingly and return the full corrected JSON list.",
            ]
        )
    else:
        payload.extend(
            [
                "",
                "Return the full annotation list now.",
            ]
        )

    return "\n".join(payload)


def render_verification_request(
    *,
    sentence: str,
    candidate_annotations: list[TemporalAnnotation],
    review_round: int,
    max_review_rounds: int,
) -> str:
    return "\n".join(
        [
            f"Verification round {review_round} of {max_review_rounds}.",
            "Review the candidate annotation list for correctness and completeness.",
            "",
            "<input>",
            sentence,
            "</input>",
            "",
            "Candidate annotations:",
            _annotations_json(candidate_annotations),
            "",
            "Return your structured verification decision now.",
        ]
    )


def _annotations_json(annotations: list[TemporalAnnotation]) -> str:
    return json.dumps([annotation.model_dump() for annotation in annotations], ensure_ascii=False, indent=2)
