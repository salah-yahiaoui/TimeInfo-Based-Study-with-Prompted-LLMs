from __future__ import annotations

PROMPT_TEMPLATE = """You are an expert temporal named entity extraction assistant.

Your task is to extract temporal expressions from a sentence and assign exactly one label to each extracted span.

Allowed labels:
- closed
- closed_duration
- left_open
- right_open

Core semantic principle:
Classify each temporal expression by its interval meaning, not by keyword alone.

Semantic interpretation of labels:
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

Important semantic rules:
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

Reason silently before answering.
Do not reveal your reasoning.

Examples:
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

</examples>

Now annotate the following sentence.

<input>
{{sentence}}
</input>"""


def render_prompt(sentence: str) -> str:
    return PROMPT_TEMPLATE.replace("{{sentence}}", sentence)
