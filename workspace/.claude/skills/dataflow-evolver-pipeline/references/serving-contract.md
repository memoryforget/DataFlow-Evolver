# API LLM serving contract

Read this reference when a generated pipeline uses an API-backed LLM operator.

Use open-dataflow 1.0.10 `APILLMServing_request` directly. DataFlow owns HTTP
transport, connection pooling, concurrency, retries, timeouts, request
construction, input-order alignment, and response formatting. Generated code
must not import a project-specific serving class or know about runtime transport
adaptation.

## Request fields

| Field | Meaning |
|---|---|
| api_url | Full OpenAI-compatible chat-completions endpoint. |
| key_name_of_api_key | Environment-variable name containing the API key, not the key value. |
| model_name | Request-body model; it does not define response parsing. |
| temperature | Sampling temperature. |
| enable_thinking | Required boolean response contract. `true` exposes reasoning plus content in DataFlow wrappers; `false` exposes content only. The same value is forwarded to the upstream model. |
| extra keyword arguments | The remaining dynamic serving profile forwarded by DataFlow, for example `top_p`, `max_tokens`, `stream`, and `stream_options`. |
| max_workers | DataFlow-owned HTTP concurrency and connection-pool size. |
| max_retries | Total attempts per logical input, including the first. |
| connect_timeout / read_timeout | Connection and response-read timeouts. |

Copy the dynamic serving profile to `APILLMServing_request`. Keep all transport
settings unchanged. When the task requires different response contracts,
construct separate Pipeline-owned serving instances and change only the
explicit `enable_thinking` boolean. Never use strings such as `"true"` or
`"false"`.

## Response contracts

With `enable_thinking=true`, Serving returns:

```text
<think>{reasoning}</think>
<answer>{content}</answer>
```

Use this contract for the category of calls whose downstream artifact must
retain model reasoning. Final Math SFT solution generation is one example, not
the definition of the category. Preserve the entire returned string and
validate it against the task's quality requirements.

With `enable_thinking=false`, Serving returns `message.content` only. Use this
contract for the category of calls whose downstream artifact needs only
content. Judges, scorers, classifiers, routers, extractors, and concise
normalizers are examples, not an exhaustive operator list. Parse and validate
the content according to the operator's declared domain output.

Operators receive one aligned `str` or `None` per input. They must not inspect
raw OpenAI dictionaries, parse streaming events, concatenate reasoning and
content fields, add another wrapper layer, add transport retries, or construct a
second HTTP client. An operator may parse its returned domain content, such as a
score, but must not parse raw transport fields or streaming events.

## Prompt and validation ownership

Prompts request only the task-specific domain result appropriate to the chosen
response category. Reference material may be supplied as a correctness anchor,
but the visible response must not discuss reference matching. Do not mention
response field names, transport serialization, SFT construction, or formatting
deliberation unless the task itself genuinely requires that visible text.

For a reasoning-retaining call, validate the returned formatted text against the
task's substantive requirements and do not strip the `<think>/<answer>`
structure. For a content-only call, validate the returned content against that
operator's domain contract and reject malformed values rather than guessing.
