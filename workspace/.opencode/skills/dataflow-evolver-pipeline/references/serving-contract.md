# API LLM serving contract

Read this reference when a generated pipeline uses an API-backed LLM operator.

EvolverAPILLMServing is a thin response-contract adapter over open-dataflow
1.0.10 APILLMServing_request. DataFlow owns HTTP transport, connection pooling,
concurrency, retries, timeouts, request construction, and input-order alignment.
The adapter adds explicit finish-reason and required-field checks.

## Request fields

| Field | Meaning |
|---|---|
| api_url | Full OpenAI-compatible chat-completions endpoint. |
| key_name_of_api_key | Environment-variable name containing the API key, not the key value. |
| model_name | Request-body model; it does not define response parsing. |
| temperature | Sampling temperature. |
| extra keyword arguments | Forwarded to the request body, for example top_p, max_tokens, or chat_template_kwargs. |
| max_workers | DataFlow-owned HTTP concurrency and connection-pool size. |
| max_retries | Total attempts per logical input, including the first. |
| connect_timeout / read_timeout | Connection and response-read timeouts. |
| allowed_finish_reasons | Accepted finish reasons; other values become None and are retried. |
| require_content | Reject missing or empty message.content. |
| response_mode | Explicitly selects content, reasoning_and_content, or auto; never infer it from a model name. |

## Formal math profile

The formal math task uses response_mode="content",
allowed_finish_reasons=["stop"], and require_content=true.

In this mode the returned string is trimmed message.content. Raw
message.reasoning and message.reasoning_content are ignored. A reasoning model's
internal planning is not assumed to be clean, pedagogical training text.

Operators receive one aligned str or None per input. They must not parse raw
OpenAI dictionaries, concatenate hidden reasoning, add transport retries, or
construct another HTTP client.

## Prompt and validation ownership

Prompts request only the domain result: a correct, complete mathematical
derivation followed by the final answer. Do not mention hidden reasoning fields,
transport serialization, SFT construction, reference-answer matching, or prompt
format deliberation unless the task itself genuinely requires that visible text.

Validate the returned content for mathematical correctness, completeness,
non-truncation, and a non-empty final answer. Do not require wrapper tokens unless
the task explicitly defines them as part of its output schema.

Record truthful outcomes such as llm_rewritten, fallback_original, and
rejected_response. A fallback may enter the final dataset only when it already
contains a substantive derivation and correct final answer.
