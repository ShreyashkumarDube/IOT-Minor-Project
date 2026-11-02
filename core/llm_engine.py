"""LLM-based trust and policy evaluator.

Works with Ollama by default and with any OpenAI-compatible chat endpoint.
The LLM is constrained to a JSON schema-like response; the caller validates and
clamps the output before using it for a simulated access decision.
"""
import json, time, urllib.request, urllib.error
from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMConfig:
    provider: str = 'ollama'       # ollama | openai_compatible
    model: str = 'llama3.2:3b'
    endpoint: str = 'http://localhost:11434/api/chat'
    api_key: str = ''
    timeout: int = 45
    temperature: float = 0.0
    # Speed options (Live mode sets them; None = not sent, which keeps the Classic path exactly as before)
    max_tokens: Optional[int] = None     # Ollama options.num_predict / OpenAI max_tokens
    keep_alive: Optional[str] = None     # Ollama keep_alive, e.g. '30m'
    num_ctx: Optional[int] = None        # Ollama options.num_ctx
    think: Optional[bool] = None         # Ollama "think" (False turns a thinking mode off); dropped if the model rejects it
    compact_json: bool = False           # serialise the context without indentation

class LLMUnavailable(RuntimeError):
    pass

SYSTEM_PROMPT = """You are an IoT zero-trust Trust Evaluation (TE) and Policy Engine (PE).
Evaluate only the supplied telemetry. Do not invent facts. Trust is the estimated
probability in [0,1] that the requesting device is legitimate for this request.
Use the supplied prior trust as a starting point. Suspicious behavior, compromised
posture, failed ZK authentication, sensitive targets and unusual topology should
reduce confidence. A valid ZK proof authenticates knowledge of the registered secret,
but does not by itself make a risky device trustworthy.

Return ONLY valid JSON with exactly these keys:
trust_score (number 0..1), action (one of ALLOW, STEP-UP, BLOCK), risk_score (number 0..1),
reason (string), required_validation (string), confidence (number 0..1),
factors (array of short strings), behavior_assessment (array of objects with behavior and risk 0..1).
"""

def _json_from_text(text):
    text = text.strip()
    if text.startswith('```'):
        text = text.strip('`')
        if text.startswith('json'):
            text = text[4:].strip()
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        raise ValueError('LLM did not return JSON')
    return json.loads(text[start:end+1])

class LLMEvaluator:
    def __init__(self, config: LLMConfig, system_prompt: str = None, response_schema: dict = None):
        self.config = config
        self.system_prompt = system_prompt or SYSTEM_PROMPT  # Live mode passes its own facts-only prompt
        self.response_schema = response_schema               # JSON schema for constrained output (Live mode)
        self.think_supported = True    # set to False once the server rejects the "think" field
        self.schema_supported = True   # set to False once the server rejects a schema in format / response_format

    def user_prompt(self, context: dict) -> str:
        """The context as sent: compact JSON in Live mode (fewer tokens), indented JSON in Classic mode."""
        if self.config.compact_json:
            return json.dumps(context, separators=(',', ':'))
        return json.dumps(context, indent=2)

    def _payload(self, user_prompt):
        c = self.config
        messages = [{'role':'system','content':self.system_prompt},{'role':'user','content':user_prompt}]
        if c.provider == 'ollama':
            options = {'temperature': c.temperature}
            if c.max_tokens is not None:
                options['num_predict'] = c.max_tokens
            if c.num_ctx is not None:
                options['num_ctx'] = c.num_ctx
            body = {'model': c.model, 'messages': messages, 'stream': False, 'options': options,
                    'format': self.response_schema if (self.response_schema and self.schema_supported) else 'json'}
            if c.keep_alive is not None:
                body['keep_alive'] = c.keep_alive
            if c.think is not None and self.think_supported:
                body['think'] = c.think
            return body
        body = {'model': c.model, 'messages': messages, 'temperature': c.temperature,
                'response_format': {'type':'json_object'}}
        if self.response_schema and self.schema_supported:
            body['response_format'] = {'type': 'json_schema',
                                       'json_schema': {'name': 'trust_evaluation', 'schema': self.response_schema, 'strict': False}}
        if c.max_tokens is not None:
            body['max_tokens'] = c.max_tokens
        return body

    def _post(self, body: dict) -> dict:
        c = self.config
        req = urllib.request.Request(c.endpoint, data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
        if c.api_key:
            req.add_header('Authorization', f'Bearer {c.api_key}')
        with urllib.request.urlopen(req, timeout=c.timeout) as r:
            return json.loads(r.read().decode())

    def _request(self, user_prompt: str) -> dict:
        """POST the request; if the server rejects an optional feature (think, schema), retry once without it."""
        for _ in range(3):
            body = self._payload(user_prompt)
            try:
                return self._post(body)
            except urllib.error.HTTPError as exc:
                detail = ''
                try:
                    detail = exc.read().decode(errors='replace').lower()
                except Exception:
                    pass
                if exc.code in (400, 422, 500) and 'think' in body and 'think' in detail:
                    self.think_supported = False        # model has no thinking mode: drop the field and retry
                    continue
                schema_sent = isinstance(body.get('format'), dict) or body.get('response_format', {}).get('type') == 'json_schema'
                if exc.code in (400, 422) and schema_sent and self.schema_supported:
                    self.schema_supported = False       # old server: fall back to plain JSON mode and retry
                    continue
                raise LLMUnavailable(f'LLM request failed: HTTP {exc.code} {detail[:160]}') from exc
            except Exception as exc:
                raise LLMUnavailable(f'LLM request failed: {exc}') from exc
        raise LLMUnavailable('LLM request failed after dropping optional features')

    def evaluate(self, context: dict):
        """Classic path: request + lenient validation (fills defaults)."""
        return self.validate(self.evaluate_raw(context))

    def evaluate_raw(self, context: dict):
        """Request and parse JSON only — no defaults filled in. Live mode validates strictly itself.
        Raises LLMUnavailable (network/timeout) or ValueError (no JSON in the reply)."""
        c = self.config
        t0 = time.perf_counter()
        raw = self._request(self.user_prompt(context))
        latency_ms = (time.perf_counter()-t0)*1000
        if c.provider == 'ollama':
            text = raw.get('message',{}).get('content','')
            usage = {'prompt_tokens': raw.get('prompt_eval_count'), 'output_tokens': raw.get('eval_count')}
        else:
            text = raw.get('choices',[{}])[0].get('message',{}).get('content','')
            u = raw.get('usage') or {}
            usage = {'prompt_tokens': u.get('prompt_tokens'), 'output_tokens': u.get('completion_tokens')}
        result = _json_from_text(text)
        if not isinstance(result, dict):
            raise ValueError('LLM JSON is not an object')
        result['_latency_ms'] = latency_ms
        result['_raw_text'] = text   # the model's reply exactly as received (shown in the calculation view)
        if any(v is not None for v in usage.values()):
            result['_usage'] = usage
        return result

    def warm_up(self) -> float:
        """Send one tiny request so the model is loaded before the first real tick. Returns the latency in ms."""
        c = self.config
        msg = [{'role': 'user', 'content': 'Reply with {}'}]
        if c.provider == 'ollama':
            body = {'model': c.model, 'messages': msg, 'stream': False, 'options': {'num_predict': 1}}
            if c.num_ctx is not None:
                body['options']['num_ctx'] = c.num_ctx
            if c.keep_alive is not None:
                body['keep_alive'] = c.keep_alive
        else:
            body = {'model': c.model, 'messages': msg, 'max_tokens': 1}
        t0 = time.perf_counter()
        try:
            self._post(body)
        except Exception as exc:
            raise LLMUnavailable(f'warm-up failed: {exc}') from exc
        return (time.perf_counter() - t0) * 1000

    @staticmethod
    def validate(r):
        trust = float(r.get('trust_score', 0.0))
        risk = float(r.get('risk_score', 1-trust))
        confidence = float(r.get('confidence', 0.0))
        r['trust_score'] = max(0.0, min(1.0, trust))
        r['risk_score'] = max(0.0, min(1.0, risk))
        r['confidence'] = max(0.0, min(1.0, confidence))
        if r.get('action') not in {'ALLOW','STEP-UP','BLOCK'}:
            r['action'] = 'BLOCK'
        if not isinstance(r.get('factors'), list): r['factors'] = []
        if not isinstance(r.get('behavior_assessment'), list): r['behavior_assessment'] = []
        r['reason'] = str(r.get('reason','No reason supplied.'))
        r['required_validation'] = str(r.get('required_validation','Additional verification required.'))
        return r
