"""One small helper so every Module 09 notebook works with any LLM provider.

    from course_llm import ask, ask_json, info

    info()                                   # which provider and model am I using?
    ask("Summarise this review: ...")        # -> str
    ask_json("Extract the product and rating", schema)   # -> dict

Choose a provider in a `.env` file at the repo root (never commit it):

    LLM_PROVIDER=ollama        # anthropic | openai | ollama | groq | gemini | lmstudio
    LLM_MODEL=llama3.2         # optional: override the provider's default model
    ANTHROPIC_API_KEY=...      # whichever key your provider needs

If LLM_PROVIDER is not set, the first available option is used: an
Anthropic key, then an OpenAI key, then a running Ollama server.

There are three code paths. Claude uses the Anthropic SDK. Ollama uses its own
API, because that is the only way to set the context window, and the window
decides how much memory the model takes (see setup/README.md). Every other
provider speaks the OpenAI chat format, so one OpenAI-client path covers them
all; only the web address, key and model name change. Lesson 09.9 shows the
raw calls before you use this helper.

Not sure your model works with the course? Run `self_test()`.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

# name -> (base_url, env var holding the key, default model)
# Defaults change over time; set LLM_MODEL to use anything your provider offers.
OPENAI_COMPATIBLE = {
    "openai":   (None, "OPENAI_API_KEY", "gpt-4o-mini"),
    "ollama":   ("http://localhost:11434/v1", None, "llama3.2"),
    "groq":     ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "gemini":   ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", "gemini-2.5-flash"),
    "lmstudio": ("http://localhost:1234/v1", None, "local-model"),
}
ANTHROPIC_DEFAULT_MODEL = "claude-opus-5-5"
OLLAMA_URL = "http://localhost:11434"


def _load_dotenv() -> None:
    """Read KEY=value lines from the nearest .env file into os.environ."""
    for folder in [Path.cwd(), *Path.cwd().parents]:
        env = folder / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.split("#")[0].strip().strip('"'))
            return


def _ollama_running() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=1)
        return True
    except OSError:
        return False


def _choose_provider() -> str:
    _load_dotenv()
    chosen = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if chosen:
        if chosen != "anthropic" and chosen not in OPENAI_COMPATIBLE:
            raise ValueError(f"Unknown LLM_PROVIDER '{chosen}'. "
                             f"Use one of: anthropic, {', '.join(OPENAI_COMPATIBLE)}")
        return chosen
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if _ollama_running():
        return "ollama"
    raise RuntimeError(
        "No LLM available. Pick one:\n"
        "  free, local : install Ollama (https://ollama.com), run `ollama pull llama3.2`\n"
        "  Claude      : add ANTHROPIC_API_KEY=... to .env\n"
        "  OpenAI      : add OPENAI_API_KEY=... to .env\n"
        "  others      : set LLM_PROVIDER=groq|gemini|lmstudio and its key (see setup/README.md)"
    )


PROVIDER = None
MODEL = None
last_usage = {}          # tokens used by the most recent call: {"input": n, "output": n}


def _setup() -> None:
    global PROVIDER, MODEL
    if PROVIDER is None:
        PROVIDER = _choose_provider()
        default = (ANTHROPIC_DEFAULT_MODEL if PROVIDER == "anthropic"
                   else OPENAI_COMPATIBLE[PROVIDER][2])
        MODEL = os.environ.get("LLM_MODEL") or default


def info() -> str:
    """Print and return which provider and model the notebook is using."""
    _setup()
    text = f"LLM provider: {PROVIDER} · model: {MODEL}"
    print(text)
    return text


# --- Claude (Anthropic SDK) -------------------------------------------------

def _ask_anthropic(messages, system, max_tokens, schema):
    import anthropic
    client = anthropic.Anthropic()
    kwargs = dict(model=MODEL, max_tokens=max_tokens, messages=messages)
    if system:
        kwargs["system"] = system
    if schema:
        kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
    # If the model declines a request, the API retries it on a fallback model.
    response = client.beta.messages.create(
        **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined this request.")
    last_usage.update(input=response.usage.input_tokens, output=response.usage.output_tokens)
    return "".join(block.text for block in response.content if block.type == "text")


# --- Ollama (its own API, so the context window can be set) ------------------

def _ask_ollama(messages, system, max_tokens, schema, temperature):
    if system:
        messages = [{"role": "system", "content": system}, *messages]
    options = {"num_ctx": int(os.environ.get("LLM_CONTEXT", 8192)), "num_predict": max_tokens}
    if temperature is not None:
        options["temperature"] = temperature
    payload = {"model": MODEL, "messages": messages, "stream": False, "options": options}
    if schema:
        payload["format"] = schema
    request = urllib.request.Request(f"{OLLAMA_URL}/api/chat", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as reply:
            data = json.load(reply)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(f"Ollama has no model '{MODEL}'. Run:  ollama pull {MODEL}") from None
        raise
    except urllib.error.URLError:
        raise RuntimeError("Ollama is not running. Start the Ollama app (or `ollama serve`).") from None
    last_usage.update(input=data.get("prompt_eval_count", 0), output=data.get("eval_count", 0))
    return data["message"]["content"]


# --- Everyone else (OpenAI-compatible chat API) ------------------------------

def _ask_openai_compatible(messages, system, max_tokens, schema, temperature):
    from openai import OpenAI
    base_url, key_var, _ = OPENAI_COMPATIBLE[PROVIDER]
    api_key = os.environ.get(key_var) if key_var else "not-needed"
    if key_var and not api_key:
        raise RuntimeError(f"LLM_PROVIDER={PROVIDER} needs {key_var} in your .env file.")
    client = OpenAI(base_url=base_url, api_key=api_key)
    if system:
        messages = [{"role": "system", "content": system}, *messages]
    kwargs = dict(model=MODEL, messages=messages, max_tokens=max_tokens)
    if temperature is not None:
        kwargs["temperature"] = temperature
    if schema:
        kwargs["response_format"] = {"type": "json_schema",
                                     "json_schema": {"name": "answer", "schema": schema}}
    response = client.chat.completions.create(**kwargs)
    if response.usage:
        last_usage.update(input=response.usage.prompt_tokens,
                          output=response.usage.completion_tokens)
    return response.choices[0].message.content or ""


# --- Public functions ---------------------------------------------------------

def chat(messages, system=None, max_tokens=16000, temperature=None, schema=None) -> str:
    """Send a list of {"role": "user"|"assistant", "content": str} messages; return the reply.

    `temperature` is passed to providers that accept it. Current Claude models
    do not accept sampling settings, so it is ignored there (lesson 09.7).
    """
    _setup()
    if PROVIDER == "anthropic":
        return _ask_anthropic(messages, system, max_tokens, schema)
    if PROVIDER == "ollama":
        return _ask_ollama(messages, system, max_tokens, schema, temperature)
    return _ask_openai_compatible(messages, system, max_tokens, schema, temperature)


def ask(prompt: str, system=None, max_tokens=16000, temperature=None) -> str:
    """One question in, one answer out."""
    return chat([{"role": "user", "content": prompt}], system, max_tokens, temperature)


def ask_json(prompt: str, schema: dict, system=None, max_tokens=16000) -> dict:
    """Ask for an answer that matches a JSON schema and return it as a dict."""
    text = chat([{"role": "user", "content": prompt}], system, max_tokens, schema=schema)
    return json.loads(text)


def self_test() -> bool:
    """Check that the chosen model can do what Module 09 needs. Prints a verdict."""
    import time
    info()
    schema = {"type": "object",
              "properties": {"product": {"type": "string"},
                             "sentiment": {"type": "string",
                                           "enum": ["positive", "neutral", "negative"]}},
              "required": ["product", "sentiment"], "additionalProperties": False}
    tests = [("The mug arrived chipped and nobody answered my emails.", "negative"),
             ("Beautiful fairy lights, my whole family loves them!", "positive"),
             ("The doormat is fine. It does the job.", "neutral")]
    start = time.time()
    reply = ask("Reply with exactly one word: hello", max_tokens=200)
    print(f"  plain answer     : {reply.strip()[:40]!r} ({time.time() - start:.1f} s, includes loading)")
    passed = 0
    for text, expected in tests:
        try:
            passed += ask_json(f"Extract the product and sentiment: {text}", schema,
                               max_tokens=2000)["sentiment"] == expected
        except Exception as e:
            print(f"  structured output: failed ({type(e).__name__})")
    print(f"  structured output: {passed}/{len(tests)} correct")
    ok = passed >= 2
    print("  verdict          :", "OK for this course" if ok else
          "struggles with structured output; try a different model (setup/README.md)")
    return ok
