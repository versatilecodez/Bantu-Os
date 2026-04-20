"""
Comprehensive kernel integration tests.

Covers:
- Kernel initialization with various configurations
- Tool registration (sync and async)
- LLM provider abstraction (swapping providers)
- Context isolation between sessions
- Concurrent async request handling
"""
import asyncio
import pytest

from bantu_os.core.kernel.kernel import Kernel
from bantu_os.core.kernel.llm_manager import LLMManager
from bantu_os.core.kernel.providers.base import ChatMessage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_llm(monkeypatch):
    """Replace _build_provider so Kernel doesn't need real API keys."""
    from bantu_os.core.kernel.llm_manager import LLMManager

    class DummyProvider:
        def __init__(self, model: str, **kwargs):
            self.model = model

        async def generate(self, *, messages, temperature=0.7, max_tokens=None, **kwargs):
            return {
                "text": f"dummy-response-for-model-{self.model}",
                "raw": {"provider": "dummy", "model": self.model},
            }

    def _fake_build(self, provider: str, model: str, **kwargs):
        return DummyProvider(model=model, **kwargs)

    monkeypatch.setattr(LLMManager, "_build_provider", _fake_build, raising=True)
    yield DummyProvider


# ---------------------------------------------------------------------------
# Tests: Initialization
# ---------------------------------------------------------------------------

def test_kernel_initialization_default(dummy_llm):
    """Kernel() with no args starts with defaults and no tools."""
    kernel = Kernel()
    assert isinstance(kernel.llm, LLMManager)
    assert kernel.tools == {}
    assert kernel.memory is None


def test_kernel_initialization_with_tools(dummy_llm):
    """Kernel() accepts a tools dict and registers them."""
    def adder(a: int, b: int) -> int:
        return a + b

    kernel = Kernel(tools={"adder": adder})
    assert "adder" in kernel.tools
    assert kernel.use_tool("adder", a=2, b=3) == 5


def test_kernel_initialization_with_memory(dummy_llm):
    """Kernel() accepts a Memory instance."""
    from bantu_os.memory.memory import Memory
    from bantu_os.memory.vector_store import VectorDBStore

    store = VectorDBStore(dim=768)
    mem = Memory(store=store, dim=768)
    kernel = Kernel(memory=mem)
    assert kernel.memory is mem


def test_kernel_initialization_custom_provider(monkeypatch):
    """Kernel() accepts provider and provider_model kwargs."""
    from bantu_os.core.kernel.llm_manager import LLMManager

    class CustomProvider:
        def __init__(self, model: str, **kwargs):
            self.model = model

        async def generate(self, *, messages, **kwargs):
            return {"text": f"custom-{model}", "raw": {}}

    def _fake_build(self, provider, model, **kwargs):
        return CustomProvider(model=model, **kwargs)

    monkeypatch.setattr(LLMManager, "_build_provider", _fake_build, raising=True)

    kernel = Kernel(provider="custom", provider_model="my-model")
    assert kernel.llm.active_model == "default"


# ---------------------------------------------------------------------------
# Tests: Tool Registration
# ---------------------------------------------------------------------------

def test_kernel_tool_registration_single(dummy_llm):
    """register_tool() adds a single tool callable."""
    kernel = Kernel()

    def greet(name: str) -> str:
        return f"Hello, {name}!"

    kernel.register_tool("greet", greet)
    assert "greet" in kernel.tools
    assert kernel.use_tool("greet", name="Alice") == "Hello, Alice!"


def test_kernel_tool_registration_override(dummy_llm):
    """register_tool() overwrites an existing tool with the same name."""
    kernel = Kernel()
    kernel.register_tool("echo", lambda value: "first")
    kernel.register_tool("echo", lambda value: "second")
    assert kernel.use_tool("echo", value="x") == "second"


def test_kernel_tool_missing_keyerror(dummy_llm):
    """use_tool() raises KeyError when tool is not registered."""
    kernel = Kernel()
    with pytest.raises(KeyError, match="nonexistent"):
        kernel.use_tool("nonexistent")


@pytest.mark.asyncio
async def test_kernel_use_tool_async_success(dummy_llm):
    """use_tool_async() awaits async callables correctly."""
    kernel = Kernel()

    async def slow_echo(text: str) -> str:
        await asyncio.sleep(0.01)
        return text

    kernel.register_tool("slow_echo", slow_echo)
    result = await kernel.use_tool_async("slow_echo", text="async-hello")
    assert result == "async-hello"


@pytest.mark.asyncio
async def test_kernel_use_tool_async_sync(dummy_llm):
    """use_tool_async() also works with synchronous callables."""
    kernel = Kernel()
    kernel.register_tool("sync_add", lambda a, b: a + b)
    result = await kernel.use_tool_async("sync_add", a=10, b=20)
    assert result == 30


# ---------------------------------------------------------------------------
# Tests: LLM Provider Abstraction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_kernel_llm_provider_abstraction_single(dummy_llm):
    """Kernel works with a single loaded provider and generates correctly."""
    kernel = Kernel(provider_model="mock-model-v1")

    async def mock_generate(*, messages, **kwargs):
        return {"text": f"provider=mock-model-v1", "raw": {}}

    kernel.llm.generate = mock_generate

    msgs = [{"role": "user", "content": "hi"}]
    result = await kernel.generate_response(messages=msgs)
    assert result["text"] == "provider=mock-model-v1"


@pytest.mark.asyncio
async def test_kernel_llm_provider_abstraction_multiple_models(dummy_llm):
    """LLMManager can load multiple providers and switch between them."""
    kernel = Kernel(provider_model="model-a")

    async def gen_a(*, messages, **kw):
        return {"text": "response-from-model-a", "raw": {}}

    async def gen_b(*, messages, **kw):
        return {"text": "response-from-model-b", "raw": {}}

    # Load second model, then patch its provider's generate
    kernel.llm.load_model("model-b", provider="openai", model="model-b")
    kernel.llm.models["default"].generate = gen_a
    kernel.llm.models["model-b"].generate = gen_b

    kernel.llm.set_active_model("default")
    r1 = await kernel.generate_response(messages=[{"role": "user", "content": "a"}])
    assert r1["text"] == "response-from-model-a"

    kernel.llm.set_active_model("model-b")
    r2 = await kernel.generate_response(messages=[{"role": "user", "content": "b"}])
    assert r2["text"] == "response-from-model-b"


@pytest.mark.asyncio
async def test_kernel_llm_provider_temperature_passed(dummy_llm):
    """generate_response() passes temperature and max_tokens to provider."""
    captured = {}

    class CapturingProvider:
        def __init__(self, model: str, **kw):
            self.model = model

        async def generate(self, *, messages, temperature=0.7, max_tokens=None, **kw):
            captured["temperature"] = temperature
            captured["max_tokens"] = max_tokens
            return {"text": "ok", "raw": {}}

    kernel = Kernel()
    # Replace the loaded provider with our capturing one
    provider_instance = CapturingProvider(model="cap-model")
    kernel.llm.models[kernel.llm.active_model] = provider_instance

    await kernel.generate_response(
        messages=[{"role": "user", "content": "x"}],
        temperature=0.5,
        max_tokens=256,
    )
    assert captured["temperature"] == 0.5
    assert captured["max_tokens"] == 256


# ---------------------------------------------------------------------------
# Tests: Context Isolation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_kernel_context_isolation_no_shared_state(dummy_llm):
    """Two Kernel instances do not share tool registries or LLMManager state."""
    kernel_a = Kernel(tools={"tool_a": lambda: "a"})
    kernel_b = Kernel(tools={"tool_b": lambda: "b"})

    assert "tool_a" in kernel_a.tools
    assert "tool_a" not in kernel_b.tools
    assert "tool_b" in kernel_b.tools
    assert "tool_b" not in kernel_a.tools


@pytest.mark.asyncio
async def test_kernel_context_isolation_llm_manager(dummy_llm):
    """Each Kernel has its own LLMManager instance."""
    kernel_a = Kernel(provider_model="model-a")
    kernel_b = Kernel(provider_model="model-b")

    assert kernel_a.llm is not kernel_b.llm
    assert kernel_a.llm.active_model == "default"
    assert kernel_b.llm.active_model == "default"


@pytest.mark.asyncio
async def test_kernel_context_isolation_messages(dummy_llm):
    """process_input() context parameter is not mutated across calls."""
    call_count = [0]

    async def counting_generate(*, messages, **kw):
        call_count[0] += 1
        count = call_count[0]
        return {"text": f"call-{count}", "raw": {}}

    kernel = Kernel(provider_model="ctx-test")
    kernel.llm.generate = counting_generate

    ctx1 = [{"role": "user", "content": "first"}]
    r1 = await kernel.process_input(text="second", context=ctx1)
    assert r1 == "call-1"

    ctx2 = [{"role": "user", "content": "third"}]
    r2 = await kernel.process_input(text="fourth", context=ctx2)
    assert r2 == "call-2"

    assert ctx1 == [{"role": "user", "content": "first"}]
    assert ctx2 == [{"role": "user", "content": "third"}]


# ---------------------------------------------------------------------------
# Tests: Concurrent Requests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_kernel_concurrent_generate_response(dummy_llm):
    """Multiple concurrent generate_response() calls all complete successfully."""
    kernel = Kernel()

    call_index = [0]

    async def fake_gen(**kw):
        idx = call_index[0]
        call_index[0] += 1
        return {"text": f"result-{idx}", "raw": {}}

    kernel.llm.generate = fake_gen

    async def make_coro(i):
        return await kernel.generate_response(messages=[{"role": "user", "content": str(i)}])

    results = await asyncio.gather(*[make_coro(i) for i in range(10)])
    texts = {r["text"] for r in results}
    assert len(texts) == 10


@pytest.mark.asyncio
async def test_kernel_concurrent_process_input(dummy_llm):
    """process_input() handles concurrent calls without interleaving messages."""
    kernel = Kernel()

    captured_messages = []

    async def capturing_generate(*, messages, **kw):
        captured_messages.append(list(messages))
        return {"text": "concurrent-ok", "raw": {}}

    kernel.llm.generate = capturing_generate

    async def make_coro(i):
        return await kernel.process_input(text=f"input-{i}", system_prompt=f"sys-{i}")

    await asyncio.gather(*[make_coro(i) for i in range(5)])

    assert len(captured_messages) == 5
    for i, msgs in enumerate(captured_messages):
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == f"input-{i}"


@pytest.mark.asyncio
async def test_kernel_concurrent_tool_calls(dummy_llm):
    """run_tool_calls() executes multiple tool calls concurrently."""
    kernel = Kernel()

    async def slow_tool(delay: float, name: str):
        await asyncio.sleep(delay)
        return name.upper()

    kernel.register_tool("slow_upper", slow_tool)

    calls = [
        {"name": "slow_upper", "args": {"delay": 0.02, "name": "alice"}},
        {"name": "slow_upper", "args": {"delay": 0.01, "name": "bob"}},
        {"name": "slow_upper", "args": {"delay": 0.03, "name": "carol"}},
    ]

    outcomes = await kernel.run_tool_calls(calls)
    assert len(outcomes) == 3
    names = {o.get("result") for o in outcomes if "result" in o}
    assert names == {"ALICE", "BOB", "CAROL"}


@pytest.mark.asyncio
async def test_kernel_concurrent_mixed_tool_and_llm(dummy_llm):
    """Concurrent mix of LLM calls and tool calls completes without error."""
    kernel = Kernel()

    kernel.register_tool("add", lambda a, b: a + b)

    async def fake_gen(**kw):
        return {"text": "llm-ok", "raw": {}}

    kernel.llm.generate = fake_gen

    async def run_query(i):
        return await kernel.process_input(text=f"query-{i}")

    async def run_tool(i):
        return await kernel.use_tool_async("add", a=i, b=i)

    async def run_response():
        return await kernel.generate_response(messages=[{"role": "user", "content": "hi"}])

    results = await asyncio.gather(
        run_query(0),
        run_tool(1),
        run_response(),
        return_exceptions=True,
    )
    assert len(results) == 3
    text, add_result, llm_result = results
    assert isinstance(text, str)
    assert isinstance(add_result, int)
    assert isinstance(llm_result, dict)


# ---------------------------------------------------------------------------
# Tests: Error Handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_kernel_run_tool_calls_keyerror(dummy_llm):
    """run_tool_calls() returns error dict for missing tools, not raising."""
    kernel = Kernel()
    calls = [{"name": "does_not_exist", "args": {}}]
    outcomes = await kernel.run_tool_calls(calls)
    assert outcomes[0].get("error") is not None
    assert "does_not_exist" in outcomes[0]["error"]


@pytest.mark.asyncio
async def test_kernel_run_tool_calls_exception(dummy_llm):
    """run_tool_calls() catches and reports tool exceptions."""
    kernel = Kernel()

    def raising_tool():
        raise ValueError("boom")

    kernel.register_tool("raise_err", raising_tool)
    calls = [{"name": "raise_err"}]
    outcomes = await kernel.run_tool_calls(calls)
    assert "error" in outcomes[0]
    assert "boom" in outcomes[0]["error"]
