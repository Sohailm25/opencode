# RLM (Recursive Language Model) Implementation Plan for OpenCode

## Document Information

- **Version**: 1.0
- **Date**: 2026-01-04
- **Paper Reference**: `docs/rlm/rlm-paper.pdf` (arXiv:2512.24601)
- **Official Implementation**: https://github.com/alexzhang13/rlm

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Paper Reference Guide](#2-paper-reference-guide)
3. [GitHub Codebase Reference](#3-github-codebase-reference)
4. [Current OpenCode Implementation Gap Analysis](#4-current-opencode-implementation-gap-analysis)
5. [Architecture Design](#5-architecture-design)
6. [Implementation Phases](#6-implementation-phases)
7. [Detailed Component Specifications](#7-detailed-component-specifications)
8. [Integration with OpenCode](#8-integration-with-opencode)
9. [Testing Strategy](#9-testing-strategy)
10. [Configuration & User Interface](#10-configuration--user-interface)

---

## 1. Executive Summary

### What is RLM?

From **Paper §1, Page 2**:
> "The key insight is that long prompts should not be fed into the neural network (e.g., Transformer) directly but should instead be treated as part of the environment that the LLM can symbolically interact with."

### Core Innovation

RLMs replace the standard `llm.completion(prompt, model)` paradigm with a REPL-based approach where:
1. Context is loaded as a **variable** in a Python environment (not in the prompt)
2. The LLM writes **executable code** to interact with this context
3. The LLM can **recursively call sub-LLMs** via `llm_query()` functions
4. Execution continues in a loop until `FINAL()` or `FINAL_VAR()` is called

### Performance Results (Paper Table 1, Page 4)

| Benchmark | Base GPT-5 | RLM(GPT-5) | Improvement |
|-----------|------------|------------|-------------|
| BrowseComp+ (1K) | 0.00%* | 91.33% | +91.33% |
| OOLONG | 44.00% | 56.50% | +28.4% |
| OOLONG-Pairs | 0.04% | 58.00% | +1450x |
| CodeQA | 24.00%* | 62.00% | +158% |

*Context exceeded model limits

---

## 2. Paper Reference Guide

### Section 1: Introduction (Pages 1-2)

**Key Concepts:**
- **Context Rot** (Page 1): "the phenomenon where the quality of even frontier models like GPT-5 degrades quickly as context gets longer"
- **Out-of-core Inspiration** (Page 1): "data-processing systems with a small but fast main memory can process far larger datasets by cleverly managing how data is fetched into memory"

**Figure 2 (Page 2)** - Critical Architecture Diagram:
- Shows the RLM treating prompts as environment variables
- Illustrates the REPL loop with `In[1]`, `Out[1]`, etc.
- Shows recursive sub-LM calls at depth=1

### Section 2: Methods (Pages 3-4)

**2.2 Methods and Baselines (Page 4):**
> "We implement an RLM that loads its context as a string in the memory of a Python REPL environment. The REPL environment also loads in a module that allows it to query a sub-LM inside the environment. The system prompt is fixed across all experiments (see Appendix D)."

**Key Implementation Details:**
- GPT-5 uses GPT-5-mini for recursive sub-calls
- Max recursion depth = 1 (sub-calls are plain LMs, not RLMs)
- System prompt is task-agnostic

### Section 3: Results (Pages 4-6)

**Observation 2 (Page 5):**
> "The REPL environment is necessary for handling long inputs, while the recursive sub-calling of RLMs provides strong benefits on information-dense inputs."

**Cost Analysis (Figure 3, Page 5):**
- RLM median cost is comparable to or cheaper than base model
- High variance due to trajectory length differences

### Section 3.1: Emergent Patterns (Pages 6-7)

**Key Strategies Observed:**
1. **Filtering via regex** - Using code to filter context based on model priors
2. **Chunking + recursive sub-calls** - Decomposing context uniformly or by keyword
3. **Answer verification** - Using sub-LMs with small contexts to verify
4. **Variable-based long output** - Stitching sub-LM outputs through REPL variables

### Section 5: Limitations (Page 8)

**Critical Limitations to Address:**
1. Synchronous sub-calls (blocking) - could use async
2. Max recursion depth of 1 - deeper recursion unexplored
3. Models not trained specifically as RLMs

### Appendix A: Negative Results (Page 13)

**Important Warnings:**
1. "Using the exact same RLM system prompt across all models can be problematic"
2. "Models without sufficient coding capabilities struggle as RLMs"
3. "Thinking models without sufficient output tokens struggle as RLMs"
4. "RLMs without asynchronous LM calls are slow"

### Appendix D: System Prompts (Pages 24-28)

**CRITICAL: The complete system prompts are in Appendix D.1**

---

## 3. GitHub Codebase Reference

### Repository Structure

```
github.com/alexzhang13/rlm/
├── rlm/
│   ├── __init__.py              # Exports RLM class
│   ├── core/
│   │   ├── rlm.py               # Main RLM class (11,365 bytes)
│   │   ├── types.py             # Type definitions (7,409 bytes)
│   │   ├── lm_handler.py        # LM Handler for sub-calls (6,448 bytes)
│   │   └── comms_utils.py       # Socket protocol (8,222 bytes)
│   ├── environments/
│   │   ├── __init__.py          # get_environment() router
│   │   ├── base_env.py          # BaseEnv abstract class (1,898 bytes)
│   │   ├── local_repl.py        # LocalREPL implementation (9,786 bytes)
│   │   ├── docker_repl.py       # DockerREPL implementation (10,077 bytes)
│   │   └── modal_repl.py        # ModalREPL implementation (15,721 bytes)
│   ├── clients/
│   │   ├── __init__.py          # Client exports
│   │   ├── base_lm.py           # BaseLM abstract class (1,011 bytes)
│   │   ├── openai.py            # OpenAI client (4,152 bytes)
│   │   ├── anthropic.py         # Anthropic client (4,256 bytes)
│   │   ├── litellm.py           # LiteLLM client (3,934 bytes)
│   │   └── portkey.py           # Portkey client (3,692 bytes)
│   ├── utils/
│   │   ├── prompts.py           # System prompts (9,285 bytes) ⭐ CRITICAL
│   │   ├── parsing.py           # Code block/FINAL parsing (5,548 bytes)
│   │   └── rlm_utils.py         # Utilities (370 bytes)
│   └── logger/
│       └── logger.py            # RLMLogger for trajectories
├── examples/
│   ├── rlm_example.py           # Basic usage example
│   ├── lm_in_repl.py            # LLM query within REPL
│   └── docker_repl_example.py   # Docker REPL example
└── visualizer/                   # Node.js trajectory visualizer
```

### Key File References

#### `rlm/utils/prompts.py` - System Prompt (MOST IMPORTANT)

```python
# Lines 1-100: RLM_SYSTEM_PROMPT
RLM_SYSTEM_PROMPT = textwrap.dedent("""
You are tasked with answering a query with associated context. You can access,
transform, and analyze this context interactively in a REPL environment that
can recursively query sub-LLMs...

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information...
2. A `llm_query` function that allows you to query an LLM (that can handle
   around 500K chars) inside your REPL environment.
3. A `llm_query_batched` function for concurrent queries...
4. The ability to use `print()` statements...

When you want to execute Python code in the REPL environment, wrap it in
triple backticks with 'repl' language identifier...

IMPORTANT: When you are done with the iterative process, you MUST provide a
final answer inside a FINAL function:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable...
""")

# Lines 101-130: build_rlm_system_prompt()
def build_rlm_system_prompt(system_prompt, query_metadata):
    """Build initial system prompt with context metadata."""
    metadata_prompt = f"Your context is a {context_type} with {context_total_length}
    total characters, and is broken up into chunks of char lengths: {context_lengths}."
    return [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": metadata_prompt},
    ]

# Lines 131-150: USER_PROMPT templates
USER_PROMPT = """Think step-by-step on what to do using the REPL environment..."""
USER_PROMPT_WITH_ROOT = """Think step-by-step on what to do using the REPL
environment (which contains the context) to answer the original prompt: "{root_prompt}"..."""
```

#### `rlm/utils/parsing.py` - Response Parsing

```python
# find_code_blocks(): Extract ```repl``` blocks
def find_code_blocks(response: str) -> list[str]:
    pattern = r"```repl\s*(.*?)```"
    return re.findall(pattern, response, re.DOTALL)

# find_final_answer(): Detect FINAL() or FINAL_VAR()
def find_final_answer(response: str) -> tuple[str, str] | None:
    final_var_match = re.search(r"FINAL_VAR\(([^)]+)\)", response)
    if final_var_match:
        return ("FINAL_VAR", final_var_match.group(1).strip())

    final_match = re.search(r"FINAL\((.*?)\)", response, re.DOTALL)
    if final_match:
        return ("FINAL", final_match.group(1).strip())
    return None

# check_for_final_answer(): Resolve FINAL_VAR from REPL locals
def check_for_final_answer(final_answer, repl_locals):
    if final_answer[0] == "FINAL_VAR":
        var_name = final_answer[1]
        if var_name in repl_locals:
            return str(repl_locals[var_name])
        return f"Error: Variable '{var_name}' not found"
    return final_answer[1]
```

#### `rlm/environments/local_repl.py` - REPL Implementation

```python
class LocalREPL(NonIsolatedEnv):
    # Safe builtins - blocks dangerous functions
    SAFE_BUILTINS = {
        # 80+ safe builtins
        # EXCLUDES: eval, exec, compile, globals, locals, input
    }

    def __init__(self, lm_handler_address, context_payload, init_code=None):
        self._globals = {"__builtins__": self.SAFE_BUILTINS}
        self._locals = {}
        self._lm_handler_address = lm_handler_address

    def load_context(self, context_payload):
        """Load context as 'context' variable in REPL namespace."""
        if isinstance(context_payload, str):
            self._locals["context"] = context_payload
        elif isinstance(context_payload, dict):
            self._locals["context"] = context_payload
        # ...

    def execute_code(self, code: str) -> REPLResult:
        combined_globals = {
            **self._globals,
            "llm_query": self._llm_query,
            "llm_query_batched": self._llm_query_batched,
            "FINAL_VAR": self._final_var,
        }

        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        with redirect_stdout(stdout_capture):
            exec(code, combined_globals, self._locals)

        return REPLResult(
            stdout=stdout_capture.getvalue(),
            stderr=stderr,
            locals=self._locals,
            rlm_calls=self._rlm_calls
        )

    def _llm_query(self, prompt: str) -> str:
        """Send query to LM Handler via socket."""
        response = send_lm_request(self._lm_handler_address, prompt)
        return response.content

    def _llm_query_batched(self, prompts: list[str]) -> list[str]:
        """Send batched queries concurrently."""
        return send_lm_request_batched(self._lm_handler_address, prompts)
```

#### `rlm/core/rlm.py` - Main RLM Class

```python
class RLM:
    def __init__(
        self,
        backend: str = "openai",
        backend_kwargs: dict = {},
        environment: str = "local",
        environment_kwargs: dict = {},
        max_depth: int = 1,
        max_iterations: int = 20,
        verbose: bool = False,
        logger: RLMLogger = None,
    ):
        self.backend = backend
        self.environment = environment
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        # ...

    def completion(self, query: str, context: str = None) -> RLMChatCompletion:
        """Main entry point - replaces llm.completion()"""

        # 1. Setup REPL environment with context
        repl = get_environment(self.environment, **self.environment_kwargs)
        repl.setup()
        repl.load_context(context or query)

        # 2. Build system prompt with metadata
        query_metadata = QueryMetadata(context)
        messages = build_rlm_system_prompt(RLM_SYSTEM_PROMPT, query_metadata)

        # 3. Main iteration loop
        for iteration in range(self.max_iterations):
            # Add user prompt
            messages.append(build_user_prompt(query, iteration))

            # Get LLM response
            response = self.client.completion(messages)
            messages.append({"role": "assistant", "content": response})

            # Parse for ```repl``` code blocks
            code_blocks = find_code_blocks(response)

            # Execute each code block
            for code in code_blocks:
                result = repl.execute_code(code)
                # Add truncated result to messages
                messages.append({
                    "role": "user",
                    "content": format_execution_result(result, max_chars=8192)
                })

            # Check for FINAL/FINAL_VAR
            final_answer = find_final_answer(response)
            if final_answer:
                answer = check_for_final_answer(final_answer, repl._locals)
                return RLMChatCompletion(response=answer, ...)

        # 4. No final answer after max_iterations
        return self._default_answer(messages)
```

#### `rlm/core/lm_handler.py` - Sub-LM Management

```python
class LMHandler:
    """Manages sub-LM clients and provides socket-based access for REPL."""

    def __init__(self, host="localhost", port=0):
        self.host = host
        self.port = port
        self.clients = {}
        self.primary_client = None

    def register_client(self, model_name: str, client: BaseLM):
        """Register a client for sub-LM calls."""
        self.clients[model_name] = client
        if self.primary_client is None:
            self.primary_client = client

    def start(self):
        """Start TCP server for llm_query requests."""
        self.server = ThreadingTCPServer(
            (self.host, self.port),
            LMRequestHandler
        )
        self.server.lm_handler = self
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def address(self):
        return f"{self.host}:{self.server.server_address[1]}"
```

---

## 4. Current OpenCode Implementation Gap Analysis

### What OpenCode Currently Has

| File | Content | RLM Compliance |
|------|---------|----------------|
| `src/session/prompt/rlm.txt` | 32-line "think harder" prompt | ❌ Wrong approach |
| `src/provider/transform.ts` | Sets `rlm.enabled`, `maxDepth: 3` | ❌ Does nothing |
| `src/flag/flag.ts` | `OPENCODE_EXPERIMENTAL_RLM` flag | ✅ Good |
| `src/config/config.ts` | `experimental.rlm_mode` option | ✅ Good |
| `src/session/llm.ts` | Appends RLM prompt, enables thinking | ⚠️ Partial |
| UI components | Toggle and status indicator | ✅ Good |

### Critical Gaps

| Component | Required | OpenCode Has | Gap |
|-----------|----------|--------------|-----|
| REPL Environment | Python/JS sandbox with `context` variable | None | **CRITICAL** |
| `llm_query()` | Socket-based sub-LM calls | None | **CRITICAL** |
| `llm_query_batched()` | Concurrent sub-LM calls | None | **CRITICAL** |
| Code Block Parsing | Extract `\`\`\`repl` blocks | None | **HIGH** |
| FINAL/FINAL_VAR | Termination detection | None | **HIGH** |
| Iteration Loop | Multi-turn with REPL execution | None | **HIGH** |
| Output Truncation | ~8192 char limit | None | **MEDIUM** |
| Context Metadata | `{context_type}`, `{context_total_length}` | None | **MEDIUM** |
| System Prompt | Official ~3500 char prompt | Wrong prompt | **HIGH** |

---

## 5. Architecture Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           OpenCode Session                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  User Query + Context                                                    │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     RLM Orchestrator                             │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │    │
│  │  │ System      │  │ Iteration   │  │ Response Parser         │  │    │
│  │  │ Prompt      │  │ Controller  │  │ - find_code_blocks()    │  │    │
│  │  │ Builder     │  │             │  │ - find_final_answer()   │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│         │                    │                      │                    │
│         ▼                    ▼                      ▼                    │
│  ┌─────────────┐     ┌─────────────┐        ┌─────────────┐             │
│  │ Root LLM    │     │ REPL        │        │ LM Handler  │             │
│  │ (Claude,    │◄───►│ Environment │◄──────►│ (Sub-model  │             │
│  │  GPT-5)     │     │ (Sandbox)   │        │  manager)   │             │
│  └─────────────┘     └─────────────┘        └─────────────┘             │
│                             │                      │                     │
│                             ▼                      ▼                     │
│                      ┌─────────────┐        ┌─────────────┐             │
│                      │ context     │        │ Sub-LLM     │             │
│                      │ variable    │        │ (Haiku,     │             │
│                      │ llm_query() │        │  GPT-5-mini)│             │
│                      │ print()     │        └─────────────┘             │
│                      └─────────────┘                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Interactions

```
┌─────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  User   │───►│ RLM         │───►│ Root LLM     │───►│ Response    │
│  Query  │    │ Orchestrator│    │ (Main Model) │    │ with code   │
└─────────┘    └─────────────┘    └──────────────┘    └─────────────┘
                     │                                       │
                     │                                       ▼
                     │                              ┌─────────────────┐
                     │                              │ Parse for       │
                     │                              │ ```repl``` and  │
                     │                              │ FINAL()         │
                     │                              └─────────────────┘
                     │                                       │
                     ▼                                       ▼
              ┌─────────────┐                        ┌─────────────┐
              │ REPL        │◄───────────────────────│ Execute     │
              │ Environment │                        │ Code Blocks │
              └─────────────┘                        └─────────────┘
                     │                                       │
                     │  llm_query() called                   │
                     ▼                                       │
              ┌─────────────┐                                │
              │ LM Handler  │                                │
              │ (Sub-model) │                                │
              └─────────────┘                                │
                     │                                       │
                     │  Response                             │
                     ▼                                       ▼
              ┌─────────────────────────────────────────────────┐
              │ Truncated stdout/stderr added to message history │
              └─────────────────────────────────────────────────┘
                                      │
                                      ▼
                              ┌─────────────┐
                              │ Next        │
                              │ Iteration   │
                              │ or FINAL    │
                              └─────────────┘
```

---

## 6. Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)

**Goal**: Create the foundational REPL environment and parsing utilities.

#### 1.1 REPL Environment

**Files to Create:**
- `packages/opencode/src/rlm/environments/base.ts`
- `packages/opencode/src/rlm/environments/local-repl.ts`

**Reference**: `github.com/alexzhang13/rlm/blob/main/rlm/environments/local_repl.py`

**Key Implementation:**
```typescript
// base.ts
export interface REPLResult {
  stdout: string;
  stderr: string;
  locals: Record<string, any>;
  executionTime: number;
  rlmCalls: number;
}

export abstract class BaseEnvironment {
  abstract setup(): Promise<void>;
  abstract loadContext(context: string | Record<string, any>): void;
  abstract executeCode(code: string): Promise<REPLResult>;
  abstract cleanup(): void;
}
```

**Implementation Options for LocalREPL:**

| Option | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| `isolated-vm` | Fast, true isolation | JS only, not Python | Good for MVP |
| Pyodide (Python WASM) | True Python | 10MB+ download, slow startup | Best for full compatibility |
| Docker containers | Full isolation, real Python | Requires Docker, slow | Production option |
| `vm2` (deprecated) | Simple | Security issues | Avoid |

#### 1.2 Response Parsing

**Files to Create:**
- `packages/opencode/src/rlm/parsing/code-blocks.ts`
- `packages/opencode/src/rlm/parsing/final-answer.ts`

**Reference**: `github.com/alexzhang13/rlm/blob/main/rlm/utils/parsing.py`

```typescript
// code-blocks.ts
export function findCodeBlocks(response: string): string[] {
  const pattern = /```repl\s*([\s\S]*?)```/g;
  const matches: string[] = [];
  let match;
  while ((match = pattern.exec(response)) !== null) {
    matches.push(match[1].trim());
  }
  return matches;
}

// final-answer.ts
export type FinalAnswer =
  | { type: 'FINAL'; value: string }
  | { type: 'FINAL_VAR'; variable: string };

export function findFinalAnswer(response: string): FinalAnswer | null {
  // Check FINAL_VAR first (more specific)
  const varMatch = response.match(/FINAL_VAR\(([^)]+)\)/);
  if (varMatch) {
    return { type: 'FINAL_VAR', variable: varMatch[1].trim() };
  }

  // Check FINAL
  const finalMatch = response.match(/FINAL\(([\s\S]*?)\)/);
  if (finalMatch) {
    return { type: 'FINAL', value: finalMatch[1].trim() };
  }

  return null;
}
```

#### 1.3 System Prompts

**Files to Create:**
- `packages/opencode/src/rlm/prompts/system.ts`
- `packages/opencode/src/rlm/prompts/user.ts`

**Reference**:
- Paper Appendix D.1 (Pages 24-25)
- `github.com/alexzhang13/rlm/blob/main/rlm/utils/prompts.py`

**CRITICAL**: Use the exact prompt from the paper, adapted for our context:

```typescript
// system.ts
export const RLM_SYSTEM_PROMPT = `You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A \`context\` variable that contains extremely important information about your query. You should check the content of the \`context\` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A \`llm_query\` function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment.
3. A \`llm_query_batched\` function that allows you to query multiple prompts concurrently: \`llm_query_batched(prompts: List[str]) -> List[str]\`. This is much faster than sequential \`llm_query\` calls when you have multiple independent queries. Results are returned in the same order as the input prompts.
4. The ability to use \`print()\` statements to view the output of your REPL code and continue your reasoning.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.

Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example:
\`\`\`repl
chunk = context[:10000]
answer = llm_query(f"What is the magic number in the context? Here is the chunk: {chunk}")
print(answer)
\`\`\`

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.`;

export interface QueryMetadata {
  contextType: string;
  contextTotalLength: number;
  contextLengths: number[];
}

export function buildSystemPrompt(metadata: QueryMetadata): Array<{role: string, content: string}> {
  const metadataPrompt = `Your context is a ${metadata.contextType} with ${metadata.contextTotalLength} total characters, and is broken up into chunks of char lengths: ${JSON.stringify(metadata.contextLengths)}.`;

  return [
    { role: "system", content: RLM_SYSTEM_PROMPT },
    { role: "assistant", content: metadataPrompt },
  ];
}
```

---

### Phase 2: LM Handler & Sub-Model Integration (Week 2-3)

**Goal**: Enable recursive sub-LM calls from within the REPL.

#### 2.1 LM Handler

**Files to Create:**
- `packages/opencode/src/rlm/lm-handler/handler.ts`
- `packages/opencode/src/rlm/lm-handler/protocol.ts`

**Reference**: `github.com/alexzhang13/rlm/blob/main/rlm/core/lm_handler.py`

```typescript
// handler.ts
export class LMHandler {
  private clients: Map<string, BaseLM> = new Map();
  private primaryClient: BaseLM | null = null;
  private server: net.Server | null = null;

  registerClient(modelName: string, client: BaseLM): void {
    this.clients.set(modelName, client);
    if (!this.primaryClient) {
      this.primaryClient = client;
    }
  }

  async start(): Promise<string> {
    return new Promise((resolve) => {
      this.server = net.createServer(this.handleConnection.bind(this));
      this.server.listen(0, 'localhost', () => {
        const address = this.server!.address() as net.AddressInfo;
        resolve(`localhost:${address.port}`);
      });
    });
  }

  private async handleConnection(socket: net.Socket): Promise<void> {
    // Handle llm_query requests from REPL
    socket.on('data', async (data) => {
      const request = JSON.parse(data.toString());
      const response = await this.primaryClient!.completion(request.prompt);
      socket.write(JSON.stringify({ content: response }));
    });
  }
}
```

#### 2.2 Sub-Model Client Integration

**Files to Modify:**
- `packages/opencode/src/provider/provider.ts`

**Reference**: `github.com/alexzhang13/rlm/blob/main/rlm/clients/`

**Sub-model Selection Strategy (from Paper Page 4):**
> "For the GPT-5 experiments, we use GPT-5-mini for the recursive LMs and GPT-5 for the root LM"

**Recommended OpenCode Mappings:**
| Root Model | Sub-Model | Rationale |
|------------|-----------|-----------|
| Claude Opus | Claude Haiku | Cost-effective, fast |
| Claude Sonnet | Claude Haiku | Cost-effective, fast |
| GPT-5 | GPT-5-mini | Paper recommendation |
| GPT-4o | GPT-4o-mini | Cost-effective |
| Gemini Pro | Gemini Flash | Cost-effective |

---

### Phase 3: Main RLM Orchestrator (Week 3-4)

**Goal**: Implement the main completion loop that ties everything together.

#### 3.1 RLM Orchestrator

**Files to Create:**
- `packages/opencode/src/rlm/orchestrator.ts`
- `packages/opencode/src/rlm/types.ts`

**Reference**: `github.com/alexzhang13/rlm/blob/main/rlm/core/rlm.py`

```typescript
// orchestrator.ts
export interface RLMConfig {
  maxIterations: number;      // Default: 20
  maxDepth: number;           // Default: 1
  outputTruncation: number;   // Default: 8192
  environment: 'local' | 'docker';
  subModel: string;           // Model for llm_query
  verbose: boolean;
}

export interface RLMCompletion {
  response: string;
  iterations: number;
  totalCost: number;
  executionTime: number;
  trajectory: RLMIteration[];
}

export class RLMOrchestrator {
  private config: RLMConfig;
  private repl: BaseEnvironment;
  private lmHandler: LMHandler;
  private rootClient: LLMClient;

  async completion(query: string, context?: string): Promise<RLMCompletion> {
    // 1. Setup
    await this.repl.setup();
    this.repl.loadContext(context || query);
    const lmAddress = await this.lmHandler.start();

    // 2. Build messages
    const metadata = this.computeMetadata(context || query);
    const messages = buildSystemPrompt(metadata);

    // 3. Main loop
    for (let i = 0; i < this.config.maxIterations; i++) {
      // Add user prompt
      messages.push(buildUserPrompt(query, i));

      // Get LLM response
      const response = await this.rootClient.completion(messages);
      messages.push({ role: 'assistant', content: response });

      // Parse code blocks
      const codeBlocks = findCodeBlocks(response);

      // Execute each block
      for (const code of codeBlocks) {
        const result = await this.repl.executeCode(code);
        const truncatedOutput = this.truncateOutput(result.stdout, this.config.outputTruncation);
        messages.push({
          role: 'user',
          content: `REPL Output:\n${truncatedOutput}`
        });
      }

      // Check for final answer
      const finalAnswer = findFinalAnswer(response);
      if (finalAnswer) {
        const answer = this.resolveFinalAnswer(finalAnswer);
        return { response: answer, iterations: i + 1, ... };
      }
    }

    // 4. Default answer after max iterations
    return this.generateDefaultAnswer(messages);
  }

  private truncateOutput(output: string, maxChars: number): string {
    if (output.length <= maxChars) return output;
    return output.slice(0, maxChars) + `\n... [truncated ${output.length - maxChars} chars]`;
  }

  private resolveFinalAnswer(answer: FinalAnswer): string {
    if (answer.type === 'FINAL') {
      return answer.value;
    }
    // FINAL_VAR - get from REPL locals
    const value = this.repl.getLocal(answer.variable);
    if (value === undefined) {
      return `Error: Variable '${answer.variable}' not found in REPL`;
    }
    return String(value);
  }
}
```

---

### Phase 4: Integration & UI (Week 4-5)

**Goal**: Integrate RLM into OpenCode's existing architecture and update UI.

#### 4.1 Session Integration

**Files to Modify:**
- `packages/opencode/src/session/llm.ts`
- `packages/opencode/src/session/message-v2.ts`

**Integration Approach:**
```typescript
// In llm.ts, modify the stream() function
export async function stream(input: StreamInput) {
  // Check if RLM mode is enabled
  const rlmMode = input.rlmMode ?? cfg.experimental?.rlm_mode ?? Flag.OPENCODE_EXPERIMENTAL_RLM;

  if (rlmMode && shouldUseRLM(input)) {
    // Use RLM orchestrator instead of direct LLM call
    const orchestrator = new RLMOrchestrator({
      maxIterations: 20,
      maxDepth: 1,
      environment: 'local',
      subModel: getSubModel(input.model),
    });

    return orchestrator.completion(input.query, input.context);
  }

  // Existing non-RLM flow
  return streamText({ ... });
}

function shouldUseRLM(input: StreamInput): boolean {
  // Heuristics for when to use RLM
  const contextLength = estimateTokens(input.messages);
  return contextLength > 50000; // Use RLM for large contexts
}
```

#### 4.2 UI Updates

**Files to Modify:**
- `packages/opencode/src/cli/cmd/tui/routes/session/index.tsx`
- `packages/opencode/src/cli/cmd/tui/routes/session/footer.tsx`

**New UI Components Needed:**
1. **RLM Execution Panel** - Show REPL execution in real-time
2. **Iteration Counter** - Display current iteration number
3. **Sub-call Indicator** - Show when llm_query is being called

---

### Phase 5: Advanced Features (Week 5-6)

**Goal**: Implement batched queries, Docker environment, and visualization.

#### 5.1 Batched Queries

**Reference**: Paper mentions `llm_query_batched` for concurrent processing

```typescript
// In local-repl.ts
async llmQueryBatched(prompts: string[]): Promise<string[]> {
  return Promise.all(prompts.map(p => this.llmQuery(p)));
}
```

#### 5.2 Docker Environment (Optional)

**Reference**: `github.com/alexzhang13/rlm/blob/main/rlm/environments/docker_repl.py`

**Implementation:**
- Use `dockerode` npm package
- Pull `python:3.11-slim` image
- Mount context as volume
- Execute via HTTP proxy

#### 5.3 Trajectory Logging

**Reference**: `github.com/alexzhang13/rlm/blob/main/rlm/logger/`

```typescript
export interface RLMIteration {
  iterationNumber: number;
  prompt: string;
  response: string;
  codeBlocks: string[];
  replOutputs: string[];
  subLMCalls: number;
  executionTime: number;
}

export class RLMLogger {
  private logDir: string;

  logIteration(iteration: RLMIteration): void {
    // Write to JSONL file
  }
}
```

---

## 7. Detailed Component Specifications

### 7.1 LocalREPL Implementation Using isolated-vm

```typescript
// packages/opencode/src/rlm/environments/local-repl.ts
import ivm from 'isolated-vm';

export class LocalREPL extends BaseEnvironment {
  private isolate: ivm.Isolate;
  private context: ivm.Context;
  private lmHandlerAddress: string;

  async setup(): Promise<void> {
    this.isolate = new ivm.Isolate({ memoryLimit: 128 });
    this.context = await this.isolate.createContext();

    // Setup safe globals
    const jail = this.context.global;
    await jail.set('global', jail.derefInto());

    // Add console.log -> capture stdout
    await this.setupConsole();

    // Add llm_query function (callback to main thread)
    await this.setupLLMQuery();
  }

  private async setupLLMQuery(): Promise<void> {
    const jail = this.context.global;

    // Create a reference that the isolate can call back to
    const llmQueryRef = new ivm.Reference(async (prompt: string) => {
      // Make HTTP/socket call to LM Handler
      const response = await fetch(`http://${this.lmHandlerAddress}/query`, {
        method: 'POST',
        body: JSON.stringify({ prompt }),
      });
      return (await response.json()).content;
    });

    await jail.set('_llm_query_ref', llmQueryRef);

    // Wrap in async function
    await this.context.eval(`
      async function llm_query(prompt) {
        return await _llm_query_ref.apply(undefined, [prompt], { result: { promise: true }});
      }
    `);
  }

  async executeCode(code: string): Promise<REPLResult> {
    const startTime = Date.now();

    try {
      // Wrap code to capture output
      const wrappedCode = `
        (async () => {
          ${code}
        })()
      `;

      const result = await this.context.eval(wrappedCode, {
        timeout: 30000,
        promise: true
      });

      return {
        stdout: this.capturedOutput,
        stderr: '',
        locals: await this.getLocals(),
        executionTime: Date.now() - startTime,
        rlmCalls: this.rlmCallCount,
      };
    } catch (error) {
      return {
        stdout: this.capturedOutput,
        stderr: error.message,
        locals: {},
        executionTime: Date.now() - startTime,
        rlmCalls: this.rlmCallCount,
      };
    }
  }

  loadContext(context: string | Record<string, any>): void {
    // Set context variable in isolate
    const contextValue = typeof context === 'string' ? context : JSON.stringify(context);
    this.context.evalSync(`var context = ${JSON.stringify(contextValue)};`);
  }

  cleanup(): void {
    this.context.release();
    this.isolate.dispose();
  }
}
```

### 7.2 Output Truncation Strategy

**From Paper (Page 24, Appendix D):**
> "You will only be able to see truncated outputs from the REPL environment"

**Default truncation: 8192 characters**

```typescript
export function formatExecutionResult(result: REPLResult, maxChars = 8192): string {
  let output = '';

  if (result.stdout) {
    output += `stdout:\n${result.stdout}\n`;
  }

  if (result.stderr) {
    output += `stderr:\n${result.stderr}\n`;
  }

  // Add notable variables (excluding internal ones)
  const notableVars = Object.entries(result.locals)
    .filter(([key]) => !key.startsWith('_'))
    .slice(0, 10);

  if (notableVars.length > 0) {
    output += `\nVariables:\n`;
    for (const [key, value] of notableVars) {
      const valueStr = String(value).slice(0, 200);
      output += `  ${key} = ${valueStr}\n`;
    }
  }

  // Truncate if necessary
  if (output.length > maxChars) {
    const truncated = output.length - maxChars;
    output = output.slice(0, maxChars) + `\n... [truncated ${truncated} chars]`;
  }

  return output;
}
```

### 7.3 Context Metadata Computation

**From Paper (Page 24):**
> "Your context is a {context_type} with {context_total_length} total characters, and is broken up into chunks of char lengths: {context_lengths}."

```typescript
export function computeQueryMetadata(context: string | any[] | Record<string, any>): QueryMetadata {
  if (typeof context === 'string') {
    return {
      contextType: 'string',
      contextTotalLength: context.length,
      contextLengths: [context.length],
    };
  }

  if (Array.isArray(context)) {
    const lengths = context.map(item =>
      typeof item === 'string' ? item.length : JSON.stringify(item).length
    );
    return {
      contextType: `List[str] with ${context.length} items`,
      contextTotalLength: lengths.reduce((a, b) => a + b, 0),
      contextLengths: lengths.length > 100
        ? [...lengths.slice(0, 100), `... [${lengths.length - 100} others]`]
        : lengths,
    };
  }

  // Dict/object
  const serialized = JSON.stringify(context);
  return {
    contextType: 'dict',
    contextTotalLength: serialized.length,
    contextLengths: [serialized.length],
  };
}
```

---

## 8. Integration with OpenCode

### 8.1 File Structure

```
packages/opencode/src/rlm/
├── index.ts                    # Main exports
├── orchestrator.ts             # RLMOrchestrator class
├── types.ts                    # Type definitions
├── config.ts                   # RLM configuration
│
├── environments/
│   ├── index.ts                # Environment exports
│   ├── base.ts                 # BaseEnvironment abstract class
│   ├── local-repl.ts           # LocalREPL (isolated-vm)
│   └── docker-repl.ts          # DockerREPL (optional)
│
├── lm-handler/
│   ├── index.ts                # LM Handler exports
│   ├── handler.ts              # LMHandler class
│   └── protocol.ts             # Communication protocol
│
├── prompts/
│   ├── index.ts                # Prompt exports
│   ├── system.ts               # RLM_SYSTEM_PROMPT
│   └── user.ts                 # USER_PROMPT templates
│
├── parsing/
│   ├── index.ts                # Parsing exports
│   ├── code-blocks.ts          # findCodeBlocks()
│   └── final-answer.ts         # findFinalAnswer()
│
└── logger/
    ├── index.ts                # Logger exports
    └── trajectory.ts           # RLMLogger class
```

### 8.2 Configuration Schema

**Add to `src/config/config.ts`:**

```typescript
// In the experimental section
rlm: z.object({
  enabled: z.boolean().optional().default(false),
  maxIterations: z.number().optional().default(20),
  maxDepth: z.number().optional().default(1),
  outputTruncation: z.number().optional().default(8192),
  environment: z.enum(['local', 'docker']).optional().default('local'),
  subModel: z.string().optional(), // Auto-detect if not specified
  verbose: z.boolean().optional().default(false),
}).optional().describe("RLM (Recursive Language Model) configuration"),
```

### 8.3 Flag Updates

**Modify `src/flag/flag.ts`:**

```typescript
// Already exists:
export const OPENCODE_EXPERIMENTAL_RLM = OPENCODE_EXPERIMENTAL || truthy("OPENCODE_EXPERIMENTAL_RLM")

// Add:
export const OPENCODE_RLM_MAX_ITERATIONS = number("OPENCODE_RLM_MAX_ITERATIONS") || 20
export const OPENCODE_RLM_OUTPUT_TRUNCATION = number("OPENCODE_RLM_OUTPUT_TRUNCATION") || 8192
export const OPENCODE_RLM_ENVIRONMENT = process.env["OPENCODE_RLM_ENVIRONMENT"] || "local"
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

**Test Files to Create:**
- `packages/opencode/src/rlm/__tests__/parsing.test.ts`
- `packages/opencode/src/rlm/__tests__/local-repl.test.ts`
- `packages/opencode/src/rlm/__tests__/orchestrator.test.ts`

**Parsing Tests:**
```typescript
describe('findCodeBlocks', () => {
  it('extracts single repl block', () => {
    const response = 'Some text\n```repl\nprint("hello")\n```\nMore text';
    expect(findCodeBlocks(response)).toEqual(['print("hello")']);
  });

  it('extracts multiple repl blocks', () => {
    const response = '```repl\nx = 1\n```\n```repl\ny = 2\n```';
    expect(findCodeBlocks(response)).toEqual(['x = 1', 'y = 2']);
  });

  it('ignores other code blocks', () => {
    const response = '```python\nprint("hello")\n```';
    expect(findCodeBlocks(response)).toEqual([]);
  });
});

describe('findFinalAnswer', () => {
  it('detects FINAL()', () => {
    const response = 'The answer is FINAL(42)';
    expect(findFinalAnswer(response)).toEqual({ type: 'FINAL', value: '42' });
  });

  it('detects FINAL_VAR()', () => {
    const response = 'Returning FINAL_VAR(result)';
    expect(findFinalAnswer(response)).toEqual({ type: 'FINAL_VAR', variable: 'result' });
  });

  it('prefers FINAL_VAR over FINAL', () => {
    const response = 'FINAL_VAR(x) or FINAL(y)';
    expect(findFinalAnswer(response)).toEqual({ type: 'FINAL_VAR', variable: 'x' });
  });
});
```

### 9.2 Integration Tests

**Test with real models using the paper's benchmarks:**

1. **S-NIAH (Simple)** - Find a needle in haystack
2. **OOLONG** - Aggregate semantic transformations
3. **Code Understanding** - Answer questions about code

### 9.3 Benchmark Tests

**Reference**: Paper Section 2.1 (Pages 3-4)

Create test cases from each benchmark:
- S-NIAH: 50 needle-in-haystack tasks
- OOLONG: Semantic aggregation over trec_coarse split
- OOLONG-Pairs: Pairwise aggregation (quadratic complexity)

---

## 10. Configuration & User Interface

### 10.1 CLI Configuration

**Example `opencode.json`:**
```json
{
  "experimental": {
    "rlm": {
      "enabled": true,
      "maxIterations": 20,
      "maxDepth": 1,
      "outputTruncation": 8192,
      "environment": "local",
      "subModel": "claude-3-5-haiku-latest",
      "verbose": false
    }
  }
}
```

### 10.2 Environment Variables

```bash
# Enable RLM
export OPENCODE_EXPERIMENTAL_RLM=true

# Configure RLM
export OPENCODE_RLM_MAX_ITERATIONS=20
export OPENCODE_RLM_OUTPUT_TRUNCATION=8192
export OPENCODE_RLM_ENVIRONMENT=local
```

### 10.3 UI Indicators

**Footer Enhancement:**
```
◈ RLM [3/20] | ⊙ 2 sub-calls
```

**Session View:**
- Show REPL execution blocks in collapsible sections
- Display iteration progress
- Show sub-model calls with cost

---

## Appendix A: Complete System Prompt (from Paper)

See Paper Appendix D.1 (Pages 24-25) for the full system prompt.

The prompt is ~3500 characters and includes:
1. Environment description
2. Available functions (`context`, `llm_query`, `llm_query_batched`, `print`)
3. Code block format (`\`\`\`repl`)
4. Multiple examples (chunking, iteration, regex)
5. FINAL/FINAL_VAR instructions
6. Guidance on batching and efficiency

## Appendix B: Qwen3-Coder Warning

**From Paper Appendix D.1 (Page 25):**

For models like Qwen that are aggressive with sub-calls, add this warning:

> "IMPORTANT: Be very careful about using 'llm_query' as it incurs high runtime costs. Always batch as much information as reasonably possible into each call (aim for around ~200k characters per call). For example, if you have 1000 lines of information to process, it's much better to split into chunks of 5 and call 'llm_query' on each chunk (200 calls total) rather than making 1000 individual calls. Minimize the number of 'llm_query' calls by batching related information together."

## Appendix C: Implementation Checklist

- [ ] Phase 1: Core Infrastructure
  - [ ] BaseEnvironment abstract class
  - [ ] LocalREPL with isolated-vm
  - [ ] findCodeBlocks() parser
  - [ ] findFinalAnswer() parser
  - [ ] RLM_SYSTEM_PROMPT constant
  - [ ] buildSystemPrompt() function
  - [ ] buildUserPrompt() function

- [ ] Phase 2: LM Handler
  - [ ] LMHandler class
  - [ ] Socket/HTTP protocol
  - [ ] Sub-model client registration
  - [ ] Batched query support

- [ ] Phase 3: Orchestrator
  - [ ] RLMOrchestrator class
  - [ ] Main completion loop
  - [ ] Output truncation
  - [ ] FINAL/FINAL_VAR resolution
  - [ ] Default answer generation

- [ ] Phase 4: Integration
  - [ ] Modify llm.ts for RLM routing
  - [ ] Update UI components
  - [ ] Add configuration schema
  - [ ] Update flags

- [ ] Phase 5: Advanced
  - [ ] Docker environment (optional)
  - [ ] Trajectory logging
  - [ ] Visualization

---

## References

1. **Paper**: Zhang, A. L., Kraska, T., & Khattab, O. (2025). Recursive Language Models. arXiv:2512.24601. [Local: `docs/rlm/rlm-paper.pdf`]

2. **Official Implementation**: https://github.com/alexzhang13/rlm

3. **Key Files**:
   - System Prompt: `rlm/utils/prompts.py`
   - Parsing: `rlm/utils/parsing.py`
   - LocalREPL: `rlm/environments/local_repl.py`
   - RLM Core: `rlm/core/rlm.py`
   - LM Handler: `rlm/core/lm_handler.py`
