# RLM (Recursive Language Model) Implementation Plan for OpenCode

## Document Information

- **Version**: 2.0 (Staff Engineer Review)
- **Date**: 2026-01-04
- **Reviewer**: Staff Software Engineer Critical Review
- **Paper Reference**: `docs/rlm/rlm-paper.pdf` (arXiv:2512.24601)
- **Official Implementation**: https://github.com/alexzhang13/rlm

---

## CRITICAL REVIEW SUMMARY

### Review Status: ⚠️ SIGNIFICANT GAPS IDENTIFIED

This document has been critically reviewed against the RLM paper (arXiv:2512.24601), the official GitHub implementation, and the OpenCode codebase. While the implementation plan is comprehensive in scope, several **critical gaps** must be addressed before implementation.

### Key Findings

| Category | Status | Severity |
|----------|--------|----------|
| Current OpenCode RLM Code | ❌ **Fundamentally Wrong** | CRITICAL |
| System Prompt Completeness | ⚠️ **Partial** | HIGH |
| REPL Environment Choice | ⚠️ **Needs Revision** | HIGH |
| User Prompt Templates | ❌ **Missing** | HIGH |
| Batched Query Implementation | ⚠️ **Oversimplified** | MEDIUM |
| Error Handling & Edge Cases | ❌ **Missing** | HIGH |
| UI/UX Streaming Integration | ⚠️ **Incomplete** | MEDIUM |
| Abort/Cancellation Handling | ❌ **Missing** | MEDIUM |
| Model-Specific Adaptations | ❌ **Missing** | HIGH |
| Cost/Budget Management | ❌ **Missing** | MEDIUM |

---

## Table of Contents

0. [Critical Review Findings](#0-critical-review-findings)
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
11. [Edge Cases & Error Handling](#11-edge-cases--error-handling)
12. [Model-Specific Adaptations](#12-model-specific-adaptations)
13. [Cost Management & Budgeting](#13-cost-management--budgeting)

---

## 0. Critical Review Findings

### 0.1 CRITICAL: Current OpenCode RLM Implementation is Fundamentally Wrong

**Current State Analysis:**

The existing RLM implementation in OpenCode consists of:

| File | Current Content | Paper Requirement | Status |
|------|-----------------|-------------------|--------|
| `src/session/prompt/rlm.txt` | 32-line "think harder" prompt with iterative refinement guidance | Full RLM system prompt with REPL instructions, `llm_query()` docs, `FINAL()`/`FINAL_VAR()` | ❌ **WRONG** |
| `src/provider/transform.ts` | Sets placebo options: `rlm.enabled`, `recursiveProcessing`, `maxDepth: 3` | These options do nothing without actual REPL environment | ❌ **PLACEBO** |
| `src/session/llm.ts` | Appends RLM prompt to system prompts, enables thinking modes | Should route to RLM orchestrator, not just append prompt | ❌ **INCOMPLETE** |

**Why the Current Implementation Cannot Work:**

1. **No REPL Environment**: The paper's core innovation is treating context as a variable in a Python REPL. Without this, there's nothing for the LLM to interact with programmatically.

2. **No `llm_query()` Function**: The recursive sub-LM calling mechanism doesn't exist. The current code just enables "thinking" modes which is completely different.

3. **No Code Execution**: The LLM generates `\`\`\`repl` blocks but nothing parses or executes them.

4. **No Iteration Loop**: RLM requires iterative prompting until `FINAL()` is called. Current implementation is single-shot.

**Current `rlm.txt` Content (WRONG):**
```
# RLM (Recursive Language Model) Mode
You are operating in RLM mode, which enables enhanced recursive reasoning...
1. **Decompose Complex Problems**: Break down large, complex prompts...
2. **Iterative Refinement**: For complex reasoning tasks, work through...
```

This is a prompt engineering approach, NOT an RLM implementation.

---

### 0.2 HIGH: REPL Environment Choice Must Be Python-First

**Issue with Current Plan:**

The plan recommends `isolated-vm` (JavaScript) as the primary option with Pyodide as secondary. This is backwards.

**Why Python is Required:**

1. **Paper Uses Python**: All system prompts, examples, and trajectories use Python syntax
2. **Model Training**: LLMs are trained/fine-tuned on Python code execution patterns
3. **llm_query() Signature**: `llm_query(prompt: str) -> str` is Python
4. **Examples in Prompts**: All chunking, regex, iteration examples are Python

**Corrected Recommendation:**

| Priority | Technology | Use Case |
|----------|------------|----------|
| **PRIMARY** | **Pyodide (Python in WASM)** | Full Python compatibility, works in browser & Node |
| Secondary | Docker + Python 3.11 | Production isolation, network capabilities |
| Fallback | `isolated-vm` with Python-like API | Performance-critical scenarios |

**Pyodide Initialization (Corrected):**
```typescript
// packages/opencode/src/rlm/environments/pyodide-repl.ts
import { loadPyodide, PyodideInterface } from 'pyodide';

export class PyodideREPL extends BaseEnvironment {
  private pyodide: PyodideInterface | null = null;
  private lmHandlerAddress: string = '';

  async setup(): Promise<void> {
    // Load Pyodide (one-time, ~10MB download, cached thereafter)
    this.pyodide = await loadPyodide({
      indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.25.0/full/',
    });

    // Install llm_query as a global function
    this.pyodide.registerJsModule('_opencode_rlm', {
      llm_query: this.llmQueryBridge.bind(this),
      llm_query_batched: this.llmQueryBatchedBridge.bind(this),
    });

    // Setup the Python environment
    await this.pyodide.runPythonAsync(`
from _opencode_rlm import llm_query, llm_query_batched
import re
import json

# Safe globals - exclude dangerous builtins
_BLOCKED = {'eval', 'exec', 'compile', '__import__', 'open', 'input'}
    `);
  }

  loadContext(context: string | string[] | Record<string, any>): void {
    const serialized = typeof context === 'string'
      ? JSON.stringify(context)
      : JSON.stringify(context);

    this.pyodide!.runPython(`
context = json.loads('''${serialized.replace(/'/g, "\\'")}''')
    `);
  }

  async executeCode(code: string): Promise<REPLResult> {
    const startTime = Date.now();
    let stdout = '';
    let stderr = '';

    // Capture stdout
    this.pyodide!.setStdout({
      batched: (text: string) => { stdout += text; }
    });

    try {
      await this.pyodide!.runPythonAsync(code);
      const locals = this.pyodide!.globals.toJs();

      return {
        stdout,
        stderr: '',
        locals: this.serializeLocals(locals),
        executionTime: Date.now() - startTime,
        rlmCalls: this.rlmCallCount,
      };
    } catch (error: any) {
      return {
        stdout,
        stderr: error.message || String(error),
        locals: {},
        executionTime: Date.now() - startTime,
        rlmCalls: this.rlmCallCount,
      };
    }
  }
}
```

---

### 0.3 HIGH: Missing User Prompt Templates for Iterations

**Issue:**

The plan includes the system prompt but omits the **user prompt templates** that vary by iteration number.

**From Official Implementation (`rlm/utils/prompts.py`):**

```python
# Iteration 0 (First turn)
USER_PROMPT_INITIAL = """You have not interacted with the REPL environment or seen your prompt / context yet.
Think step-by-step on what to do using the REPL environment (which contains the context) to answer the original prompt: "{root_prompt}"

Remember: Do NOT provide a final answer until you have sufficiently explored the context in the REPL environment."""

# Iteration > 0 (Subsequent turns)
USER_PROMPT_CONTINUE = """The history before is your previous interactions with the REPL environment.
Continue thinking step-by-step to answer the original prompt: "{root_prompt}"

If you have gathered enough information, provide your final answer using FINAL() or FINAL_VAR()."""
```

**Required Addition to Implementation:**

```typescript
// packages/opencode/src/rlm/prompts/user.ts

export function buildUserPrompt(
  rootPrompt: string,
  iteration: number,
  hasCodeExecuted: boolean
): string {
  if (iteration === 0) {
    return `You have not interacted with the REPL environment or seen your prompt / context yet.
Think step-by-step on what to do using the REPL environment (which contains the context) to answer the original prompt: "${rootPrompt}"

Remember: Do NOT provide a final answer until you have sufficiently explored the context in the REPL environment.`;
  }

  if (!hasCodeExecuted) {
    return `You have not executed any code yet. Use the REPL environment to explore the context variable.
Original prompt: "${rootPrompt}"`;
  }

  return `The history before is your previous interactions with the REPL environment.
Continue thinking step-by-step to answer the original prompt: "${rootPrompt}"

If you have gathered enough information, provide your final answer using FINAL() or FINAL_VAR().`;
}
```

---

### 0.4 HIGH: Model-Specific Prompt Adaptations Missing

**Issue:**

The paper explicitly states (Appendix A): "Using the exact same RLM system prompt across all models can be problematic."

**Required Model-Specific Adaptations:**

| Model Family | Required Adaptation | Reason |
|--------------|---------------------|--------|
| **Qwen3-Coder** | Add sub-call cost warning | Makes hundreds of unnecessary calls without warning |
| **Claude** | Adjust FINAL() detection | May include FINAL in reasoning before actual answer |
| **GPT-5** | Standard prompt works | Reference implementation |
| **Gemini** | TBD - needs testing | May require output format adjustments |

**Qwen3-Coder Specific Addition (FROM PAPER APPENDIX D.1):**

```typescript
const QWEN_SUB_CALL_WARNING = `
IMPORTANT: Be very careful about using 'llm_query' as it incurs high runtime costs.
Always batch as much information as reasonably possible into each call (aim for around ~200k characters per call).
For example, if you have 1000 lines of information to process, it's much better to split into chunks of 5 and
call 'llm_query' on each chunk (200 calls total) rather than making 1000 individual calls.
Minimize the number of 'llm_query' calls by batching related information together.
`;

export function buildSystemPrompt(
  model: Provider.Model,
  metadata: QueryMetadata
): Array<{role: string, content: string}> {
  let prompt = RLM_SYSTEM_PROMPT;

  // Add model-specific warnings
  if (model.id.toLowerCase().includes('qwen')) {
    prompt = QWEN_SUB_CALL_WARNING + '\n\n' + prompt;
  }

  // ... rest of implementation
}
```

---

### 0.5 HIGH: Error Handling & Edge Cases Missing

**From Paper Appendix A - Failure Modes:**

1. **Output Token Exhaustion**: Models run out of tokens during thinking
2. **FINAL Tag Disambiguation**: Models output plans as final answers
3. **Trajectory Instability**: Excessive sub-calls without safeguards
4. **Code Execution Errors**: Syntax errors, runtime errors, timeouts

**Required Error Handling Implementation:**

```typescript
// packages/opencode/src/rlm/orchestrator.ts

interface RLMSafetyLimits {
  maxSubCalls: number;           // Default: 100 (prevent runaway costs)
  maxCodeExecutionTime: number;  // Default: 30000ms per block
  maxTotalTime: number;          // Default: 300000ms (5 min)
  maxOutputLength: number;       // Default: 1000000 chars
  maxIterationsWithoutCode: number; // Default: 3 (force exploration)
}

const DEFAULT_LIMITS: RLMSafetyLimits = {
  maxSubCalls: 100,
  maxCodeExecutionTime: 30000,
  maxTotalTime: 300000,
  maxOutputLength: 1000000,
  maxIterationsWithoutCode: 3,
};

export class RLMOrchestrator {
  private subCallCount = 0;
  private iterationsWithoutCode = 0;
  private startTime = 0;

  private checkSafetyLimits(): { ok: boolean; reason?: string } {
    // Check sub-call budget
    if (this.subCallCount >= this.config.limits.maxSubCalls) {
      return {
        ok: false,
        reason: `Sub-call limit reached (${this.config.limits.maxSubCalls}). Forcing final answer.`
      };
    }

    // Check total time
    const elapsed = Date.now() - this.startTime;
    if (elapsed >= this.config.limits.maxTotalTime) {
      return {
        ok: false,
        reason: `Time limit reached (${elapsed}ms). Forcing final answer.`
      };
    }

    // Check for stuck model (no code execution)
    if (this.iterationsWithoutCode >= this.config.limits.maxIterationsWithoutCode) {
      return {
        ok: false,
        reason: 'Model not using REPL environment. Forcing final answer.'
      };
    }

    return { ok: true };
  }

  private handleCodeExecutionError(error: Error, code: string): string {
    // Provide helpful error message back to model
    const errorTypes = {
      SyntaxError: 'Your code has a syntax error. Please fix and retry.',
      NameError: 'You referenced an undefined variable. Check variable names.',
      TypeError: 'Type mismatch in your code. Check data types.',
      TimeoutError: 'Code execution timed out. Simplify your approach.',
      RecursionError: 'Too much recursion. Use iterative approach instead.',
    };

    const errorType = error.constructor.name;
    const helpText = errorTypes[errorType as keyof typeof errorTypes] ||
                     'An error occurred during code execution.';

    return `REPL Error:\n${error.message}\n\nHint: ${helpText}\n\nYour code:\n\`\`\`python\n${code}\n\`\`\``;
  }
}
```

---

### 0.6 MEDIUM: Streaming Integration Incomplete

**Issue:**

OpenCode uses streaming for LLM responses. The plan doesn't specify how RLM mode integrates with this.

**Required Streaming Architecture:**

```typescript
// RLM streaming events
type RLMStreamEvent =
  | { type: 'iteration-start'; iteration: number; maxIterations: number }
  | { type: 'llm-response-delta'; delta: string }
  | { type: 'code-detected'; code: string }
  | { type: 'code-executing'; code: string }
  | { type: 'code-result'; stdout: string; stderr: string; executionTime: number }
  | { type: 'sub-call-start'; prompt: string; callNumber: number }
  | { type: 'sub-call-complete'; response: string; callNumber: number }
  | { type: 'final-detected'; type: 'FINAL' | 'FINAL_VAR'; value: string }
  | { type: 'iteration-complete'; iteration: number }
  | { type: 'complete'; response: string; totalIterations: number; totalSubCalls: number };

// Integration with existing OpenCode streaming
export async function* streamRLM(input: RLMInput): AsyncGenerator<RLMStreamEvent> {
  const orchestrator = new RLMOrchestrator(input.config);

  for await (const event of orchestrator.execute(input)) {
    yield event;

    // Allow UI to update
    if (event.type === 'iteration-complete') {
      // Bus.publish for UI updates
      await Bus.publish(Session.Event.RLMIteration, {
        sessionID: input.sessionID,
        iteration: event.iteration,
      });
    }
  }
}
```

---

### 0.7 MEDIUM: Abort/Cancellation Handling Missing

**Issue:**

OpenCode supports aborting operations via `AbortSignal`. RLM must properly handle:
- Aborting mid-iteration
- Aborting during sub-LM calls
- Cleaning up REPL state on abort

**Required Implementation:**

```typescript
export class RLMOrchestrator {
  private abortController: AbortController | null = null;

  async completion(
    query: string,
    context: string,
    abort: AbortSignal
  ): Promise<RLMCompletion> {
    // Link to parent abort signal
    this.abortController = new AbortController();
    abort.addEventListener('abort', () => {
      this.abortController?.abort();
    });

    try {
      // ... main loop
      for (let i = 0; i < this.config.maxIterations; i++) {
        // Check abort before each iteration
        if (this.abortController.signal.aborted) {
          return this.createAbortedResponse('User cancelled');
        }

        // Pass abort to sub-calls
        const response = await this.rootClient.completion(messages, {
          signal: this.abortController.signal,
        });

        // ... rest of iteration
      }
    } finally {
      // Always cleanup
      await this.cleanup();
    }
  }

  private async cleanup(): Promise<void> {
    // Stop LM handler
    await this.lmHandler?.stop();

    // Cleanup REPL environment
    await this.repl?.cleanup();

    // Clear state
    this.subCallCount = 0;
    this.iterationsWithoutCode = 0;
  }
}
```

---

### 0.8 MEDIUM: `llm_query_batched` Implementation Oversimplified

**Issue:**

The plan shows:
```typescript
async llmQueryBatched(prompts: string[]): Promise<string[]> {
  return Promise.all(prompts.map(p => this.llmQuery(p)));
}
```

This doesn't handle:
- Rate limiting
- Error isolation (one failure shouldn't fail all)
- Progress reporting
- Memory constraints

**Corrected Implementation:**

```typescript
async llmQueryBatched(
  prompts: string[],
  options: { concurrency?: number; stopOnError?: boolean } = {}
): Promise<string[]> {
  const { concurrency = 5, stopOnError = false } = options;
  const results: string[] = new Array(prompts.length);
  const errors: Array<{ index: number; error: Error }> = [];

  // Process in batches for memory efficiency
  for (let i = 0; i < prompts.length; i += concurrency) {
    const batch = prompts.slice(i, i + concurrency);
    const batchPromises = batch.map(async (prompt, batchIndex) => {
      const index = i + batchIndex;
      try {
        results[index] = await this.llmQuery(prompt);
      } catch (error) {
        if (stopOnError) throw error;
        errors.push({ index, error: error as Error });
        results[index] = `[ERROR: ${(error as Error).message}]`;
      }
    });

    await Promise.all(batchPromises);

    // Check abort between batches
    if (this.abortSignal?.aborted) {
      throw new Error('Batched query aborted');
    }
  }

  if (errors.length > 0) {
    console.warn(`llm_query_batched: ${errors.length}/${prompts.length} queries failed`);
  }

  return results;
}
```

---

### 0.9 Corrected Code Block Parsing (Regex Fix)

**Issue:**

The plan's regex doesn't match the official implementation exactly.

**Official Pattern (from `rlm/utils/parsing.py`):**
```python
pattern = r"```repl\s*\n(.*?)\n```"
```

**Corrected TypeScript:**
```typescript
export function findCodeBlocks(response: string): string[] {
  // Match ```repl followed by optional whitespace, newline, content, newline, ```
  // DOTALL mode: ([\s\S]*?) to match across lines
  const pattern = /```repl\s*\n([\s\S]*?)\n```/g;
  const matches: string[] = [];
  let match;

  while ((match = pattern.exec(response)) !== null) {
    const code = match[1].trim();
    if (code) {
      matches.push(code);
    }
  }

  return matches;
}

export function findFinalAnswer(response: string): FinalAnswer | null {
  // FINAL_VAR must be checked first (more specific)
  // Pattern must match at start of line (^ with multiline)
  const varMatch = response.match(/^\s*FINAL_VAR\(([^)]+)\)/m);
  if (varMatch) {
    // Strip quotes from variable name
    const varName = varMatch[1].trim().replace(/^["']|["']$/g, '');
    return { type: 'FINAL_VAR', variable: varName };
  }

  // FINAL can contain multi-line content
  const finalMatch = response.match(/^\s*FINAL\(([\s\S]*?)\)/m);
  if (finalMatch) {
    return { type: 'FINAL', value: finalMatch[1].trim() };
  }

  return null;
}
```

---

### 0.10 Updated Implementation Checklist

Replace Appendix C checklist with:

- [ ] **Phase 0: Remove/Fix Existing Wrong Code**
  - [ ] Delete or comment out current `rlm.txt` "think harder" prompt
  - [ ] Remove placebo `rlm` options from `transform.ts`
  - [ ] Update `llm.ts` to route to RLM orchestrator (not just append prompt)

- [ ] **Phase 1: Core Infrastructure**
  - [ ] Pyodide REPL environment (PRIMARY)
  - [ ] `findCodeBlocks()` parser with corrected regex
  - [ ] `findFinalAnswer()` parser with FINAL_VAR priority
  - [ ] Full system prompt from paper Appendix D
  - [ ] User prompt templates for iteration 0 vs N
  - [ ] Context metadata computation

- [ ] **Phase 2: LM Handler & Sub-Model**
  - [ ] LMHandler class with HTTP/WebSocket protocol
  - [ ] Sub-model client registration
  - [ ] `llm_query()` implementation with error handling
  - [ ] `llm_query_batched()` with concurrency control
  - [ ] Model-specific sub-model mappings

- [ ] **Phase 3: Orchestrator**
  - [ ] Main completion loop with iteration control
  - [ ] Safety limits (sub-calls, time, iterations)
  - [ ] Output truncation (8192 chars default)
  - [ ] FINAL/FINAL_VAR resolution with variable lookup
  - [ ] Default answer generation on max iterations
  - [ ] Abort signal handling

- [ ] **Phase 4: Integration**
  - [ ] Streaming event system for UI
  - [ ] Session integration in `llm.ts`
  - [ ] Bus events for real-time updates
  - [ ] Configuration schema updates

- [ ] **Phase 5: Model Adaptations**
  - [ ] Qwen sub-call warning integration
  - [ ] Claude FINAL detection adjustments
  - [ ] GPT-5/GPT-5-mini sub-model pairing
  - [ ] Testing matrix for all supported providers

- [ ] **Phase 6: Safety & Observability**
  - [ ] Cost tracking per iteration and sub-call
  - [ ] Trajectory logging (JSONL format)
  - [ ] Error categorization and recovery
  - [ ] Memory monitoring for REPL

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

---

## 11. Edge Cases & Error Handling

This section provides comprehensive coverage of edge cases and error handling strategies based on the RLM paper's failure modes (Appendix A) and practical implementation concerns.

### 11.1 REPL Execution Edge Cases

#### 11.1.1 Infinite Loops / Long-Running Code

**Problem**: Model-generated code may contain infinite loops or computationally expensive operations.

**Solution**:
```typescript
// packages/opencode/src/rlm/environments/execution-guard.ts

export class ExecutionGuard {
  private executionTimeout: number;
  private memoryLimit: number;

  async executeWithGuards(
    repl: BaseEnvironment,
    code: string,
    config: ExecutionConfig
  ): Promise<REPLResult> {
    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => {
        reject(new ExecutionTimeoutError(
          `Code execution exceeded ${config.timeout}ms limit`
        ));
      }, config.timeout);
    });

    const executionPromise = repl.executeCode(code);

    try {
      return await Promise.race([executionPromise, timeoutPromise]);
    } catch (error) {
      if (error instanceof ExecutionTimeoutError) {
        // Force cleanup of runaway execution
        await repl.forceTerminate();
        return {
          stdout: '',
          stderr: `Execution timeout: Your code took longer than ${config.timeout}ms. ` +
                  `Consider breaking it into smaller operations.`,
          locals: {},
          executionTime: config.timeout,
          rlmCalls: 0,
        };
      }
      throw error;
    }
  }
}
```

#### 11.1.2 Memory Exhaustion

**Problem**: Code may allocate excessive memory (e.g., `range(10**12)`).

**Solution**:
```typescript
// Memory monitoring for Pyodide
export class PyodideMemoryMonitor {
  private wasmMemory: WebAssembly.Memory;
  private maxBytes: number;

  checkMemoryUsage(): { used: number; limit: number; ok: boolean } {
    const used = this.wasmMemory.buffer.byteLength;
    return {
      used,
      limit: this.maxBytes,
      ok: used < this.maxBytes * 0.9, // Warn at 90%
    };
  }

  getMemoryWarning(): string | null {
    const { used, limit, ok } = this.checkMemoryUsage();
    if (!ok) {
      const usedMB = Math.round(used / 1024 / 1024);
      const limitMB = Math.round(limit / 1024 / 1024);
      return `Memory warning: Using ${usedMB}MB of ${limitMB}MB limit. ` +
             `Consider freeing variables with 'del variable_name'.`;
    }
    return null;
  }
}
```

#### 11.1.3 Syntax Errors in Generated Code

**Problem**: LLM generates syntactically invalid Python code.

**Solution**:
```typescript
export function preprocessCode(code: string): { valid: boolean; error?: string; cleaned?: string } {
  // Common fixes for LLM-generated code
  let cleaned = code;

  // Fix 1: Remove markdown artifacts
  cleaned = cleaned.replace(/^```\w*\n?/gm, '');
  cleaned = cleaned.replace(/\n?```$/gm, '');

  // Fix 2: Handle incomplete string literals
  const quotes = (cleaned.match(/"""/g) || []).length;
  if (quotes % 2 !== 0) {
    cleaned += '"""';
  }

  // Fix 3: Handle unclosed parentheses (basic)
  const openParens = (cleaned.match(/\(/g) || []).length;
  const closeParens = (cleaned.match(/\)/g) || []).length;
  if (openParens > closeParens) {
    cleaned += ')'.repeat(openParens - closeParens);
  }

  // Validate with Python AST (via Pyodide)
  try {
    // This would be done in Python:
    // compile(cleaned, '<string>', 'exec')
    return { valid: true, cleaned };
  } catch (e) {
    return { valid: false, error: (e as Error).message };
  }
}
```

### 11.2 LLM Response Edge Cases

#### 11.2.1 Empty Responses

**Problem**: LLM returns empty or whitespace-only responses.

**Solution**:
```typescript
function handleEmptyResponse(
  response: string,
  iteration: number,
  context: RLMContext
): { action: 'retry' | 'force-explore' | 'abort'; message?: string } {
  const trimmed = response.trim();

  if (!trimmed) {
    if (context.emptyResponseCount < 2) {
      return {
        action: 'retry',
        message: 'Your previous response was empty. Please provide a response ' +
                 'with either code to execute or a FINAL() answer.'
      };
    }
    return {
      action: 'force-explore',
      message: 'Multiple empty responses detected. Here is a suggested starting point:\n\n' +
               '```repl\nprint(type(context))\nprint(len(context) if hasattr(context, "__len__") else "N/A")\n```'
    };
  }

  return { action: 'retry' }; // Not empty, proceed normally
}
```

#### 11.2.2 Malformed FINAL() Statements

**Problem**: LLM outputs `FINAL()` in non-standard formats.

**Solution**:
```typescript
export function findFinalAnswerRobust(response: string): FinalAnswer | null {
  // Standard patterns
  const patterns = [
    // Standard: FINAL(answer)
    /^\s*FINAL\(([\s\S]*?)\)\s*$/m,
    // With quotes: FINAL("answer")
    /^\s*FINAL\("([\s\S]*?)"\)\s*$/m,
    // Variable: FINAL_VAR(name)
    /^\s*FINAL_VAR\(["']?(\w+)["']?\)\s*$/m,
    // Markdown code block variant: ```FINAL\n...\n```
    /```FINAL\s*\n([\s\S]*?)\n```/,
  ];

  for (const pattern of patterns) {
    const match = response.match(pattern);
    if (match) {
      if (pattern.source.includes('FINAL_VAR')) {
        return { type: 'FINAL_VAR', variable: match[1] };
      }
      return { type: 'FINAL', value: match[1].trim() };
    }
  }

  // Heuristic: Check for "final answer" language followed by content
  const heuristicMatch = response.match(
    /(?:my final answer is|the answer is|therefore,? the answer is)[:\s]*(.+)/i
  );
  if (heuristicMatch && response.includes('FINAL')) {
    // Model tried to give final answer but malformed it
    return { type: 'FINAL', value: heuristicMatch[1].trim() };
  }

  return null;
}
```

#### 11.2.3 FINAL() in Reasoning (False Positive)

**Problem**: Claude and other models may mention `FINAL()` while reasoning about it, not as actual answer.

**Solution**:
```typescript
export function isTrueFinal(response: string, finalMatch: FinalAnswer): boolean {
  // Check if FINAL appears in a reasoning context
  const reasoningPatterns = [
    /I (?:will|should|need to) (?:use|call) FINAL/i,
    /(?:using|calling) FINAL\(\) (?:to|will|would)/i,
    /FINAL\(\) (?:function|method|call)/i,
    /when (?:I'm|I am) ready.* FINAL/i,
  ];

  for (const pattern of reasoningPatterns) {
    if (pattern.test(response)) {
      // FINAL appears in reasoning context - likely not actual answer
      // Check if there's a "real" FINAL after the reasoning
      const reasoningEnd = response.search(pattern) + response.match(pattern)![0].length;
      const afterReasoning = response.slice(reasoningEnd);

      // Only count as true FINAL if it appears definitively after reasoning
      if (afterReasoning.match(/^\s*FINAL\(/m)) {
        return true;
      }
      return false;
    }
  }

  return true; // No reasoning patterns found, treat as real FINAL
}
```

### 11.3 Network & Provider Edge Cases

#### 11.3.1 Sub-LM Call Failures

**Problem**: `llm_query()` calls may fail due to rate limits, network issues, or provider errors.

**Solution**:
```typescript
export class ResilientLMHandler {
  private retryConfig = {
    maxRetries: 3,
    baseDelay: 1000,
    maxDelay: 10000,
    retryableErrors: ['RATE_LIMIT', 'TIMEOUT', 'NETWORK_ERROR', 'SERVICE_UNAVAILABLE'],
  };

  async llmQuery(prompt: string): Promise<string> {
    let lastError: Error | null = null;

    for (let attempt = 0; attempt < this.retryConfig.maxRetries; attempt++) {
      try {
        return await this.executeQuery(prompt);
      } catch (error) {
        lastError = error as Error;
        const errorType = this.classifyError(error);

        if (!this.retryConfig.retryableErrors.includes(errorType)) {
          throw error; // Non-retryable, fail immediately
        }

        const delay = Math.min(
          this.retryConfig.baseDelay * Math.pow(2, attempt),
          this.retryConfig.maxDelay
        );

        await this.sleep(delay);
      }
    }

    // All retries exhausted - return error as string for model to handle
    return `[LLM_QUERY_ERROR: ${lastError?.message || 'Unknown error'} after ${this.retryConfig.maxRetries} retries]`;
  }

  private classifyError(error: unknown): string {
    const message = (error as Error).message?.toLowerCase() || '';
    if (message.includes('rate limit')) return 'RATE_LIMIT';
    if (message.includes('timeout')) return 'TIMEOUT';
    if (message.includes('network') || message.includes('fetch')) return 'NETWORK_ERROR';
    if (message.includes('503') || message.includes('unavailable')) return 'SERVICE_UNAVAILABLE';
    return 'UNKNOWN';
  }
}
```

#### 11.3.2 Provider-Specific Token Limits

**Problem**: Different providers have different token limits for sub-calls.

**Solution**:
```typescript
const PROVIDER_LIMITS: Record<string, { maxInput: number; maxOutput: number }> = {
  'gpt-4o-mini': { maxInput: 128000, maxOutput: 16384 },
  'gpt-4o': { maxInput: 128000, maxOutput: 16384 },
  'claude-3-5-haiku': { maxInput: 200000, maxOutput: 8192 },
  'claude-3-5-sonnet': { maxInput: 200000, maxOutput: 8192 },
  'gemini-flash': { maxInput: 1000000, maxOutput: 8192 },
};

export function validateSubCallPrompt(
  prompt: string,
  subModel: string
): { valid: boolean; error?: string; truncated?: string } {
  const limits = PROVIDER_LIMITS[subModel] || { maxInput: 100000, maxOutput: 4096 };
  const estimatedTokens = Math.ceil(prompt.length / 4); // Rough estimate

  if (estimatedTokens > limits.maxInput * 0.9) {
    // Truncate with warning
    const maxChars = Math.floor(limits.maxInput * 0.8 * 4);
    return {
      valid: true,
      truncated: prompt.slice(0, maxChars) +
                 `\n\n[TRUNCATED: Original prompt was ${prompt.length} chars, ` +
                 `reduced to ${maxChars} to fit model limits]`,
    };
  }

  return { valid: true };
}
```

### 11.4 State Management Edge Cases

#### 11.4.1 Variable Name Collisions

**Problem**: Model may overwrite important variables like `context` or `llm_query`.

**Solution**:
```typescript
// Protected names that cannot be overwritten
const PROTECTED_NAMES = new Set([
  'context', 'llm_query', 'llm_query_batched', 'print',
  '__builtins__', '__name__', '__doc__',
]);

export function validateCodeSafety(code: string): { safe: boolean; warnings: string[] } {
  const warnings: string[] = [];

  // Check for assignments to protected names
  const assignmentPattern = /^(\w+)\s*=/gm;
  let match;
  while ((match = assignmentPattern.exec(code)) !== null) {
    if (PROTECTED_NAMES.has(match[1])) {
      warnings.push(
        `Warning: Attempting to reassign protected variable '${match[1]}'. ` +
        `This assignment will be blocked.`
      );
    }
  }

  // Check for del statements on protected names
  const delPattern = /\bdel\s+(\w+)/g;
  while ((match = delPattern.exec(code)) !== null) {
    if (PROTECTED_NAMES.has(match[1])) {
      warnings.push(
        `Warning: Attempting to delete protected variable '${match[1]}'. ` +
        `This will be blocked.`
      );
    }
  }

  return { safe: warnings.length === 0, warnings };
}
```

#### 11.4.2 REPL State Corruption

**Problem**: Error in one code block may leave REPL in corrupted state.

**Solution**:
```typescript
export class REPLStateManager {
  private snapshots: Map<number, REPLSnapshot> = new Map();

  async createSnapshot(repl: BaseEnvironment, iteration: number): Promise<void> {
    const locals = await repl.getLocals();
    this.snapshots.set(iteration, {
      iteration,
      locals: structuredClone(locals),
      timestamp: Date.now(),
    });

    // Keep only last 5 snapshots to limit memory
    if (this.snapshots.size > 5) {
      const oldest = Math.min(...this.snapshots.keys());
      this.snapshots.delete(oldest);
    }
  }

  async restoreSnapshot(
    repl: BaseEnvironment,
    targetIteration: number
  ): Promise<boolean> {
    const snapshot = this.snapshots.get(targetIteration);
    if (!snapshot) return false;

    await repl.reset();
    for (const [name, value] of Object.entries(snapshot.locals)) {
      await repl.setLocal(name, value);
    }
    return true;
  }
}
```

### 11.5 UI/UX Edge Cases

#### 11.5.1 Very Long REPL Output

**Problem**: Code produces extremely long output that would overwhelm the UI.

**Solution**:
```typescript
export function formatOutputForUI(
  output: string,
  config: { maxLines: number; maxCharsPerLine: number; collapsible: boolean }
): UIOutput {
  const lines = output.split('\n');

  if (lines.length <= config.maxLines) {
    return { type: 'inline', content: output };
  }

  // Create collapsible output
  const preview = lines.slice(0, 10).join('\n');
  const hidden = lines.slice(10);

  return {
    type: 'collapsible',
    preview: preview + `\n... (${hidden.length} more lines)`,
    fullContent: output,
    metadata: {
      totalLines: lines.length,
      previewLines: 10,
      hiddenLines: hidden.length,
    },
  };
}
```

#### 11.5.2 User Cancellation Mid-Iteration

**Problem**: User presses Ctrl+C during RLM execution.

**Solution**:
```typescript
export class RLMAbortHandler {
  private currentState: 'idle' | 'llm-call' | 'code-execution' | 'sub-call' = 'idle';

  async handleAbort(signal: AbortSignal): Promise<AbortResult> {
    if (!signal.aborted) return { aborted: false };

    switch (this.currentState) {
      case 'llm-call':
        // Can safely abort, just stop waiting for response
        return {
          aborted: true,
          partialResult: await this.getPartialLLMResponse(),
          canResume: false,
        };

      case 'code-execution':
        // Need to force-terminate REPL
        await this.repl.forceTerminate();
        return {
          aborted: true,
          partialResult: await this.getExecutionProgress(),
          canResume: true, // Can resume from last snapshot
          resumePoint: this.lastSnapshot,
        };

      case 'sub-call':
        // Abort the sub-call, log partial
        return {
          aborted: true,
          partialResult: `Sub-call aborted at iteration ${this.currentIteration}`,
          canResume: true,
          resumePoint: this.lastSnapshot,
        };

      default:
        return { aborted: true, canResume: false };
    }
  }
}
```

---

## 12. Model-Specific Adaptations

This section details the specific adaptations required for each supported model family based on the RLM paper's findings (Appendix A) and empirical testing.

### 12.1 Anthropic Claude Models

#### 12.1.1 Claude-Specific Challenges

| Issue | Description | Mitigation |
|-------|-------------|------------|
| FINAL in reasoning | Claude often mentions FINAL() while planning | Use `isTrueFinal()` detection |
| Verbose code comments | Generates excessive comments in REPL blocks | Strip comments from execution |
| Eager sub-calls | May call `llm_query()` before exploring context | Add exploration reminder |

#### 12.1.2 Claude System Prompt Modifications

```typescript
const CLAUDE_ADDITIONS = `
## Important Notes for This Environment

1. **Explore Before Querying**: Always examine the context variable using Python code
   before calling llm_query(). You have powerful tools like len(), type(), slicing,
   and regex at your disposal.

2. **FINAL() Timing**: Only use FINAL() or FINAL_VAR() when you have definitively
   determined the answer. Do NOT mention FINAL() in your reasoning - only use it
   as the actual final statement.

3. **Code Efficiency**: Keep your REPL code concise. Comments are not necessary
   in this environment as you can explain your reasoning in natural language
   outside the code blocks.
`;

export function buildClaudeSystemPrompt(basePrompt: string): string {
  return basePrompt + '\n\n' + CLAUDE_ADDITIONS;
}
```

#### 12.1.3 Claude Sub-Model Recommendations

| Root Model | Recommended Sub-Model | Rationale |
|------------|----------------------|-----------|
| claude-opus-4 | claude-3-5-haiku | Best cost/performance ratio |
| claude-sonnet-4 | claude-3-5-haiku | Fast, 200K context |
| claude-3-5-sonnet | claude-3-5-haiku | Same family compatibility |

### 12.2 OpenAI GPT Models

#### 12.2.1 GPT-Specific Challenges

| Issue | Description | Mitigation |
|-------|-------------|------------|
| Output truncation | o1/o3 may hit output limits during long reasoning | Increase max_tokens, add checkpoints |
| Code formatting | Sometimes uses `python` instead of `repl` | Accept both, normalize to `repl` |
| Parallel thinking | o1/o3 internal reasoning not visible | Trust the process, longer timeouts |

#### 12.2.2 GPT Prompt Adjustments

```typescript
const GPT_ADJUSTMENTS = {
  // o1/o3 models need explicit instruction about code blocks
  thinkingModels: `
When writing code for the REPL environment, you MUST use the \`\`\`repl tag,
not \`\`\`python. Only \`\`\`repl blocks will be executed.

Example:
\`\`\`repl
# This will be executed
print(len(context))
\`\`\`

\`\`\`python
# This will NOT be executed
print("This is just for illustration")
\`\`\`
`,

  // Standard GPT-4o works well with base prompt
  standard: '',
};

export function buildGPTSystemPrompt(basePrompt: string, modelId: string): string {
  if (modelId.startsWith('o1') || modelId.startsWith('o3')) {
    return basePrompt + '\n\n' + GPT_ADJUSTMENTS.thinkingModels;
  }
  return basePrompt;
}
```

#### 12.2.3 GPT Sub-Model Recommendations

| Root Model | Recommended Sub-Model | Rationale |
|------------|----------------------|-----------|
| gpt-4o | gpt-4o-mini | Paper's recommendation |
| o1 | gpt-4o-mini | Thinking model + fast sub-calls |
| o3 | gpt-4o-mini | Thinking model + fast sub-calls |
| gpt-4o-mini | gpt-4o-mini | Same model (self-recursion) |

### 12.3 Google Gemini Models

#### 12.3.1 Gemini-Specific Challenges

| Issue | Description | Mitigation |
|-------|-------------|------------|
| Massive context | 1M+ context may encourage lazy approaches | Encourage chunking anyway |
| Output format | May use different code block styles | Flexible parsing |
| Safety filters | Aggressive content filtering | Wrap code safely |

#### 12.3.2 Gemini Prompt Adjustments

```typescript
const GEMINI_ADDITIONS = `
## Environment Notes

Even though you may be able to fit large amounts of text in your context window,
the REPL environment is designed for efficient, programmatic access to data.
Using Python code to filter, chunk, and process the context will yield better
results than attempting to reason about very long strings directly.

The \`llm_query()\` function accepts prompts up to approximately 500K characters.
For larger analyses, chunk your data and use \`llm_query_batched()\`.
`;

export function buildGeminiSystemPrompt(basePrompt: string): string {
  return basePrompt + '\n\n' + GEMINI_ADDITIONS;
}
```

#### 12.3.3 Gemini Sub-Model Recommendations

| Root Model | Recommended Sub-Model | Rationale |
|------------|----------------------|-----------|
| gemini-2.0-pro | gemini-2.0-flash | Fast, large context |
| gemini-2.0-flash | gemini-2.0-flash | Self-recursion (already fast) |
| gemini-1.5-pro | gemini-1.5-flash | Legacy support |

### 12.4 Qwen Models

#### 12.4.1 Qwen-Specific Challenges (CRITICAL)

| Issue | Description | Mitigation |
|-------|-------------|------------|
| Excessive sub-calls | Makes 100s of unnecessary llm_query() calls | **MUST add cost warning** |
| Fine-grained processing | Processes line-by-line instead of batching | Enforce batching guidance |
| Code verbosity | Very detailed but inefficient code | Token limits on code blocks |

#### 12.4.2 Qwen Mandatory Warning (FROM PAPER APPENDIX D.1)

```typescript
// THIS WARNING IS MANDATORY FOR QWEN MODELS
const QWEN_MANDATORY_WARNING = `
╔══════════════════════════════════════════════════════════════════════════════╗
║  CRITICAL: llm_query() COST WARNING                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Each call to llm_query() incurs significant runtime and cost overhead.      ║
║                                                                              ║
║  DO:                                                                         ║
║  • Batch information: aim for ~200K characters per llm_query() call          ║
║  • If processing 1000 items, split into 5-10 chunks, not 1000 calls          ║
║  • Use llm_query_batched() for parallel processing                           ║
║                                                                              ║
║  DON'T:                                                                      ║
║  • Call llm_query() inside loops over individual items                       ║
║  • Make more than 50 total llm_query() calls per task                        ║
║  • Process line-by-line when you can process chunk-by-chunk                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
`;

export function buildQwenSystemPrompt(basePrompt: string): string {
  return QWEN_MANDATORY_WARNING + '\n\n' + basePrompt;
}
```

#### 12.4.3 Qwen Sub-Model Recommendations

| Root Model | Recommended Sub-Model | Rationale |
|------------|----------------------|-----------|
| qwen3-coder | qwen3-coder-fast (if available) | Same family |
| qwen-2.5-72b | qwen-2.5-7b | Smaller variant |

### 12.5 Model Detection and Routing

```typescript
// packages/opencode/src/rlm/prompts/model-adapter.ts

export type ModelFamily = 'claude' | 'openai' | 'gemini' | 'qwen' | 'other';

export function detectModelFamily(modelId: string): ModelFamily {
  const id = modelId.toLowerCase();

  if (id.includes('claude')) return 'claude';
  if (id.includes('gpt') || id.includes('o1') || id.includes('o3')) return 'openai';
  if (id.includes('gemini')) return 'gemini';
  if (id.includes('qwen')) return 'qwen';

  return 'other';
}

export function buildAdaptedSystemPrompt(
  basePrompt: string,
  modelId: string,
  metadata: QueryMetadata
): string {
  const family = detectModelFamily(modelId);

  switch (family) {
    case 'claude':
      return buildClaudeSystemPrompt(basePrompt);
    case 'openai':
      return buildGPTSystemPrompt(basePrompt, modelId);
    case 'gemini':
      return buildGeminiSystemPrompt(basePrompt);
    case 'qwen':
      return buildQwenSystemPrompt(basePrompt);
    default:
      return basePrompt; // Use base prompt for unknown models
  }
}

export function getRecommendedSubModel(rootModelId: string): string | null {
  const mappings: Record<string, string> = {
    // Claude
    'claude-opus-4': 'claude-3-5-haiku-latest',
    'claude-sonnet-4': 'claude-3-5-haiku-latest',
    'claude-3-5-sonnet': 'claude-3-5-haiku-latest',
    'claude-3-opus': 'claude-3-5-haiku-latest',

    // OpenAI
    'gpt-4o': 'gpt-4o-mini',
    'gpt-4-turbo': 'gpt-4o-mini',
    'o1': 'gpt-4o-mini',
    'o1-preview': 'gpt-4o-mini',
    'o3': 'gpt-4o-mini',
    'o3-mini': 'gpt-4o-mini',

    // Gemini
    'gemini-2.0-pro': 'gemini-2.0-flash',
    'gemini-1.5-pro': 'gemini-1.5-flash',

    // Qwen - note: sub-models may vary by provider
    'qwen3-coder': 'qwen-2.5-7b',
  };

  // Exact match
  if (mappings[rootModelId]) return mappings[rootModelId];

  // Partial match
  for (const [pattern, subModel] of Object.entries(mappings)) {
    if (rootModelId.toLowerCase().includes(pattern.toLowerCase())) {
      return subModel;
    }
  }

  return null; // Caller should handle fallback
}
```

---

## 13. Cost Management & Budgeting

This section provides comprehensive cost tracking, budgeting, and optimization strategies for RLM execution.

### 13.1 Cost Model

#### 13.1.1 Token Cost Estimation

```typescript
// packages/opencode/src/rlm/cost/estimator.ts

export interface TokenPricing {
  inputPer1M: number;   // USD per 1M input tokens
  outputPer1M: number;  // USD per 1M output tokens
  cachedInputPer1M?: number; // For providers with prompt caching
}

const MODEL_PRICING: Record<string, TokenPricing> = {
  // Claude (as of 2024)
  'claude-opus-4': { inputPer1M: 15.0, outputPer1M: 75.0 },
  'claude-sonnet-4': { inputPer1M: 3.0, outputPer1M: 15.0 },
  'claude-3-5-sonnet': { inputPer1M: 3.0, outputPer1M: 15.0, cachedInputPer1M: 0.30 },
  'claude-3-5-haiku': { inputPer1M: 0.80, outputPer1M: 4.0, cachedInputPer1M: 0.08 },

  // OpenAI
  'gpt-4o': { inputPer1M: 2.50, outputPer1M: 10.0, cachedInputPer1M: 1.25 },
  'gpt-4o-mini': { inputPer1M: 0.15, outputPer1M: 0.60, cachedInputPer1M: 0.075 },
  'o1': { inputPer1M: 15.0, outputPer1M: 60.0 },
  'o3-mini': { inputPer1M: 1.10, outputPer1M: 4.40 },

  // Gemini
  'gemini-2.0-pro': { inputPer1M: 1.25, outputPer1M: 5.0 },
  'gemini-2.0-flash': { inputPer1M: 0.075, outputPer1M: 0.30 },
};

export function estimateTokens(text: string): number {
  // Rough estimate: ~4 characters per token for English
  // More accurate would use tiktoken or provider-specific tokenizers
  return Math.ceil(text.length / 4);
}

export function estimateCost(
  inputText: string,
  outputText: string,
  modelId: string,
  cached: boolean = false
): { inputCost: number; outputCost: number; totalCost: number } {
  const pricing = MODEL_PRICING[modelId] || { inputPer1M: 1.0, outputPer1M: 2.0 };

  const inputTokens = estimateTokens(inputText);
  const outputTokens = estimateTokens(outputText);

  const inputRate = cached && pricing.cachedInputPer1M
    ? pricing.cachedInputPer1M
    : pricing.inputPer1M;

  const inputCost = (inputTokens / 1_000_000) * inputRate;
  const outputCost = (outputTokens / 1_000_000) * pricing.outputPer1M;

  return {
    inputCost,
    outputCost,
    totalCost: inputCost + outputCost,
  };
}
```

#### 13.1.2 RLM-Specific Cost Tracking

```typescript
// packages/opencode/src/rlm/cost/tracker.ts

export interface RLMCostBreakdown {
  rootModelCosts: {
    model: string;
    iterations: number;
    inputTokens: number;
    outputTokens: number;
    cost: number;
  };
  subModelCosts: {
    model: string;
    calls: number;
    inputTokens: number;
    outputTokens: number;
    cost: number;
  };
  totalCost: number;
  costPerIteration: number;
  costPerSubCall: number;
}

export class RLMCostTracker {
  private rootCosts: Array<{ input: string; output: string; model: string }> = [];
  private subCosts: Array<{ input: string; output: string; model: string }> = [];

  recordRootCall(input: string, output: string, model: string): void {
    this.rootCosts.push({ input, output, model });
  }

  recordSubCall(input: string, output: string, model: string): void {
    this.subCosts.push({ input, output, model });
  }

  getBreakdown(): RLMCostBreakdown {
    let rootInputTokens = 0, rootOutputTokens = 0, rootCost = 0;
    let subInputTokens = 0, subOutputTokens = 0, subCost = 0;

    for (const call of this.rootCosts) {
      const est = estimateCost(call.input, call.output, call.model);
      rootInputTokens += estimateTokens(call.input);
      rootOutputTokens += estimateTokens(call.output);
      rootCost += est.totalCost;
    }

    for (const call of this.subCosts) {
      const est = estimateCost(call.input, call.output, call.model);
      subInputTokens += estimateTokens(call.input);
      subOutputTokens += estimateTokens(call.output);
      subCost += est.totalCost;
    }

    const totalCost = rootCost + subCost;

    return {
      rootModelCosts: {
        model: this.rootCosts[0]?.model || 'unknown',
        iterations: this.rootCosts.length,
        inputTokens: rootInputTokens,
        outputTokens: rootOutputTokens,
        cost: rootCost,
      },
      subModelCosts: {
        model: this.subCosts[0]?.model || 'unknown',
        calls: this.subCosts.length,
        inputTokens: subInputTokens,
        outputTokens: subOutputTokens,
        cost: subCost,
      },
      totalCost,
      costPerIteration: this.rootCosts.length > 0 ? totalCost / this.rootCosts.length : 0,
      costPerSubCall: this.subCosts.length > 0 ? subCost / this.subCosts.length : 0,
    };
  }
}
```

### 13.2 Budget Enforcement

#### 13.2.1 Budget Configuration

```typescript
// packages/opencode/src/rlm/cost/budget.ts

export interface RLMBudgetConfig {
  maxTotalCost: number;        // Max USD for entire RLM execution
  maxCostPerIteration: number; // Max USD per iteration
  maxSubCallCost: number;      // Max USD for all sub-calls combined
  warningThreshold: number;    // Percentage (0-1) to warn at
  hardLimit: boolean;          // Whether to abort at limit or just warn
}

export const DEFAULT_BUDGET: RLMBudgetConfig = {
  maxTotalCost: 5.00,          // $5 max per RLM execution
  maxCostPerIteration: 0.50,   // $0.50 per iteration
  maxSubCallCost: 2.00,        // $2 for all sub-calls
  warningThreshold: 0.75,      // Warn at 75%
  hardLimit: true,             // Abort at limit
};

export class RLMBudgetEnforcer {
  private tracker: RLMCostTracker;
  private config: RLMBudgetConfig;
  private warnings: string[] = [];

  constructor(config: RLMBudgetConfig = DEFAULT_BUDGET) {
    this.config = config;
    this.tracker = new RLMCostTracker();
  }

  checkBudget(): BudgetStatus {
    const breakdown = this.tracker.getBreakdown();

    // Check total cost
    if (breakdown.totalCost >= this.config.maxTotalCost) {
      return {
        ok: false,
        reason: 'TOTAL_BUDGET_EXCEEDED',
        message: `Total cost ($${breakdown.totalCost.toFixed(2)}) exceeded budget ($${this.config.maxTotalCost.toFixed(2)})`,
        breakdown,
      };
    }

    // Check sub-call cost
    if (breakdown.subModelCosts.cost >= this.config.maxSubCallCost) {
      return {
        ok: false,
        reason: 'SUBCALL_BUDGET_EXCEEDED',
        message: `Sub-call cost ($${breakdown.subModelCosts.cost.toFixed(2)}) exceeded limit ($${this.config.maxSubCallCost.toFixed(2)})`,
        breakdown,
      };
    }

    // Check warnings
    const totalPct = breakdown.totalCost / this.config.maxTotalCost;
    const subPct = breakdown.subModelCosts.cost / this.config.maxSubCallCost;

    if (totalPct >= this.config.warningThreshold) {
      this.warnings.push(
        `Cost warning: ${Math.round(totalPct * 100)}% of total budget used ` +
        `($${breakdown.totalCost.toFixed(2)}/$${this.config.maxTotalCost.toFixed(2)})`
      );
    }

    if (subPct >= this.config.warningThreshold) {
      this.warnings.push(
        `Sub-call warning: ${Math.round(subPct * 100)}% of sub-call budget used`
      );
    }

    return {
      ok: true,
      breakdown,
      warnings: this.warnings,
    };
  }

  // Estimate cost before making a call
  estimateNextCall(
    inputText: string,
    expectedOutputLength: number,
    model: string,
    isSubCall: boolean
  ): { estimated: number; wouldExceedBudget: boolean } {
    const estimated = estimateCost(
      inputText,
      'x'.repeat(expectedOutputLength),
      model
    ).totalCost;

    const breakdown = this.tracker.getBreakdown();
    const projectedTotal = breakdown.totalCost + estimated;

    let wouldExceedBudget = projectedTotal > this.config.maxTotalCost;
    if (isSubCall) {
      const projectedSub = breakdown.subModelCosts.cost + estimated;
      wouldExceedBudget = wouldExceedBudget || projectedSub > this.config.maxSubCallCost;
    }

    return { estimated, wouldExceedBudget };
  }
}
```

### 13.3 Cost Optimization Strategies

#### 13.3.1 Prompt Caching Utilization

```typescript
// Leverage prompt caching for repeated system prompts
export class CachingOptimizer {
  private systemPromptHash: string | null = null;
  private cachedSystemTokens: number = 0;

  async optimizeForCaching(
    systemPrompt: string,
    provider: 'anthropic' | 'openai'
  ): Promise<{ prompt: string; cacheHint?: string }> {
    const hash = await this.hashPrompt(systemPrompt);

    if (provider === 'anthropic') {
      // Anthropic: Use cache_control breakpoints
      return {
        prompt: systemPrompt,
        cacheHint: 'ephemeral', // Cache for session duration
      };
    }

    if (provider === 'openai') {
      // OpenAI: Automatic caching for prompts >1024 tokens
      // No explicit hint needed, but ensure prefix stability
      return { prompt: systemPrompt };
    }

    return { prompt: systemPrompt };
  }
}
```

#### 13.3.2 Sub-Call Batching for Cost Efficiency

```typescript
export class SubCallOptimizer {
  // Combine multiple small queries into fewer larger ones
  static optimizeBatch(
    prompts: string[],
    maxCharsPerCall: number = 200000
  ): string[][] {
    const batches: string[][] = [];
    let currentBatch: string[] = [];
    let currentChars = 0;

    for (const prompt of prompts) {
      if (currentChars + prompt.length > maxCharsPerCall && currentBatch.length > 0) {
        batches.push(currentBatch);
        currentBatch = [];
        currentChars = 0;
      }
      currentBatch.push(prompt);
      currentChars += prompt.length;
    }

    if (currentBatch.length > 0) {
      batches.push(currentBatch);
    }

    return batches;
  }

  // Create a combined prompt for batch processing
  static createBatchPrompt(prompts: string[]): string {
    return prompts.map((p, i) =>
      `=== Query ${i + 1} ===\n${p}\n=== End Query ${i + 1} ===`
    ).join('\n\n');
  }

  // Parse batch response back into individual answers
  static parseBatchResponse(response: string, count: number): string[] {
    const pattern = /=== Answer (\d+) ===\n([\s\S]*?)\n=== End Answer \1 ===/g;
    const answers: string[] = new Array(count).fill('[No answer found]');

    let match;
    while ((match = pattern.exec(response)) !== null) {
      const index = parseInt(match[1]) - 1;
      if (index >= 0 && index < count) {
        answers[index] = match[2].trim();
      }
    }

    return answers;
  }
}
```

### 13.4 Cost Reporting UI

```typescript
// packages/opencode/src/rlm/cost/reporter.ts

export function formatCostReport(breakdown: RLMCostBreakdown): string {
  const lines = [
    '╔════════════════════════════════════════╗',
    '║         RLM Cost Summary               ║',
    '╠════════════════════════════════════════╣',
    `║ Root Model: ${breakdown.rootModelCosts.model.padEnd(25)} ║`,
    `║   Iterations: ${String(breakdown.rootModelCosts.iterations).padEnd(23)} ║`,
    `║   Tokens: ${formatTokens(breakdown.rootModelCosts.inputTokens)}/${formatTokens(breakdown.rootModelCosts.outputTokens)} (in/out) ║`,
    `║   Cost: $${breakdown.rootModelCosts.cost.toFixed(4).padEnd(28)} ║`,
    '╠────────────────────────────────────────╣',
    `║ Sub-Model: ${breakdown.subModelCosts.model.padEnd(26)} ║`,
    `║   Calls: ${String(breakdown.subModelCosts.calls).padEnd(28)} ║`,
    `║   Tokens: ${formatTokens(breakdown.subModelCosts.inputTokens)}/${formatTokens(breakdown.subModelCosts.outputTokens)} (in/out) ║`,
    `║   Cost: $${breakdown.subModelCosts.cost.toFixed(4).padEnd(28)} ║`,
    '╠════════════════════════════════════════╣',
    `║ TOTAL COST: $${breakdown.totalCost.toFixed(4).padEnd(24)} ║`,
    `║ Cost/Iteration: $${breakdown.costPerIteration.toFixed(4).padEnd(20)} ║`,
    '╚════════════════════════════════════════╝',
  ];

  return lines.join('\n');
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}
```

### 13.5 Configuration Schema Update

```typescript
// Add to packages/opencode/src/config/config.ts

const rlmCostConfigSchema = z.object({
  budget: z.object({
    maxTotalCost: z.number().min(0).default(5.0),
    maxCostPerIteration: z.number().min(0).default(0.5),
    maxSubCallCost: z.number().min(0).default(2.0),
    warningThreshold: z.number().min(0).max(1).default(0.75),
    hardLimit: z.boolean().default(true),
  }).optional(),
  tracking: z.object({
    enabled: z.boolean().default(true),
    logToFile: z.boolean().default(false),
    logPath: z.string().optional(),
  }).optional(),
}).optional();

// Example configuration in opencode.json
/*
{
  "experimental": {
    "rlm": {
      "enabled": true,
      "cost": {
        "budget": {
          "maxTotalCost": 10.0,
          "maxSubCallCost": 5.0,
          "hardLimit": true
        },
        "tracking": {
          "enabled": true,
          "logToFile": true,
          "logPath": "./rlm-costs.jsonl"
        }
      }
    }
  }
}
*/
```

---

## Appendix D: Complete Updated Implementation Checklist

This checklist supersedes Appendix C and incorporates all findings from the critical review.

### Phase 0: Remove/Fix Existing Wrong Code
- [ ] Delete or comment out current `rlm.txt` "think harder" prompt
- [ ] Remove placebo `rlm` options from `transform.ts`
- [ ] Update `llm.ts` to route to RLM orchestrator (not just append prompt)

### Phase 1: Core Infrastructure
- [ ] Pyodide REPL environment (PRIMARY)
- [ ] `llm_query()` bridge function with error handling
- [ ] `llm_query_batched()` with concurrency control
- [ ] `findCodeBlocks()` parser with corrected regex
- [ ] `findFinalAnswer()` parser with FINAL_VAR priority and false-positive detection
- [ ] Full system prompt from paper Appendix D
- [ ] User prompt templates for iteration 0 vs N
- [ ] Context metadata computation
- [ ] Code preprocessing and syntax validation

### Phase 2: LM Handler & Sub-Model
- [ ] LMHandler class with HTTP/WebSocket protocol
- [ ] Sub-model client registration
- [ ] Retry logic with exponential backoff
- [ ] Model-specific sub-model mappings
- [ ] Provider token limit validation

### Phase 3: Orchestrator
- [ ] Main completion loop with iteration control
- [ ] Safety limits (sub-calls, time, iterations without code)
- [ ] Output truncation (8192 chars default)
- [ ] FINAL/FINAL_VAR resolution with variable lookup
- [ ] Default answer generation on max iterations
- [ ] Abort signal handling with cleanup
- [ ] REPL state snapshots for recovery

### Phase 4: Integration
- [ ] Streaming event system for UI (`RLMStreamEvent` types)
- [ ] Session integration in `llm.ts`
- [ ] Bus events for real-time updates
- [ ] Configuration schema updates
- [ ] Footer UI with iteration counter and sub-call indicator
- [ ] Collapsible REPL output panels

### Phase 5: Model Adaptations
- [ ] Claude: FINAL detection adjustments, exploration reminders
- [ ] OpenAI: Code block format guidance for o1/o3
- [ ] Gemini: Chunking encouragement for large contexts
- [ ] Qwen: **MANDATORY** sub-call cost warning
- [ ] Model family detection router
- [ ] Sub-model recommendation function

### Phase 6: Cost Management
- [ ] Token cost estimation per provider
- [ ] RLM cost tracker (root + sub-calls)
- [ ] Budget enforcement with configurable limits
- [ ] Pre-call cost estimation
- [ ] Cost optimization via prompt caching
- [ ] Sub-call batching optimizer
- [ ] Cost reporting UI in terminal
- [ ] Optional cost logging to file

### Phase 7: Safety & Observability
- [ ] Execution timeout guards
- [ ] Memory monitoring for Pyodide
- [ ] Protected variable enforcement
- [ ] Error categorization and recovery
- [ ] Trajectory logging (JSONL format)
- [ ] REPL state corruption detection and recovery

### Phase 8: Testing
- [ ] Unit tests for parsing functions
- [ ] Unit tests for cost estimation
- [ ] Integration tests with mock LLMs
- [ ] End-to-end tests with S-NIAH benchmark
- [ ] Model-specific behavior tests
- [ ] Budget enforcement tests
- [ ] Abort/cancellation tests
