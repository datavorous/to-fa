import random
from dataclasses import dataclass
from pathlib import Path

_root = Path(__file__).parent.parent

_SYSTEM_CHAT = (
    "You are a helpful assistant. Give short, direct answers. "
    "No need to explain everything. Just answer the question."
)
_SYSTEM_CODE = (
    "You are an expert software engineer. Always write the COMPLETE implementation — "
    "never truncate, never use placeholders like '# ... rest of implementation' or 'pass'. "
    "Every class, every method, every edge case, fully written out. "
    "After the code write a detailed explanation covering: architecture decisions, "
    "time and space complexity of every operation, failure modes and how the code handles them, "
    "and at least two concrete usage examples with expected output."
)

# {slots} filled from _VOCAB — every request unique, no artificial cache hits
_CHAT_TEMPLATES = [
    "What's the best way to {fix} my {thing} without spending a lot of money?",
    "How long does it take to {learn} {skill} if I practice {freq}?",
    "Is it safe to {action} when you have {condition}?",
    "Why does my {appliance} make a {sound} sound when I {usage}?",
    "What should I eat before {activity} to have more energy?",
    "How do I get rid of {problem} on my {surface}?",
    "Can I {action} and {action2} on the same day?",
    "What's the difference between {thing1} and {thing2}?",
    "How many {unit} of {thing} should I {consume} per day?",
    "Why am I so {feeling} after {activity}?",
    "Is it bad to {habit} every day?",
    "What's a cheap alternative to {expensive_thing}?",
    "How do I know if my {body_part} is {condition} or just {minor}?",
    "What happens if I {action} for too long?",
    "Should I {action} before or after {event}?",
]

_CHAT_LONG_TEMPLATES = [
    (
        "I've been dealing with {problem} for about {duration}. "
        "I've already tried {attempt1} and {attempt2} but neither worked. "
        "I {constraint}. What are some practical things I can actually do? "
        "I don't want generic advice — I want something concrete for someone in my situation."
    ),
    (
        "I want to start {goal} but I have no idea where to begin. "
        "I have {resource} and about {time} per day to dedicate to it. "
        "I've tried before and {past_failure}. "
        "What's the realistic first month look like and what should I focus on first?"
    ),
    (
        "My {relation} is {situation} and I don't know how to help. "
        "They {behavior} and whenever I try to {approach} they {reaction}. "
        "I'm worried about {concern}. What should I actually do or say?"
    ),
    (
        "I'm trying to decide between {option1} and {option2}. "
        "My situation is that I {context}. "
        "I've heard {myth1} but also {myth2} and I don't know what's true. "
        "Can you help me think through this properly?"
    ),
    (
        "I moved to {place} {duration} ago and I'm struggling with {struggle}. "
        "I {social_situation} and I find it hard to {challenge}. "
        "What are realistic ways to {goal} when you're {constraint}?"
    ),
]

_VOCAB = {
    "fix": ["repair", "clean", "fix", "restore", "unclog", "reset", "replace"],
    "thing": [
        "laptop",
        "phone",
        "bike",
        "washing machine",
        "router",
        "car",
        "blender",
        "sink",
        "lock",
    ],
    "learn": [
        "learn",
        "get decent at",
        "pick up",
        "become good at",
        "master the basics of",
    ],
    "skill": [
        "guitar",
        "cooking",
        "Spanish",
        "swimming",
        "drawing",
        "touch typing",
        "chess",
        "running",
    ],
    "freq": [
        "every day",
        "a few times a week",
        "on weekends",
        "for 20 minutes a day",
        "an hour a week",
    ],
    "action": [
        "drink coffee",
        "take ibuprofen",
        "skip breakfast",
        "exercise",
        "nap",
        "fast",
        "stretch",
        "go for a run",
    ],
    "action2": [
        "drink alcohol",
        "take a cold shower",
        "eat a big meal",
        "meditate",
        "take vitamins",
    ],
    "condition": [
        "a cold",
        "back pain",
        "a headache",
        "high blood pressure",
        "poor sleep",
        "stress",
        "low iron",
    ],
    "appliance": [
        "fridge",
        "microwave",
        "dishwasher",
        "air conditioner",
        "washing machine",
        "dryer",
        "oven",
    ],
    "sound": [
        "clicking",
        "buzzing",
        "humming",
        "rattling",
        "beeping",
        "grinding",
        "dripping",
    ],
    "usage": [
        "turn it on",
        "open it",
        "close it",
        "run it",
        "start a cycle",
        "change the temperature",
    ],
    "activity": [
        "a workout",
        "a long run",
        "an exam",
        "a job interview",
        "a flight",
        "a big meeting",
        "hiking",
    ],
    "problem": [
        "a stain",
        "mold",
        "rust",
        "scratches",
        "a smell",
        "grease",
        "limescale",
        "paint",
    ],
    "surface": [
        "carpet",
        "shirt",
        "wall",
        "ceiling",
        "pan",
        "wooden floor",
        "glass",
        "leather sofa",
    ],
    "thing1": [
        "a cold",
        "the flu",
        "food intolerance",
        "an allergy",
        "a sprain",
        "burnout",
        "dehydration",
    ],
    "thing2": [
        "food poisoning",
        "a virus",
        "a pulled muscle",
        "anxiety",
        "low blood sugar",
        "exhaustion",
    ],
    "unit": ["glasses", "cups", "grams", "mg", "servings", "hours", "litres"],
    "consume": ["drink", "eat", "take", "consume", "have"],
    "feeling": [
        "tired",
        "bloated",
        "anxious",
        "irritable",
        "unmotivated",
        "cold",
        "hungry",
        "restless",
    ],
    "habit": [
        "drink one coffee",
        "skip lunch",
        "stay up past midnight",
        "eat sugar",
        "sit all day",
        "check my phone",
    ],
    "expensive_thing": [
        "a gym membership",
        "a therapist",
        "organic food",
        "a personal trainer",
        "a standing desk",
        "meal prep services",
    ],
    "body_part": ["knee", "shoulder", "back", "wrist", "ankle", "neck", "hip", "elbow"],
    "minor": [
        "soreness",
        "stiffness",
        "a bruise",
        "normal fatigue",
        "growing pains",
        "tension",
    ],
    "event": [
        "eating",
        "sleeping",
        "working out",
        "a blood test",
        "a flight",
        "a stressful day",
    ],
    # long template slots
    "problem": [
        "back pain",
        "poor sleep",
        "low energy",
        "anxiety",
        "weight gain",
        "bad skin",
        "joint pain",
    ],
    "duration": [
        "two weeks",
        "three months",
        "six months",
        "about a year",
        "a few weeks",
    ],
    "attempt1": [
        "stretching every morning",
        "cutting out coffee",
        "taking supplements",
        "going to bed earlier",
        "drinking more water",
    ],
    "attempt2": [
        "following advice online",
        "changing my diet",
        "exercising more",
        "seeing a doctor",
        "reducing screen time",
    ],
    "constraint": [
        "work full time",
        "live alone",
        "don't have much money",
        "have a busy schedule",
        "have a small apartment",
    ],
    "goal": [
        "getting fit",
        "learning to cook",
        "saving money",
        "building a daily routine",
        "reading more",
        "meditating",
    ],
    "resource": [
        "very little equipment",
        "a basic kitchen",
        "around 50 dollars a week",
        "no prior experience",
        "limited space",
    ],
    "time": ["20 minutes", "30 minutes", "an hour", "45 minutes"],
    "past_failure": [
        "gave up after a week",
        "got bored",
        "didn't see results",
        "couldn't stay consistent",
        "lost motivation",
    ],
    "relation": ["partner", "parent", "friend", "sibling", "colleague", "roommate"],
    "situation": [
        "going through a really hard time",
        "struggling with their health",
        "dealing with job loss",
        "very stressed lately",
        "isolating themselves",
    ],
    "behavior": [
        "pushes people away",
        "refuses to talk about it",
        "gets defensive",
        "shuts down",
        "dismisses help",
    ],
    "approach": [
        "bring it up",
        "offer help",
        "check in on them",
        "suggest professional help",
        "talk about it",
    ],
    "reaction": [
        "gets angry",
        "changes the subject",
        "says they're fine",
        "goes quiet",
        "gets upset",
    ],
    "concern": [
        "their mental health",
        "them making a bad decision",
        "the situation getting worse",
        "our relationship",
    ],
    "option1": [
        "renting",
        "buying a car",
        "staying in my current job",
        "going back to school",
        "moving cities",
    ],
    "option2": [
        "buying",
        "using public transport",
        "switching careers",
        "doing an online course",
        "staying put",
    ],
    "context": [
        "don't have much savings",
        "am in my late 20s",
        "have a stable income",
        "have a family to consider",
        "am early in my career",
    ],
    "myth1": [
        "it's always better to own",
        "you need a degree to get a good job",
        "renting is throwing money away",
        "you need to be young to start",
    ],
    "myth2": [
        "it's never worth it",
        "the market is too unpredictable",
        "experience doesn't matter anymore",
        "it's too late to change",
    ],
    "place": ["a new city", "a new country", "a small town", "a big city", "abroad"],
    "struggle": [
        "making friends",
        "finding a community",
        "feeling lonely",
        "adjusting to the culture",
        "the language barrier",
    ],
    "social_situation": [
        "work remotely",
        "don't know anyone here",
        "am quite introverted",
        "work long hours",
        "am shy around new people",
    ],
    "challenge": [
        "put myself out there",
        "start conversations",
        "find people with similar interests",
        "feel comfortable socially",
    ],
}

_CODE_TEMPLATES_SHORT = [
    "Write a Python function to {task}. Include type hints and a brief docstring.",
    "Implement {data_structure} in Python with {ops}. Explain the time complexity of each operation.",
    "Write a Python {pattern} that {behavior}. Show a usage example.",
    "How do I {task} in Python without using any external libraries? Show a clean implementation.",
    "Write a Python function that takes {input_desc} and returns {output_desc}. Handle edge cases.",
]

_CODE_TEMPLATES_LONG = [
    (
        "Build a production-ready Python {system} that handles {requirement1}, {requirement2}, and {requirement3}. "
        "It should use {tech} and expose a clean API. "
        "Include proper error handling, logging, and at least one test. "
        "Walk through the design decisions after the code."
    ),
    (
        "I'm building a {system} in Python. Requirements: {requirement1}. {requirement2}. {requirement3}. "
        "The system must handle {edge_case} gracefully. "
        "Use {tech}. Write the full implementation with explanation."
    ),
    (
        "Implement a {system} from scratch in Python. "
        "It needs to support {ops} operations efficiently. "
        "Constraints: {constraint1} and {constraint2}. "
        "Write the complete working code, then explain the algorithmic choices and complexity."
    ),
]

_CODE_VOCAB = {
    "task": [
        "reverse a linked list",
        "find all permutations of a string",
        "implement LRU cache",
        "parse a CSV without pandas",
        "flatten a nested dict",
        "validate an email address",
        "merge two sorted lists",
        "find duplicates in a list",
    ],
    "data_structure": [
        "a min-heap",
        "a trie",
        "a circular buffer",
        "a doubly linked list",
        "a hash map",
        "a deque",
        "a priority queue",
    ],
    "ops": [
        "insert, delete, and search",
        "push, pop, and peek",
        "enqueue and dequeue",
        "get, put, and evict",
        "add, remove, and contains",
    ],
    "pattern": ["decorator", "context manager", "generator", "metaclass", "descriptor"],
    "behavior": [
        "retries on failure with backoff",
        "measures execution time",
        "limits concurrent calls",
        "caches results with TTL",
        "validates function arguments",
    ],
    "input_desc": [
        "a list of integers",
        "a string",
        "a nested dictionary",
        "a list of tuples",
        "a binary tree",
    ],
    "output_desc": [
        "the sorted result",
        "a frequency map",
        "the flattened structure",
        "grouped items",
        "the longest sequence",
    ],
    "system": [
        "rate limiter",
        "job queue",
        "cache layer",
        "event dispatcher",
        "connection pool",
        "task scheduler",
        "pub-sub broker",
        "circuit breaker",
    ],
    "requirement1": [
        "thread-safe operations",
        "configurable TTL per entry",
        "priority-based scheduling",
        "automatic retry on failure",
        "backpressure when full",
    ],
    "requirement2": [
        "graceful shutdown",
        "metrics collection",
        "pluggable backends",
        "async and sync support",
        "per-client rate limits",
    ],
    "requirement3": [
        "serialization to disk",
        "health check endpoint",
        "dead letter queue",
        "histogram of wait times",
        "configurable eviction policy",
    ],
    "tech": [
        "asyncio",
        "threading.Lock",
        "dataclasses",
        "contextlib",
        "heapq and asyncio.Queue",
        "collections.deque",
    ],
    "edge_case": [
        "concurrent access",
        "empty inputs",
        "timeout during processing",
        "duplicate keys",
        "malformed payloads",
        "partial failures",
    ],
    "constraint1": [
        "no external dependencies",
        "O(1) average lookup",
        "memory bounded to N items",
        "must be picklable",
    ],
    "constraint2": [
        "thread-safe without a global lock",
        "supports both sync and async callers",
        "recoverable after crash",
        "zero-copy where possible",
    ],
}


def _fill(template, vocab, rng):
    import re

    slots = re.findall(r"\{(\w+)\}", template)
    result = template
    for slot in slots:
        if slot in vocab:
            result = result.replace("{" + slot + "}", rng.choice(vocab[slot]), 1)
    return result


def _load_corpus():
    src = _root / "factory"
    files = {}
    for p in sorted(src.glob("*.py")):
        if p.name == "__init__.py":
            continue
        files[p.name] = p.read_text()
    return files


_CORPUS = _load_corpus()

_LISO_TASKS = [
    "Summarize what this module does in 3 bullet points. Be concise.",
    "List every function and class defined in this file with a one-line description of each.",
    "Identify the top 3 potential bugs or edge cases that are not handled.",
    "What are the key data flows through this module? Answer in plain English, no code.",
    "Rate the code quality on a scale of 1-10 and justify in 2-3 sentences.",
    "What would break first under high concurrency? Name the specific lines.",
    "List all external dependencies this file relies on and what each is used for.",
    "What design pattern is this code using? Name it and explain in one paragraph.",
]

_LILO_TASKS = [
    "Rewrite this entire module in Rust. Keep all logic identical, use idiomatic Rust.",
    "Translate this Python code to x86-64 assembly (NASM syntax). Include comments explaining each section.",
    "Refactor this module to use a fully async architecture with asyncio throughout. Rewrite every function.",
    "Convert this code to Go. Use goroutines where Python uses threads or async. Full implementation.",
    "Rewrite this as a C extension module that can be imported from Python via ctypes. Full working C code.",
    "Add comprehensive type annotations, docstrings, and a full pytest test suite covering every function. Write all of it out.",
    "Rewrite this module with zero global state. All state must be passed explicitly. Show the full refactored code.",
    "Port this to JavaScript (Node.js). Use async/await throughout. Full working implementation.",
]


# Shared system prompt for prefix-caching experiments (~500 tokens).
# Fixed across all LISO/LILO requests so the KV cache can reuse the prefix.
_SHARED_PREFIX_SYSTEM = (
    "You are an expert code reviewer with deep experience in systems programming, "
    "distributed systems, and production Python. Your reviews are trusted by senior "
    "engineers and are used to gate production deployments.\n\n"
    "When reviewing code, follow this protocol:\n"
    "1. CORRECTNESS first — identify any logic errors, off-by-one bugs, race conditions, "
    "or incorrect assumptions about inputs. Cite exact line numbers.\n"
    "2. SECURITY — flag injection vulnerabilities, insecure defaults, missing input "
    "validation, credential exposure, or unsafe deserialization.\n"
    "3. PERFORMANCE — identify O(n²) or worse algorithms where O(n log n) is achievable, "
    "unnecessary allocations in hot paths, missing caching opportunities, and I/O patterns "
    "that will not scale.\n"
    "4. ANTI-PATTERNS — flag global mutable state, God objects, violation of single "
    "responsibility, missing error propagation, swallowed exceptions, and magic numbers.\n"
    "5. STYLE — only after the above. PEP8, naming, docstring coverage, type hint "
    "completeness. Do not prioritise style over correctness.\n\n"
    "Format: lead with a severity summary (CRITICAL / HIGH / MEDIUM / LOW item counts), "
    "then enumerate findings in severity order. Each finding: severity tag, file and line "
    "reference, one-sentence description, and a concrete fix (code snippet preferred). "
    "End with a two-sentence overall assessment and a ship/hold recommendation.\n\n"
    "Do not summarise what the code does unless it is directly relevant to a finding. "
    "Do not praise the author. Do not use filler phrases like 'great job' or 'looks good'. "
    "Be direct, be precise, be actionable. A review that misses a critical bug is worse "
    "than one that is too harsh. When in doubt, flag it."
)


def _make_liso_prompt(rng, shared_prefix=False):
    fname, src = rng.choice(list(_CORPUS.items()))
    task = rng.choice(_LISO_TASKS)
    return f"### `{fname}`\n\n```python\n{src}\n```\n\n{task}"


def _make_lilo_prompt(rng, shared_prefix=False):
    fname, src = rng.choice(list(_CORPUS.items()))
    task = rng.choice(_LILO_TASKS)
    return f"### `{fname}`\n\n```python\n{src}\n```\n\n{task}"


def _make_chat_prompt(rng):
    if rng.random() < 0.6:
        return _fill(rng.choice(_CHAT_TEMPLATES), _VOCAB, rng)
    else:
        return _fill(rng.choice(_CHAT_LONG_TEMPLATES), _VOCAB, rng)


def _make_code_prompt(rng, long=False):
    if long:
        return _fill(rng.choice(_CODE_TEMPLATES_LONG), _CODE_VOCAB, rng)
    else:
        return _fill(rng.choice(_CODE_TEMPLATES_SHORT), _CODE_VOCAB, rng)


@dataclass
class Request:
    id: str
    profile: str
    system: str
    user: str
    max_tokens: int
    temperature: float
    length: str


def bucket(max_tokens):
    if max_tokens <= 512:
        return "short"
    return "long"


def generate(CFG):
    rng = random.Random(CFG.seed)
    requests = []

    # SISO: short prompt, short output (chat / Q&A)
    for i in range(CFG.siso_n):
        max_tokens = rng.randint(*CFG.siso_max_tokens)
        requests.append(
            Request(
                id=f"siso-{i:04d}",
                profile="siso",
                system=_SYSTEM_CHAT,
                user=_make_chat_prompt(rng),
                max_tokens=max_tokens,
                temperature=round(rng.uniform(*CFG.siso_temperature), 2),
                length=bucket(max_tokens),
            )
        )

    # SILO: short prompt, long output (write an essay / story / code from brief spec)
    for i in range(CFG.silo_n):
        max_tokens = rng.randint(*CFG.silo_max_tokens)
        b = bucket(max_tokens)
        requests.append(
            Request(
                id=f"silo-{i:04d}",
                profile="silo",
                system=_SYSTEM_CODE,
                user=_make_code_prompt(rng, long=(b == "long")),
                max_tokens=max_tokens,
                temperature=round(rng.uniform(*CFG.silo_temperature), 2),
                length=b,
            )
        )

    shared_prefix = getattr(CFG, "shared_prefix", False)
    liso_system = (
        _SHARED_PREFIX_SYSTEM
        if shared_prefix
        else "You are a senior engineer doing a code review. Be precise and concise."
    )
    lilo_system = _SHARED_PREFIX_SYSTEM if shared_prefix else _SYSTEM_CODE

    # LISO: long prompt (real source file) + short output (code review / summarisation)
    for i in range(getattr(CFG, "liso_n", 0)):
        max_tokens = rng.randint(*getattr(CFG, "liso_max_tokens", [64, 256]))
        requests.append(
            Request(
                id=f"liso-{i:04d}",
                profile="liso",
                system=liso_system,
                user=_make_liso_prompt(rng),
                max_tokens=max_tokens,
                temperature=0.2,
                length="short",
            )
        )

    # LILO: long prompt (real source file) + long output (code conversion / rewrite)
    for i in range(getattr(CFG, "lilo_n", 0)):
        max_tokens = rng.randint(*getattr(CFG, "lilo_max_tokens", [2048, 3072]))
        requests.append(
            Request(
                id=f"lilo-{i:04d}",
                profile="lilo",
                system=lilo_system,
                user=_make_lilo_prompt(rng),
                max_tokens=max_tokens,
                temperature=0.2,
                length="long",
            )
        )

    rng.shuffle(requests)
    return requests
