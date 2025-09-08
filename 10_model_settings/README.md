# 📘 OpenAI Agents SDK – ModelSettings Complete Guide (Roman Urdu)

## Introduction

`ModelSettings` aik **central configuration object** hai jo aapke LLM ke **behavior**, **output style**, aur **API call customization** ko control karta hai. Ye class optional model configuration parameters hold karta hai jaise temperature, top_p, penalties, truncation waghaira.

**Important**: Har model/provider sabhi parameters support nahi karta, isliye specific model aur provider ka API documentation check karna zaroori hai.

---

## 🔑 Core Parameters Detailed Breakdown

### 🎲 Randomness & Creativity Control

#### `temperature: float | None = None`
- **Kya hai**: Model responses mai randomness/creativity control karta hai
- **Range**: 0.0 (bilkul deterministic) se 2.0+ (bohot zyada creative)
- **Impact**:
  - `0.0` → Same input hamesha same output dega (data processing ke liye ideal)
  - `0.7` → Balanced creativity (chatbots ke liye achha)
  - `1.0+` → High creativity (creative writing ke liye)
- **Memory Impact**: Higher values mai model zyada token possibilities explore karta hai → compute heavy

```python
from openai.agents import ModelSettings
# Balanced creativity ke liye
settings = ModelSettings(temperature=0.7)
```

#### `top_p: float | None = None`
- **Kya hai**: Nucleus sampling - temperature ka alternative
- **Kaise kaam karta**: Model sirf un tokens ko consider karta hai jo top P% probability mass banate hain
- **Temperature se difference**:
  - `temperature` = globally sabhi tokens ko affect karta hai
  - `top_p` = probability distribution ke base par dynamic cutoff banata hai
- **Best Practice**: Ya to temperature use karo ya top_p, dono saath mai nahi

```python
# Nucleus sampling ke liye
settings = ModelSettings(top_p=0.9)
```

### 📉 Repetition Control

#### `frequency_penalty: float | None = None`
- **Purpose**: Same tokens/phrases ki repetition kam karta hai
- **Range**: 0.0 (koi penalty nahi) se 2.0 (strong penalty)
- **Use Case**: Model ko same words baar baar repeat karne se rokne ke liye
- **Example**: Diverse product descriptions generate karne ke liye useful

```python
# Repetition avoid karne ke liye
settings = ModelSettings(frequency_penalty=0.5)
```

#### `presence_penalty: float | None = None`
- **Purpose**: Model ko naye topics ke baare mai baat karne encourage karta hai
- **Effect**: Jo tokens pehle appear ho chuke hain unko penalize karta hai
- **Use Case**: Brainstorming sessions, idea generation, topic diversity ensure karne ke liye

```python
# Naye topics encourage karne ke liye
settings = ModelSettings(presence_penalty=0.8)
```

### 🛠️ Tool Management

#### `tool_choice: ToolChoice | None = None`
- **Purpose**: Control karta hai ke model kaunse tools use kar sakta hai
- **Options**:
  - `ToolChoice.auto` → Model khud decide karta hai kab tools use karna hai
  - `ToolChoice.any` → Model ko kam se kam aik tool use karna hi hoga
  - `ToolChoice.none` → Koi tool usage allowed nahi
  - Specific tool name → Particular tool ka usage force karta hai
- **Strategic Use**: Agent behavior flow control karne ke liye critical hai

```python
from openai.agents import ToolChoice
# Tool usage force karne ke liye
settings = ModelSettings(tool_choice=ToolChoice.any)
```

#### `parallel_tool_calls: bool | None = None`
- **Purpose**: Model ko multiple tools simultaneously call karne ki permission deta hai
- **Benefits**: Efficiency increase, faster task completion
- **Drawbacks**: Debug karna mushkil, conflicting tool outputs ka risk
- **Default**: Zyada tar providers (jaise OpenAI) ye by default enable rakhte hain

```python
# Multiple tools ek saath use karne ke liye
settings = ModelSettings(parallel_tool_calls=True)
```

### ✂️ Content Management

#### `truncation: Literal['auto', 'disabled'] | None = None`
- **Purpose**: Jab input model ke context window se exceed kare to kya karna hai
- **Options**:
  - `'auto'` → SDK automatically purane messages truncate kar deta hai
  - `'disabled'` → Context limit exceed hone par error throw karta hai
- **Best Practice**: Development mai debugging ke liye `'disabled'` use karo

```python
# Auto truncation ke liye
settings = ModelSettings(truncation="auto")
```

#### `max_tokens: int | None = None`
- **Purpose**: Model responses ki length limit karta hai
- **Impact**: Direct cost control aur response size management
- **Guidelines**:
  - Chatbots: 300-800 tokens
  - Code generation: 1000-2000 tokens
  - Summaries: 100-300 tokens

```python
# Response length control ke liye
settings = ModelSettings(max_tokens=500)
```

### 🧠 Advanced Model Features

#### `reasoning: Reasoning | None = None`
- **Purpose**: Reasoning models (jaise GPT-o1) ke liye special configuration
- **Contains**: Parameters jaise `max_steps`, `budget`, reasoning depth
- **Use Case**: Complex problem-solving tasks jo step-by-step thinking require karte hain

```python
from openai.agents import Reasoning
# Reasoning model ke liye
settings = ModelSettings(reasoning=Reasoning(max_steps=10))
```

#### `verbosity: Literal['low', 'medium', 'high'] | None = None`
- **Purpose**: Model responses kitne detailed honge ye control karta hai
- **Impact**:
  - `'low'` → Concise, direct answers
  - `'high'` → Detailed explanations with examples

```python
# Detailed responses ke liye
settings = ModelSettings(verbosity="high")
```

### 📊 Monitoring & Analytics

#### `store: bool | None = None`
- **Purpose**: Model responses ko later retrieval ke liye save karta hai
- **Use Cases**: Chat history, conversation persistence, analytics
- **API Differences**:
  - Responses API: By default enabled
  - Chat Completions API: By default disabled

```python
# Response storage ke liye
settings = ModelSettings(store=True)
```

#### `include_usage: bool | None = None`
- **Purpose**: Response ke saath token usage statistics return karta hai
- **Benefits**: Cost monitoring, performance optimization
- **Data Provided**: Input tokens, output tokens, total tokens

```python
# Cost monitoring ke liye
settings = ModelSettings(include_usage=True)
```

#### `metadata: dict[str, str] | None = None`
- **Purpose**: Tracking aur analytics ke liye custom tags attach karta hai
- **Use Cases**: User identification, session tracking, A/B testing

```python
# Analytics ke liye
settings = ModelSettings(metadata={"user_id": "123", "experiment": "variant_a"})
```

### 🔍 Response Analysis

#### `response_include: list[ResponseIncludable | str] | None = None`
- **Purpose**: Responses mai additional raw data include karta hai
- **Options**: `"raw_responses"`, `"tool_calls"`, `"message.output_text.logprobs"`
- **Use Case**: Debugging, advanced analysis, custom processing

```python
# Extra data include karne ke liye
settings = ModelSettings(response_include=["raw_responses", "tool_calls"])
```

#### `top_logprobs: int | None = None`
- **Purpose**: Top N tokens ke liye probability scores return karta hai
- **Use Cases**:
  - Uncertainty analysis
  - Alternative response generation
  - Model confidence assessment

```python
# Token probabilities ke liye
settings = ModelSettings(top_logprobs=3)
```

### ⚙️ Advanced Customization

#### `extra_query: Query | None = None`
- **Purpose**: API calls mai custom query parameters add karta hai
- **Use Case**: Experimental features, provider-specific options

```python
# Custom query params ke liye
settings = ModelSettings(extra_query={"exp_flag": True})
```

#### `extra_body: Body | None = None`
- **Purpose**: Request body mai custom fields add karta hai
- **Use Case**: Debug flags, experimental parameters

```python
# Custom body fields ke liye
settings = ModelSettings(extra_body={"debug": True})
```

#### `extra_headers: Headers | None = None`
- **Purpose**: Requests ke liye custom HTTP headers
- **Use Case**: Tracing, authentication, custom routing

```python
# Custom headers ke liye
settings = ModelSettings(extra_headers={"x-trace-id": "abc123"})
```

#### `extra_args: dict[str, Any] | None = None`
- **Purpose**: Model API mai arbitrary parameters directly pass karta hai
- **Warning**: Carefully use karo - sabhi models sabhi parameters support nahi karte
- **Use Case**: Bleeding-edge features, provider-specific options

```python
# Direct API parameters ke liye
settings = ModelSettings(extra_args={"stream_options": {"include_usage": True}})
```

### 🔄 Configuration Management

#### `resolve(override: ModelSettings | None) -> ModelSettings`
- **Purpose**: Base settings ko override settings ke saath merge karta hai
- **Logic**: Override mai jo non-None values hain wo precedence lete hain
- **Use Case**: Multi-layer configuration (global → user → session → request)

```python
# Settings merge karne ke liye
base = ModelSettings(temperature=0.7, max_tokens=200)
override = ModelSettings(max_tokens=100)
final = base.resolve(override)  # temperature=0.7, max_tokens=100
```

---

## 🎯 Practical Usage Patterns

### Pattern 1: Balanced Chatbot
```python
chatbot_settings = ModelSettings(
    temperature=0.7,           # Balanced creativity
    max_tokens=500,           # Reasonable response length
    frequency_penalty=0.3,    # Repetition kam karo
    store=True,              # Conversations save karo
    include_usage=True       # Costs monitor karo
)
```

### Pattern 2: Deterministic Data Processing
```python
data_settings = ModelSettings(
    temperature=0.0,          # Bilkul deterministic
    top_p=1.0,               # Koi nucleus sampling nahi
    truncation="disabled",    # Context overflow par fail karo
    max_tokens=1000,         # Structured output ke liye sufficient
    tool_choice=ToolChoice.any  # Data processing ke liye tools use karna zaroori
)
```

### Pattern 3: Creative Content Generation
```python
creative_settings = ModelSettings(
    temperature=1.2,          # High creativity
    presence_penalty=0.8,     # Naye topics encourage karo
    verbosity="high",         # Detailed responses
    max_tokens=2000          # Longer creative outputs allow karo
)
```

### Pattern 4: Production Monitoring
```python
production_settings = ModelSettings(
    temperature=0.7,
    store=True,
    include_usage=True,
    metadata={"environment": "prod", "version": "1.2.3"},
    extra_headers={"x-trace-id": generate_trace_id()}
)
```

---

## 📋 Quick Reference Categories

### Creativity Control
- `temperature` - Global randomness
- `top_p` - Local probability cutoff

### Repetition Control
- `frequency_penalty` - Same tokens avoid karo
- `presence_penalty` - Naye topics encourage karo

### Tool Behavior
- `tool_choice` - Kaunse tools use kar sakte hain
- `parallel_tool_calls` - Multiple tools ek saath

### Limits & Constraints
- `max_tokens` - Response length limit
- `truncation` - Context overflow handling

### Advanced Features
- `reasoning` - Reasoning models ke liye
- `verbosity` - Response detail level

### Monitoring & Logging
- `store` - Response storage
- `include_usage` - Token usage stats
- `response_include` - Extra raw data
- `top_logprobs` - Token probabilities
- `metadata` - Custom tracking tags

### Custom Extensibility
- `extra_query` - Custom query params
- `extra_body` - Custom body fields
- `extra_headers` - Custom HTTP headers
- `extra_args` - Direct API parameters

### Configuration Management
- `resolve()` - Settings merge karne ke liye

---

## 💡 Best Practices

1. **Simple Start Karo**: Pehle basic settings (temperature, max_tokens) use karo, phir complexity add karo
2. **Usage Monitor Karo**: Production mai hamesha `include_usage=True` enable rakho cost tracking ke liye
3. **Layer Configuration**: Hierarchical settings ke liye `resolve()` use karo (global → specific)
4. **Determinism Test Karo**: Testing ke liye `temperature=0.0` use karo reproducible results ke liye
5. **Tool Control**: `tool_choice` ke saath explicit raho unexpected tool usage prevent karne ke liye
6. **Context Management**: Development mai `truncation="disabled"` use karo context issues early catch karne ke liye

Ye comprehensive understanding aapko zyada predictable, cost-effective, aur powerful AI agents banane mai madad karega.