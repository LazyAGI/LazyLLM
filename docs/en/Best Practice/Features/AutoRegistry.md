# Inheritance-as-Registration Specification

## 1. What Is the Inheritance-as-Registration Mechanism

### 1.1 Mechanism Definition

LazyLLM widely adopts the **inheritance-based auto-registration** mechanism, which follows two core rules:

- **Inheritance determines capability ownership**
    The Base class you inherit determines the capability group you belong to.

- **Naming conventions determine access entry**
    When the class name meets the convention, the system parses a key from the class name and generates a stable access path.

Therefore, modules are automatically incorporated into LazyLLM’s capability system at the **class definition stage**.
Developers do not need to call `register()` explicitly, nor maintain a centralized registry.

### 1.2 What Registration Automatically Completes

When a class meets the registration conditions, LazyLLM automatically completes three categories of work:

1. **Capability group registration (Group / Type)**
    Create or extend the corresponding capability namespace node.

2. **Implementation binding**
    Attach the implementation class to the group and provide a unified access entry.

3. **Configuration key declaration (Configuration Keys)**
    Based on capability and supplier information, automatically declare common configuration keys (such as API keys, model names, etc.).

## 2. How the Mechanism Is Used in LazyLLM

LazyLLM widely adopts the **inheritance-based auto-registration** mechanism to manage and extend core components.
As one of LazyLLM’s foundational design patterns, it runs through the following module systems:

- Launcher (startup and runners)

- Flow (pipelines and orchestration nodes)

- Tools (tools and capability components)

- Module (model and service modules, such as Online LLM)

In LazyLLM, these modules share the following characteristics:

- Many types, frequent extensions

- Need for a unified access entry

- No desire for developers to manually maintain a registry

Therefore, LazyLLM consistently adopts:

> Auto-registration at class definition time via class inheritance + naming conventions

Developers only need to define the class itself, without calling `register()` or modifying central configuration files.

## 3. Inheritance-as-Registration in Online Modules

Among the various modules in LazyLLM, Online Modules are a typical application scenario of inheritance-based auto-registration.

Their characteristics include:

- Many capability types (Chat / STT / TTS / Embed / Rerank / Image, etc.)

- Many suppliers (OpenAI / Qwen / Doubao / SenseNova / ...)

- Highly repetitive configuration keys (API Key / Model Name, etc.)

To reduce extension cost and unify access, LazyLLM fully implements inheritance-based auto-registration in Online modules.
The following sections use Online modules as an example to illustrate how the mechanism works.

### 3.1 What “Registration” Means in Online Modules

In the context of Online modules, “registration” not only means the class is recognized by the system, but also completes the following:

1. Capability group (Group / Type) registration

    Based on the inherited Online Base class, the module is registered under the corresponding capability group, for example:

    - `lazyllm.online.chat`

    - `lazyllm.online.stt`

    - `lazyllm.online.tts`

    This grouping is determined by the Base class hierarchy rather than explicit declarations in the implementation class.

2. Supplier class registration

    Based on class naming rules, the supplier identifier is parsed from the class name and the implementation class is attached to the corresponding capability group, for example:

    - `lazyllm.online.chat.doubao`

    - `lazyllm.online.stt.sensenova`

    This provides a unified and stable access entry for users.

3. Configuration key declaration (Configuration Keys)

    During registration, LazyLLM automatically declares common configuration keys based on supplier and capability type, for example:

    - `{supplier}_api_key`

    - `{supplier}_model_name`

    - `{supplier}_{capability}_model_name`

    Developers typically do not need to declare these configuration keys explicitly; just read them in the implementation class according to the convention.

## 4. Extending LazyLLM Online Classes

### 4.1 Capability Types and Base Class Selection

LazyLLM differentiates module capability types via the Base class hierarchy. When extending a module, you must inherit the Base class corresponding to the capability type.

Typical capability types include (examples):

| Capability Type  | Corresponding Base Class             |
|------------------|--------------------------------------|
| Chat             | OnlineChatModuleBase                 |
| Embed            | LazyLLMOnlineEmbedModuleBase         |
| Rerank           | LazyLLMOnlineRerankModuleBase        |
| STT              | LazyLLMOnlineSTTModuleBase           |
| TTS              | LazyLLMOnlineTTSModuleBase           |
| Text-to-Image    | LazyLLMOnlineText2ImageModuleBase    |
| Image Editing    | LazyLLMOnlineImageEditingModuleBase  |

> 💡 Rule: the Base class a module inherits determines which capability group it is registered to.

### 4.2 Class Naming Convention (Strict)

Module class names must follow the naming rule below:

```
<SupplierName><TypeSuffix>
```

Where:

- SupplierName: supplier name (e.g., Doubao, SenseNova)
- TypeSuffix: capability type suffix, consistent with the inherited Base class (e.g., Chat, STT, TTS)

Correct examples:

- DoubaoChat
- SenseNovaSTT
- QwenTextToImage

Incorrect examples (will cause registration to fail):

- Class name does not end with a capability suffix, e.g., `DoubaoModule`
- Class name is inconsistent with the inherited Base type

## 5. Existing Online Module Architecture

### 5.1 Online Module Inheritance Hierarchy

As shown in the figure, LazyLLM Online modules adopt a layered inheritance structure:

- Top-level Base class: LazyLLMOnlineBase class, used to define the unified Online namespace (`lazyllm.online`)

- Capability Base classes
    - Capability families such as `LazyLLMOnlineChatModuleBase`, `OnlineEmbeddingModuleBase`, and `OnlineMultiModalBase`, which serve as base classes for major capability groups. Among them, `LazyLLMOnlineChatModuleBase` defines the `lazyllm.online.chat` group, while the other two classes skip group registration, letting capability subclasses register specific capability tags.

    - Capability subclasses such as `LazyLLMOnlineRerankModuleBase`, `LazyLLMOnlineTTSModuleBase`, and `LazyLLMOnlineText2ImageModuleBase`. These define specific capability groups such as `lazyllm.online.rerank`, `lazyllm.online.tts`, etc.

- Supplier classes: concrete service implementation classes, following the class naming requirements in [Section 2.2](#22-class-naming-convention-strict).

![auto_registry.png](../../../assets/auto_registry.png)

### 5.2 Registration Results and Access

After registration, modules can be accessed as follows:

```python
import lazyllm

doubao_chat_cls = lazyllm.online.chat.doubao(**kwargs)
sensenova_stt_cls = lazyllm.online.stt.sensenova(**kwargs)
```

The access path remains stable for users and does not depend on the module’s implementation path.

## 6. Extension and Customization Rules

### 6.1 General Configuration and Capability Configuration

LazyLLM distinguishes two types of configuration keys:

1. Supplier-level configuration

    - e.g., `{supplier}_api_key`

    - Independent of capability type. When a supplier’s class is registered for the first time, LazyLLM adds the corresponding API Key configuration for that supplier.

2. Capability-level configuration

    - e.g., `{supplier}_stt_model_name`

    - Only present for the corresponding capability type; provides a model name configuration for a supplier class of that capability

> Note: configuration key declaration is automatically handled by LazyLLM during registration. When extending a supplier class, you typically do not need to declare configuration explicitly unless the supplier has extra, specific requirements.

### 6.2 Basic Rules for Extending Supplier Classes

In most cases, extending a new Online supplier class requires only three steps:

- Step 1: choose the capability type and inherit the corresponding Base class

    Choose and inherit the Online Base class based on the capability implemented, for example:

    ```python
    class MyProviderChat(OnlineChatModuleBase):
        ...
    ```

    This inheritance determines that the class is registered to:

    ```bash
    lazyllm.online.chat
    ```

- Step 2: define the class name according to the naming convention

    As stated in [Section 2.2](#22-class-naming-convention-strict), the class name must follow:

    ```
    <SupplierName><TypeSuffix>
    ```

    For example:
    - `MyProviderChat`
    - `MyProviderSTT`

    LazyLLM will automatically parse the supplier identifier from the class name and generate the corresponding access entry:

    ```bash
    lazyllm.online.chat.myprovider(...)
    ```

- Step 3: implement the supplier’s own logic

    Implement initialization and call logic in the class, such as client creation and request wrapping.

    ```python
    class MyProviderChat(OnlineChatModuleBase):
        def __init__(self, api_key: str, base_url: str = "..."):
            ...
    ```

    After completing the steps above, the supplier class will participate in auto-registration with no extra operations required.

### 6.3 Extending Supplier Subclasses for Multiple Capabilities

When the same supplier needs to support multiple capabilities (such as Chat, STT, Embedding), it is recommended to organize shared logic via a supplier-specific Base class.

For example:

```python
class _MyProviderBase:
    def __init__(self, api_key: str, base_url: str):
        ...
```

Each capability implementation class inherits both the corresponding Online Base and the supplier Base:

```python
class MyProviderChat(OnlineChatModuleBase, _MyProviderBase):
    ...

class MyProviderSTT(LazyLLMOnlineSTTModuleBase, _MyProviderBase):
    ...
```

During registration, LazyLLM will automatically generate common configuration keys based on supplier and capability type, including but not limited to:

- `{supplier}_api_key`
- `{supplier}_model_name`
- `{supplier}_{capability}_model_name`

When extending a supplier class, you typically do not need to declare these configuration keys explicitly; just read them during initialization according to the convention.
