import json
from .base_prompt import PromptABC


class RAGContentIdExtractorPrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
        You extract a short "content_identifier" from the input. The input may be:
        - a natural-language question, OR
        - a document/passage (several sentences).

        Rules:
        1. Identify the main subject: person, place, organization, work title, or core topic.
        2. Use a compact phrase (usually 2–12 Chinese characters or 2–6 English words).
        3. Output MUST be valid JSON only, one line, no markdown fences, no explanation:
           {"content_identifier": "<string>"}

        Examples:
        {"content_identifier": "Avatar series"}
        {"content_identifier": "龙美术馆"}
        {"content_identifier": "Enugu"}
        '''

    def build_prompt(self, input) -> str:
        return f'''
        Input (question or passage):
        {input}

        Respond with ONLY: {{"content_identifier": "..."}}
        '''


class RAGFactsConclusionPrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
  # Conclusion Extraction and Relationship Generation Specifications

  ## I. Input/Output Requirements
  **Input**: Any document fragment
  **Output**: JSON array where each element contains `conclusion` and `R` fields

  ## II. Conclusion Extraction Rules
  1. **Atomicity**
      - Each conclusion must be an indivisible basic fact
      - ✖ Prohibited combined conclusions: "A increased by 5% and B decreased by 2%"
        → Should be split into two conclusions

  2. **Verifiability**
      - Must contain at least one definite identifier:
        ✓ Numeric value (59.0%)
        ✓ Time (2025/04/28)
        ✓ Unique name (Humpback65B)
      - ✖ Reject vague expressions: "Performance has improved"

  3. **Timeliness Handling**
      - Explicitly mark time ranges when containing time-sensitive information

  ## III. Relationship (R) Generation Standards
  ### Attribute Requirements
  - **Structured**: Put ALL attributes inside ONE JSON string value for `R`.
  - Inside that string, use semicolons to separate multiple `key=value` pairs.
  - **Wrong (invalid JSON)**: `"R":"birth_date=May 5, 1942";name=Tammy Wynette"`
    (do not put `;name=...` outside the closing quote of `R`).
  - **Correct**: `"R":"birth_date=May 5, 1942;name=Tammy Wynette"`
  - **Operational**: Directly usable for database queries or calculations

  ## IV. Output Specifications (STRICT — follow exactly)
  - The model response MUST be one JSON **array** at the top level:
    `[{"conclusion": "<string>", "R": "<string>"}, ...]`
  - Every item MUST include both `"conclusion"` and `"R"` (string values).
  - The `R` field MUST be a single JSON string. All `key=value` pairs MUST appear
    inside that string, separated by `;` if needed.
  - Even a single atomic conclusion MUST be expressed as a **one-element array**,
    not as a single JSON object. Example: `[{"conclusion":"...","R":"..."}]`.
  - Do NOT wrap the array in an outer object (no `{"conclusions": [...]}` unless
    the caller explicitly asks for that legacy shape).
  - Do NOT output markdown fences, comments, or any text before or after the array.
  - **Stop generation immediately after the closing `]` of the array.** Do not continue
    with chat tokens (e.g. `<|im_start|>`), explanations, other languages, or repeated filler.
  - At most **15** array elements; prefer fewer, higher-quality atomic facts.
        '''

    def build_prompt(self, input) -> str:
        return f'''
    The document content to be processed is as follows: {input}

    Respond with ONLY a JSON array as specified in the system message. End your reply right after `]`.
    '''


class RAGTaskToQuestionPrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''Your task is to generate a corresponding question (Q) based on the given task identifier (ID), \
relationship (R), and answer (A).

  Input/Output Specifications:
  Input:
  - ID: Data source or query scope
  - R: Logical relationship for extracting the answer from the data
  - A: Known correct answer

  Output:
  - Must be in strict JSON format: {"Q": "generated question"}
  - No explanations or extra fields allowed

  Only output JSON without additional content.
  '''

    def build_prompt(self, identifier, conclusion, relation) -> str:
        return f'''
        Data to be Processed:
        ID: {identifier}
        R: {relation}
        A: {conclusion}
        '''


class RAGQARefinementPrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''Processing Rules:
  1. Extract ONLY the exact information requested in the question
  2. Preserve the original index numbering
  3. Never omit essential information
  4. Standardize all numerical formats

  Required JSON format:
  {
      "question": str,
      "original_answer": str,
      "refined_answer": str
  }
  '''

    def build_prompt(self, input) -> str:
        return f'''
            The data need to be processed is as follows: {input}
        '''


class RAGTaskSolverPrompt(PromptABC):
    def __init__(self):
        pass

    def build_prompt(self, input) -> str:
        return f'''Please solve the following problem and return as many relevant results as possible \
that meet the query requirements.
 Ensure responses are as concise as possible, focusing only on key information while omitting redundant details.
 The task is:
 {input}
        '''.strip()


class RAGConsistencyScoringPrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
Evaluate the consistency of the core content of the golden answer and the other answer
  # Scoring Criteria
    1) 2 points: the information between the golden answer and the other answer completely consistent
    2) 1 point: the other answer contains all the information of the golden answer but has additional valid information
    3) 0 point: the other answer lacks the necessary key information

  # the output should be in JSON format
  {
    "answer_analysis":"give out the reason on how to score the llm_answer",
    "answer_score":0/1/2
  }
'''

    def build_prompt(self, golden_answer, llm_answer) -> str:
        return f'''
    The inputs are as follows:
    Golden Answer: {golden_answer}
    Other Answer: {llm_answer}
        '''


class RAGAnswerVariantsPrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
  You are an expert in **linguistic variation** and **data augmentation**. Your task is to generate \
alternative expressions, formats, and aliases for a given input entity.

  **Key Guidelines:**
  1. **Equivalence:** Each alternative expression must refer to *exactly the same entity*
  2. **Scope of Variation:** Focus on formatting conventions, abbreviations, aliases
  3. **Inclusion of Original:** Always include the original input as the first item
  4. **Format:** Output **only** a JSON array of strings. No markdown, no code fences, no text before/after.
  5. **Hard limit:** At most **20** array elements. Do not repeat the same spelling; do not pad with near-duplicates.
  6. **Valid JSON (critical):** Each element is one JSON string. **Never put an unescaped double-quote character
     inside a string value.** If a name or nickname would need quotes, rephrase without quotes (e.g. use *Vova*
     without ASCII `"`), or use apostrophes `'` inside the text, or omit the nickname.
  7. Avoid ambiguous `+` or stray punctuation right before a closing `"`; keep each string one coherent phrase.
  8. **Stop rule:** After the closing `]` of the array, output nothing else—no chat tokens, no repetition, \
no other language, no `<|...|>` markers.
  9. Prefer **shorter** variants when the answer is long so the full array fits without truncation.
        '''

    def build_prompt(self, answer) -> str:
        return f'''
    The original answer is: {answer}
    List plausible textual expressions that refer to the same entity (same meaning).
    Respond with ONLY a JSON array of strings (max 20 items), no explanation. Ensure valid JSON: no broken strings, \
no unescaped `"` inside values, array must end with `]` and then stop.
        '''


class RAGDocGroundedAnswerPrompt(PromptABC):
    def __init__(self):
        pass

    def build_prompt(self, golden_doc, question) -> str:
        return f'''You are given the following document that contains relevant information to help answer a question.
Document:
\"\"\"
{golden_doc}
\"\"\"
Question:
{question}
Please answer the question using ONLY the information in the provided document. \
Return the final answer directly, with no explanation.
        '''


class RAGDepthQueryIdPrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
        You need to extract the content_identifier from question. Here's how:
  1. For each question, identify the main subject/noun phrase that the question is about
  2. This should typically be:
    - Proper nouns (names, titles)
    - Specific technical terms
    - Unique identifiers in the question

  Return a single JSON object with exactly one key "content_identifier". The value must be one plain \
string (the subject phrase), not an array, object, or nested structure.
'''

    def build_prompt(self, input) -> str:
        return f'''
        Now process this question:{input}
        '''


class RAGDepthBackwardSupersetPrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
You must output **only** one JSON object with exactly two keys:
  "identifier": string (superset or broader concept),
  "relation": string (how the input relates to that superset).

Rules:
- No markdown, no ``` fences, no preamble or postscript.
- No text in other languages mixed outside the JSON.
- Do not repeat the JSON; output it once and stop.
        '''

    def build_prompt(self, input) -> str:
        return f'''
        Conduct divergent searches based on the input element to find an appropriate superset related to \
its attributes.

  Current input:
  {input}

  Return the JSON object as specified in the system message.
        '''


class RAGDepthSupersetValidationPrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
**Task**: Validate if a given "superset" can uniquely identify a "subset" based on the provided "relationship".

  **Output Format**:
  Return a JSON with the key `new_query`. The value should be:
  - `"valid"` if the superset and relationship can uniquely locate the subset.
  - `"invalid"` otherwise.
'''

    def build_prompt(self, new_id, relation, identifier) -> str:
        return f'''
Given superset: {new_id}
Given relationship: {relation}
Given subset: {identifier}
'''


class RAGDepthQuestionFromContextPrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
  Please generate a question based on the content of the input identifier, a certain answer, and a certain relationship.
  Please return it in JSON format, with the key of the JSON being new_query.
'''

    def build_prompt(self, new_id, relation, identifier) -> str:
        return f'''
                Certain answer: {identifier}
                Identifier: {new_id}
                Relationship: {relation}
'''


class RAGDepthSolverPrompt(PromptABC):
    def __init__(self):
        pass

    def build_prompt(self, input) -> str:
        return f'''
Please solve the following problem and return as many relevant results as possible.
Please return the result in JSON format with keys 'answer_list': List[str] the list of answers.
The task is:
{input}
        '''.strip()


class RAGDepthConsistencyScoringPrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
Evaluate the consistency of the core content of the golden answer and the other answer
  # Scoring Criteria
    1) 2 points: the information completely consistent
    2) 1 point: the other answer contains all the information but has additional valid information
    3) 0 point: the other answer lacks the necessary key information

  # the output should be in JSON format
  {
    "answer_analysis":"give out the reason",
    "answer_score":0/1/2
  }
'''

    def build_prompt(self, golden_answer, llm_answer) -> str:
        return f'''
    The inputs are as follows:
    Golden Answer: {golden_answer}
    Other Answer: {llm_answer}
        '''


class RAGWidthQuestionSynthesisPrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
        # Comprehensive Task Guide for Research Questions

  ## Core Objective:
  Intelligently merge 2-3 related research questions into high-quality comprehensive questions.

  ## Output Specifications:
  [
    {
      "question": "Text of the synthesized question",
      "index": [1,2,3], // Original indices
      "content_identifier": "Original content identifier"
    }
  ]
        '''

    def build_prompt(self, input) -> str:
        return f'''
        Here are the base questions to process:
    {json.dumps(input, indent=2, ensure_ascii=False)}
'''


class RAGWidthDecompositionCheckPrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
    Task Instructions:
  Verify if complex questions can be properly decomposed into their original questions.
  Return state=1 if all conditions are met, state=0 otherwise.

  Example Output:
  [{
      "index": 1,
      "complex_question": "original complex question",
      "state": 1
  }]
'''

    def build_prompt(self, input) -> str:
        return f'''
    Here are the base questions to process:
    {json.dumps(input, indent=2, ensure_ascii=False)}
'''


class RAGWidthVerificationPrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
  Answer the provided complex research questions based on your knowledge.

  Output JSON format:
  [{
  "index": 1
  "complex_question": original complex question,
  "llm_answer": your answer
  }]
'''

    def build_prompt(self, input) -> str:
        return f'''
    Please answer these research questions:
    {json.dumps(input, indent=2, ensure_ascii=False)}
'''


class RAGWidthSolverPrompt(PromptABC):
    def __init__(self):
        pass

    def build_prompt(self, input) -> str:
        return f'''
Please solve the following problem and return as many relevant results as possible.
Please return the result in JSON format with keys 'answer_list': List[str] the list of answers.
The task is:
{input}
        '''.strip()


class RAGWidthConsistencyScoringPrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
Evaluate the consistency of the core content of the golden answer and the other answer
  # Scoring Criteria
    1) 2 points: completely consistent
    2) 1 point: contains all info but has additional valid information
    3) 0 point: lacks the necessary key information

  # the output should be in JSON format
  {
    "answer_analysis":"reason",
    "answer_score":0/1/2
  }
'''

    def build_prompt(self, golden_answer, llm_answer) -> str:
        return f'''
    The inputs are as follows:
    Golden Answer: {golden_answer}
    Other Answer: {llm_answer}
        '''


__all__ = [
    'RAGContentIdExtractorPrompt',
    'RAGFactsConclusionPrompt',
    'RAGTaskToQuestionPrompt',
    'RAGQARefinementPrompt',
    'RAGTaskSolverPrompt',
    'RAGConsistencyScoringPrompt',
    'RAGAnswerVariantsPrompt',
    'RAGDocGroundedAnswerPrompt',
    'RAGDepthQueryIdPrompt',
    'RAGDepthBackwardSupersetPrompt',
    'RAGDepthSupersetValidationPrompt',
    'RAGDepthQuestionFromContextPrompt',
    'RAGDepthSolverPrompt',
    'RAGDepthConsistencyScoringPrompt',
    'RAGWidthQuestionSynthesisPrompt',
    'RAGWidthDecompositionCheckPrompt',
    'RAGWidthVerificationPrompt',
    'RAGWidthSolverPrompt',
    'RAGWidthConsistencyScoringPrompt',
]
