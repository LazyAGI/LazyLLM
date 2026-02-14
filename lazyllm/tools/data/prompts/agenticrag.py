import json
from .base_prompt import PromptABC


class AtomicTaskGeneratorGetIdentifierPrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
        Your task is to identify the content_identifier from the given question. Follow these steps:
  1. For each question, determine the primary subject or noun phrase that the question addresses
  2. This typically includes:
    - Proper nouns (personal names, titles)
    - Domain-specific technical terminology
    - Distinct identifiers present in the question

  Examples:
  {
      "question": "What is the third movie in the Avatar series?",
      "content_identifier": "Avatar series"
  },
  {
      "question": "龙美术馆2025年展览展览时间范围是什么",
      "content_identifier": "龙美术馆"
  }

  Output JSON format with key "content_identifier"
'''

    def build_prompt(self, input) -> str:
        return f'''
        Now process this question:{input}
        '''


class AtomicTaskGeneratorGetConclusionPrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
  # Conclusion Identification and Relationship Construction Guidelines

  ## I. Input/Output Specifications
  **Input**: Any document fragment
  **Output**: JSON array where each element includes `conclusion` and `R` fields

  ## II. Conclusion Identification Principles
  1. **Atomicity**
      - Every conclusion should represent an indivisible fundamental fact
      - ✖ Forbidden combined conclusions: "A increased by 5% and B decreased by 2%"
        → Must be divided into separate conclusions

  2. **Verifiability**
      - Should include at least one specific identifier:
        ✓ Numeric value (59.0%)
        ✓ Time reference (2025/04/28)
        ✓ Unique name (Humpback65B)
      - ✖ Exclude ambiguous expressions: "Performance has improved"

  3. **Temporal Information Management**
      - Clearly indicate time ranges when dealing with time-sensitive data

  ## III. Relationship (R) Construction Criteria
  ### Attribute Standards
  - **Structured**: Employ semicolons to separate multiple metrics
  - **Operational**: Ready for direct use in database queries or computations

  ## IV. Output Format
  Output JSON array with "conclusion" and "R" fields for each item.
        '''

    def build_prompt(self, input) -> str:
        return f'''
    Please process the following document content: {input}
    '''


class AtomicTaskGeneratorQuestionPrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''Your assignment is to create a corresponding question (Q) using the provided task identifier (ID), \
relationship (R), and answer (A).

  Input/Output Format:
  Input:
  - ID: Data source or query scope
  - R: Logical relationship for extracting the answer from the data
  - A: Known correct answer

  Output:
  - Must follow strict JSON format: {"Q": "generated question"}
  - Explanations or additional fields are not permitted

  Output only JSON without any supplementary content.
  '''

    def build_prompt(self, identifier, conclusion, relation) -> str:
        return f'''
        Please process the following data:
        ID: {identifier}
        R: {relation}
        A: {conclusion}
        '''


class AtomicTaskGeneratorCleanQAPrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''Processing Guidelines:
  1. Extract EXCLUSIVELY the precise information specified in the question
  2. Maintain the original index numbering
  3. Do not exclude any critical information
  4. Normalize all numerical formats

  Mandatory JSON format:
  {
      "question": str,
      "original_answer": str,
      "refined_answer": str
  }
  '''

    def build_prompt(self, input) -> str:
        return f'''
            Please process the following data: {input}
        '''


class AtomicTaskGeneratorAnswerPrompt(PromptABC):
    def __init__(self):
        pass

    def build_prompt(self, input) -> str:
        return f'''Please address the following problem and provide as many relevant results as possible \
that satisfy the query requirements.
 Ensure responses are as brief as possible, concentrating solely on essential information while excluding unnecessary details.
 The assignment is:
 {input}
        '''.strip()


class AtomicTaskGeneratorRecallScorePrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
Assess the alignment of core content between the golden answer and the other answer
  # Scoring Standards
    1) 2 points: the information in the golden answer and the other answer is fully aligned
    2) 1 point: the other answer includes all information from the golden answer but contains extra valid information
    3) 0 point: the other answer is missing essential key information

  # the output must be in JSON format
  {
    "answer_analysis":"provide the rationale for scoring the llm_answer",
    "answer_score":0/1/2
  }
'''

    def build_prompt(self, golden_answer, llm_answer) -> str:
        return f'''
    Please evaluate the following inputs:
    Golden Answer: {golden_answer}
    Other Answer: {llm_answer}
        '''


class AtomicTaskGeneratorOptionalAnswerPrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
  You are a specialist in **linguistic variation** and **data augmentation**. Your assignment is to create a \
comprehensive list of all reasonable and widely accepted alternative expressions, formats, and aliases \
for a provided input entity.

  **Essential Guidelines:**
  1. **Equivalence:** Every alternative expression must refer to *precisely the same entity*
  2. **Variation Scope:** Concentrate on formatting conventions, abbreviations, aliases
  3. **Original Inclusion:** Always include the original input as the first item
  5. **Format:** Output the variations as a JSON list of strings.
        '''

    def build_prompt(self, answer) -> str:
        return f'''
    The original answer is: {answer}
    Please enumerate all possible textual expressions that share the same meaning or reference the same entity.
    Respond with a JSON list of strings. Do not provide explanations.
        '''


class AtomicTaskGeneratorGoldenDocAnswerPrompt(PromptABC):
    def __init__(self):
        pass

    def build_prompt(self, golden_doc, question) -> str:
        return f'''You are provided with the following document that includes relevant information to assist in answering a question.
Document:
\"\"\"
{golden_doc}
\"\"\"
Question:
{question}
Please answer the question using EXCLUSIVELY the information in the provided document. \
Output the final answer directly, without any explanation.
        '''


class DepthQAGeneratorGetIdentifierPrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
        Your task is to identify the content_identifier from the question. Follow these steps:
  1. For each question, determine the primary subject or noun phrase that the question addresses
  2. This typically includes:
    - Proper nouns (personal names, titles)
    - Domain-specific technical terminology
    - Distinct identifiers present in the question

  Output JSON format with key "content_identifier"
'''

    def build_prompt(self, input) -> str:
        return f'''
        Now process this question:{input}
        '''


class DepthQAGeneratorBackwardTaskPrompt(PromptABC):
    def __init__(self):
        pass

    def build_prompt(self, input) -> str:
        return f'''
        Perform exploratory searches based on the input element to locate a suitable superset connected to \
its attributes.

  Output format requirements: Please output the result in JSON format with keys 'identifier': str \
(identifier) and 'relation': str (relationship).

  Current input:
  {input}
        '''


class DepthQAGeneratorSupersetCheckPrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
**Assignment**: Verify whether a given "superset" can uniquely identify a "subset" using the provided "relationship".

  **Output Format**:
  Output a JSON with the key `new_query`. The value should be:
  - `"valid"` if the superset and relationship can uniquely locate the subset.
  - `"invalid"` otherwise.
'''

    def build_prompt(self, new_id, relation, identifier) -> str:
        return f'''
Provided superset: {new_id}
Provided relationship: {relation}
Provided subset: {identifier}
'''


class DepthQAGeneratorQuestionPrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
  Please create a question using the content of the input identifier, a specific answer, and a specific relationship.
  Please output it in JSON format, with the key of the JSON being new_query.
'''

    def build_prompt(self, new_id, relation, identifier) -> str:
        return f'''
                Specific answer: {identifier}
                Identifier: {new_id}
                Relationship: {relation}
'''


class DepthQAGeneratorAnswerPrompt(PromptABC):
    def __init__(self):
        pass

    def build_prompt(self, input) -> str:
        return f'''
Please solve the following problem and return as many relevant results as possible.
Please return the result in JSON format with keys 'answer_list': List[str] the list of answers.
The task is:
{input}
        '''.strip()


class DepthQAGeneratorRecallScorePrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
Assess the alignment of core content between the golden answer and the other answer
  # Scoring Standards
    1) 2 points: the information is fully aligned
    2) 1 point: the other answer includes all information but contains extra valid information
    3) 0 point: the other answer is missing essential key information

  # the output must be in JSON format
  {
    "answer_analysis":"provide the rationale",
    "answer_score":0/1/2
  }
'''

    def build_prompt(self, golden_answer, llm_answer) -> str:
        return f'''
    Please evaluate the following inputs:
    Golden Answer: {golden_answer}
    Other Answer: {llm_answer}
        '''


class WidthQAGeneratorMergePrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
        # Complete Task Guide for Research Questions

  ## Primary Goal:
  Intelligently combine 2-3 related research questions into high-quality comprehensive questions.

  ## Output Format:
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
        Please process the following base questions:
    {json.dumps(input, indent=2, ensure_ascii=False)}
'''


class WidthQAGeneratorOriginCheckPrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
    Assignment Instructions:
  Check whether complex questions can be correctly decomposed into their original questions.
  Output state=1 if all conditions are satisfied, state=0 otherwise.

  Example Output:
  [{
      "index": 1,
      "complex_question": "original complex question",
      "state": 1
  }]
'''

    def build_prompt(self, input) -> str:
        return f'''
    Please process the following base questions:
    {json.dumps(input, indent=2, ensure_ascii=False)}
'''


class WidthQAGeneratorQuestionVerifyPrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
  Address the provided complex research questions using your knowledge.

  Output JSON format:
  [{
  "index": 1
  "complex_question": original complex question,
  "llm_answer": your answer
  }]
'''

    def build_prompt(self, input) -> str:
        return f'''
    Please address these research questions:
    {json.dumps(input, indent=2, ensure_ascii=False)}
'''


class WidthQAGeneratorAnswerPrompt(PromptABC):
    def __init__(self):
        pass

    def build_prompt(self, input) -> str:
        return f'''
Please solve the following problem and return as many relevant results as possible.
Please return the result in JSON format with keys 'answer_list': List[str] the list of answers.
The task is:
{input}
        '''.strip()


class WidthQAGeneratorRecallScorePrompt(PromptABC):
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
Assess the alignment of core content between the golden answer and the other answer
  # Scoring Standards
    1) 2 points: fully aligned
    2) 1 point: includes all info but contains extra valid information
    3) 0 point: missing essential key information

  # the output must be in JSON format
  {
    "answer_analysis":"rationale",
    "answer_score":0/1/2
  }
'''

    def build_prompt(self, golden_answer, llm_answer) -> str:
        return f'''
    Please evaluate the following inputs:
    Golden Answer: {golden_answer}
    Other Answer: {llm_answer}
        '''


__all__ = [
    'AtomicTaskGeneratorGetIdentifierPrompt',
    'AtomicTaskGeneratorGetConclusionPrompt',
    'AtomicTaskGeneratorQuestionPrompt',
    'AtomicTaskGeneratorCleanQAPrompt',
    'AtomicTaskGeneratorAnswerPrompt',
    'AtomicTaskGeneratorRecallScorePrompt',
    'AtomicTaskGeneratorOptionalAnswerPrompt',
    'AtomicTaskGeneratorGoldenDocAnswerPrompt',
    'DepthQAGeneratorGetIdentifierPrompt',
    'DepthQAGeneratorBackwardTaskPrompt',
    'DepthQAGeneratorSupersetCheckPrompt',
    'DepthQAGeneratorQuestionPrompt',
    'DepthQAGeneratorAnswerPrompt',
    'DepthQAGeneratorRecallScorePrompt',
    'WidthQAGeneratorMergePrompt',
    'WidthQAGeneratorOriginCheckPrompt',
    'WidthQAGeneratorQuestionVerifyPrompt',
    'WidthQAGeneratorAnswerPrompt',
    'WidthQAGeneratorRecallScorePrompt',
]
