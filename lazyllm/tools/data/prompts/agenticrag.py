'''Prompts for AgenticRAG pipeline operators'''
import json
from .base_prompt import PromptABC


class RAGContentIdExtractorPrompt(PromptABC):
    '''Extract content identifier from question for RAG atomic pipeline.'''
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

  Examples:
  {
      "question": "What is the third movie in the Avatar series?",
      "content_identifier": "Avatar series"
  },
  {
      "question": "龙美术馆2025年展览展览时间范围是什么",
      "content_identifier": "龙美术馆"
  }

  Return JSON format with key "content_identifier"
'''

    def build_prompt(self, input) -> str:
        return f'''
        Now process this question:{input}
        '''


class RAGFactsConclusionPrompt(PromptABC):
    '''Extract atomic facts and relationships (conclusion + R) from document.'''
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
  - **Structured**: Use semicolons to separate multi-metrics
  - **Operational**: Directly usable for database queries or calculations

  ## IV. Output Specifications
  Return JSON array with "conclusion" and "R" fields for each item.
        '''

    def build_prompt(self, input) -> str:
        return f'''
    The document content to be processed is as follows: {input}
    '''


class RAGTaskToQuestionPrompt(PromptABC):
    '''Generate question from task identifier, relation and conclusion.'''
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
    '''Refine and standardize QA pair output format.'''
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
    '''Solver-style prompt for LLM to answer the task question.'''
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
    '''Score consistency between golden answer and model answer.'''
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
    '''Generate alternative phrasings for an answer (data augmentation).'''
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return '''
  You are an expert in **linguistic variation** and **data augmentation**. Your task is to generate a \
comprehensive list of all plausible and commonly recognized alternative expressions, formats, and aliases \
for a given input entity.

  **Key Guidelines:**
  1. **Equivalence:** Each alternative expression must refer to *exactly the same entity*
  2. **Scope of Variation:** Focus on formatting conventions, abbreviations, aliases
  3. **Inclusion of Original:** Always include the original input as the first item
  5. **Format:** Output the variations as a JSON list of strings.
        '''

    def build_prompt(self, answer) -> str:
        return f'''
    The original answer is: {answer}
    Please list all possible textual expressions that have the same meaning or refer to the same entity.
    Respond with a JSON list of strings. Do not explain.
        '''


class RAGDocGroundedAnswerPrompt(PromptABC):
    '''Answer question using only the provided document context.'''
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
    '''Extract query/content identifier for depth QA pipeline.'''
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

  Return JSON format with key "content_identifier"
'''

    def build_prompt(self, input) -> str:
        return f'''
        Now process this question:{input}
        '''


class RAGDepthBackwardSupersetPrompt(PromptABC):
    '''Find superset and relation via backward search from identifier.'''
    def __init__(self):
        pass

    def build_prompt(self, input) -> str:
        return f'''
        Conduct divergent searches based on the input element to find an appropriate superset related to \
its attributes.

  Return format requirements: Please return the result in JSON format with keys 'identifier': str \
(identifier) and 'relation': str (relationship).

  Current input:
  {input}
        '''


class RAGDepthSupersetValidationPrompt(PromptABC):
    '''Validate if superset+relation uniquely identify the subset.'''
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
    '''Generate question from identifier, relation and answer context.'''
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
    '''Solver prompt for depth QA; output answer_list in JSON.'''
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
    '''Score answer consistency for depth QA recall.'''
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
    '''Synthesize comprehensive question from multiple related questions.'''
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
    '''Check if complex question decomposes to original sub-questions.'''
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
    '''Verify and answer width/complex research questions.'''
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
    '''Solver prompt for width QA; output answer_list in JSON.'''
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
    '''Score answer consistency for width QA.'''
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
