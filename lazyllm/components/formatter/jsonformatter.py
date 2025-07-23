import json
from .formatterbase import JsonLikeFormatter
import lazyllm

class JsonFormatter(JsonLikeFormatter):
    """This class is a JSON formatter, that is, the user wants the model to output content is JSON format, and can also select a field in the output content by indexing.


Examples:
    >>> import lazyllm
    >>> from lazyllm.components import JsonFormatter
    >>> toc_prompt='''
    ... You are now an intelligent assistant. Your task is to understand the user's input and convert the outline into a list of nested dictionaries. Each dictionary contains a `title` and a `describe`, where the `title` should clearly indicate the level using Markdown format, and the `describe` is a description and writing guide for that section.
    ... 
    ... Please generate the corresponding list of nested dictionaries based on the following user input:
    ... 
    ... Example output:
    ... [
    ...     {
    ...         "title": "# Level 1 Title",
    ...         "describe": "Please provide a detailed description of the content under this title, offering background information and core viewpoints."
    ...     },
    ...     {
    ...         "title": "## Level 2 Title",
    ...         "describe": "Please provide a detailed description of the content under this title, giving specific details and examples to support the viewpoints of the Level 1 title."
    ...     },
    ...     {
    ...         "title": "### Level 3 Title",
    ...         "describe": "Please provide a detailed description of the content under this title, deeply analyzing and providing more details and data support."
    ...     }
    ... ]
    ... User input is as follows:
    ... '''
    >>> query = "Please help me write an article about the application of artificial intelligence in the medical field."
    >>> m = lazyllm.TrainableModule("internlm2-chat-20b").prompt(toc_prompt).start()
    >>> ret = m(query, max_new_tokens=2048)
    >>> print(f"ret: {ret!r}")  # the model output without specifying a formatter
    'Based on your user input, here is the corresponding list of nested dictionaries:
    [
        {
            "title": "# Application of Artificial Intelligence in the Medical Field",
            "describe": "Please provide a detailed description of the application of artificial intelligence in the medical field, including its benefits, challenges, and future prospects."
        },
        {
            "title": "## AI in Medical Diagnosis",
            "describe": "Please provide a detailed description of how artificial intelligence is used in medical diagnosis, including specific examples of AI-based diagnostic tools and their impact on patient outcomes."
        },
        {
            "title": "### AI in Medical Imaging",
            "describe": "Please provide a detailed description of how artificial intelligence is used in medical imaging, including the advantages of AI-based image analysis and its applications in various medical specialties."
        },
        {
            "title": "### AI in Drug Discovery and Development",
            "describe": "Please provide a detailed description of how artificial intelligence is used in drug discovery and development, including the role of AI in identifying potential drug candidates and streamlining the drug development process."
        },
        {
            "title": "## AI in Medical Research",
            "describe": "Please provide a detailed description of how artificial intelligence is used in medical research, including its applications in genomics, epidemiology, and clinical trials."
        },
        {
            "title": "### AI in Genomics and Precision Medicine",
            "describe": "Please provide a detailed description of how artificial intelligence is used in genomics and precision medicine, including the role of AI in analyzing large-scale genomic data and tailoring treatments to individual patients."
        },
        {
            "title": "### AI in Epidemiology and Public Health",
            "describe": "Please provide a detailed description of how artificial intelligence is used in epidemiology and public health, including its applications in disease surveillance, outbreak prediction, and resource allocation."
        },
        {
            "title": "### AI in Clinical Trials",
            "describe": "Please provide a detailed description of how artificial intelligence is used in clinical trials, including its role in patient recruitment, trial design, and data analysis."
        },
        {
            "title": "## AI in Medical Practice",
            "describe": "Please provide a detailed description of how artificial intelligence is used in medical practice, including its applications in patient monitoring, personalized medicine, and telemedicine."
        },
        {
            "title": "### AI in Patient Monitoring",
            "describe": "Please provide a detailed description of how artificial intelligence is used in patient monitoring, including its role in real-time monitoring of vital signs and early detection of health issues."
        },
        {
            "title": "### AI in Personalized Medicine",
            "describe": "Please provide a detailed description of how artificial intelligence is used in personalized medicine, including its role in analyzing patient data to tailor treatments and predict outcomes."
        },
        {
            "title": "### AI in Telemedicine",
            "describe": "Please provide a detailed description of how artificial intelligence is used in telemedicine, including its applications in remote consultations, virtual diagnoses, and digital health records."
        },
        {
            "title": "## AI in Medical Ethics and Policy",
            "describe": "Please provide a detailed description of the ethical and policy considerations surrounding the use of artificial intelligence in the medical field, including issues related to data privacy, bias, and accountability."
        }
    ]'
    >>> m = lazyllm.TrainableModule("internlm2-chat-20b").formatter(JsonFormatter("[:][title]")).prompt(toc_prompt).start()
    >>> ret = m(query, max_new_tokens=2048)
    >>> print(f"ret: {ret}")  # the model output of the specified formaater
    ['# Application of Artificial Intelligence in the Medical Field', '## AI in Medical Diagnosis', '### AI in Medical Imaging', '### AI in Drug Discovery and Development', '## AI in Medical Research', '### AI in Genomics and Precision Medicine', '### AI in Epidemiology and Public Health', '### AI in Clinical Trials', '## AI in Medical Practice', '### AI in Patient Monitoring', '### AI in Personalized Medicine', '### AI in Telemedicine', '## AI in Medical Ethics and Policy']
    """
    def _extract_json_from_string(self, mixed_str: str):
        json_objects = []
        brace_level = 0
        current_json = ""
        in_string = False

        for char in mixed_str:
            if char == '"' and (len(current_json) == 0 or current_json[-1] != '\\'):
                in_string = not in_string

            if not in_string:
                if char in '{[':
                    if brace_level == 0:
                        current_json = ""
                    brace_level += 1
                elif char in '}]':
                    brace_level -= 1

            if brace_level > 0 or (brace_level == 0 and char in '}]'):
                current_json += char

            if brace_level == 0 and current_json:
                try:
                    json.loads(current_json)
                    json_objects.append(current_json)
                    current_json = ""
                except json.JSONDecodeError:
                    continue

        return json_objects

    def _load(self, msg: str):
        # Convert str to json format
        assert msg.count("{") == msg.count("}"), f"{msg} is not a valid json string."
        try:
            json_strs = self._extract_json_from_string(msg)
            if len(json_strs) == 0:
                raise TypeError(f"{msg} is not a valid json string.")
            res = []
            for json_str in json_strs:
                res.append(json.loads(json_str))
            return res if len(res) > 1 else res[0]
        except Exception as e:
            lazyllm.LOG.info(f"Error: {e}")
            return ""
