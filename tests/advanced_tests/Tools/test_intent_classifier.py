from lazyllm.tools import IntentClassifier
import lazyllm
from lazyllm.launcher import cleanup


class TestIntentClassifier(object):
    @classmethod
    def setup_class(cls):
        cls._llm = lazyllm.TrainableModule('internlm2-chat-7b').deploy_method(lazyllm.deploy.vllm)

    @classmethod
    def teardown_class(cls):
        cleanup()

    def test_intent_classifier(self):
        intent_list = ['Chat', 'Financial Knowledge Q&A', 'Employee Information Query', 'Weather Query']
        ic = IntentClassifier(self._llm, intent_list)
        ic.start()
        assert ic('What is the weather today') == 'Weather Query'
        assert ic('Who are you') == 'Chat'
        assert ic('What is the difference between stocks and funds') == 'Financial Knowledge Q&A'
        assert ic('Check the work location of Macro in the Technology Department') == 'Employee Information Query'

    def test_intent_classifier_example(self):
        intent_list = ['Chat', 'Financial Knowledge Q&A', 'Employee Information Query', 'Weather Query']
        ic = IntentClassifier(self._llm, intent_list, examples=[
            ['Who are you', 'Chat'], ['What is the weather today', 'Weather Query']])
        ic.start()
        assert ic('What is the weather today') == 'Weather Query'
        assert ic('Who are you') == 'Chat'
        assert ic('What is the difference between stocks and funds') == 'Financial Knowledge Q&A'
        assert ic('Check the work location of Macro in the Technology Department') == 'Employee Information Query'

    def test_intent_classifier_prompt_and_constrain(self):
        intent_list = ['Chat', 'Image Question and Answer', 'Music', 'Weather Query']
        prompt = ('If the input contains attachments, the intent is determined with the highest priority based on the '
                  'suffix type of the attachments: If it is an image suffix such as .jpg, .png, etc., then the output '
                  'is: Image Question and Answer. If the audio suffix is .mp3, .wav, etc., the output is: Music')
        examples = [['Hello world. <attachments>hello.jpg</attachments>', 'Image Question and Answer'],
                    ['Happy lazyllm. <attachments>hello.wav</attachments>', 'Music']]
        attention = ('Intent is determined with the highest priority based on the suffix type of the attachments '
                     'provideded by <attachments>')
        ic = IntentClassifier(self._llm, intent_list, prompt=prompt, attention=attention, examples=examples,
                              constrain='intents outside the given intent list is not allowed')
        ic.start()
        assert ic('What is the weather today') == 'Weather Query'
        assert ic('Who are you?') == 'Chat'
        assert ic('Who are you picture<attachments>who.png</attachments>') == 'Image Question and Answer'
        assert ic('Song of weather <attachments>weather.mp3</attachments>') == 'Music'

    def test_intent_classifier_enter(self):
        with IntentClassifier(self._llm) as ic:
            ic.case['Weather Query', lambda x: '38.5°C']
            ic.case['Chat', lambda x: 'permission denied']
            ic.case['Financial Knowledge Q&A', lambda x: 'Calling Financial RAG']
            ic.case['Employee Information Query', lambda x: 'Beijing']

        ic.start()
        assert ic('What is the weather today') == '38.5°C'
        assert ic('Who are you') == 'permission denied'
        assert ic('What is the difference between stocks and funds') == 'Calling Financial RAG'
        assert ic('Check the work location of Macro in the Technology Department') == 'Beijing'
