import lazyllm
from lazyllm.tools.rag.transform import (
    SentenceSplitter, CharacterSplitter, RecursiveSplitter,
    MarkdownSplitter, CodeSplitter
)
from lazyllm.tools.rag.transform.markdown import _MdSplit
from lazyllm.tools.rag.transform.base import _TextSplitterBase, _Split, _TokenTextSplitter
from lazyllm.tools.rag.doc_node import DocNode
import pytest
from unittest.mock import MagicMock
from lazyllm.tools.rag.document import Document
from lazyllm.tools.rag.retriever import Retriever

@pytest.fixture
def doc_node():
    node = MagicMock(spec=DocNode)
    node.get_text.return_value = '这是一个测试文本，用于验证分割器的逻辑是否正确。请认真检查！'
    node.get_metadata_str.return_value = ''
    return node


class TestSentenceSplitter:
    def setup_method(self):
        '''Setup for tests: initialize the SentenceSplitter.'''
        self.splitter = SentenceSplitter(chunk_size=30, chunk_overlap=10)

    def test_forward(self):
        text = ''' Before college the two main things I worked on, outside of school, were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep.'''  # noqa: E501
        docs = [DocNode(text=text)]

        result = self.splitter.batch_forward(docs, node_group='default')
        result_texts = [n.get_text() for n in result]
        expected_texts = [
            "Before college the two main things I worked on, outside of school, were writing and programming.I didn't write essays.",  # noqa: E501
            "I didn't write essays.I wrote what beginning writers were supposed to write then, and probably still are: short stories.My stories were awful.",  # noqa: E501
            'My stories were awful.They had hardly any plot, just characters with strong feelings, which I imagined made them deep.',  # noqa: E501
        ]
        assert result_texts == expected_texts

        trans = lazyllm.pipeline(lambda x: x, self.splitter)
        assert [n.get_text() for n in trans(docs[0])] == expected_texts

    def test_split(self):
        splitter = SentenceSplitter(chunk_size=10, chunk_overlap=0)
        text = 'This is a test sentence. It needs to be split into multiple chunks.'
        splits = splitter._split(text, chunk_size=10)
        assert len(splits) == 2
        assert splits[0].text == 'This is a test sentence.'
        assert splits[1].text == 'It needs to be split into multiple chunks.'

    def test_merge(self):
        splitter = SentenceSplitter(chunk_size=15, chunk_overlap=10)
        text = 'This is a test sentence. It needs to be split into multiple chunks.'
        splits = splitter._split(text, chunk_size=15)
        chunks = splitter._merge(splits, chunk_size=15)
        assert chunks == ['This is a test sentence. It needs to be split into multiple chunks.']

    def test_split_text(self):
        splitter = SentenceSplitter(chunk_size=10, chunk_overlap=0)
        text = 'This is a test sentence. It needs to be split into multiple chunks.'
        splits = splitter.split_text(text, metadata_size=0)
        assert splits == ['This is a test sentence.', 'It needs to be split into multiple chunks.']


class TestCharacterSplitter:
    def setup_method(self):
        '''Setup for tests: initialize the CharacterSplitter.'''
        self.splitter = CharacterSplitter(chunk_size=30, overlap=10, separator=',', keep_separator=True)

    def test_forward(self):
        text = ''' Before college the two main things I worked on, outside of school, were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep.'''  # noqa: E501
        docs = [DocNode(text=text)]
        result = self.splitter.batch_forward(docs, node_group='default')
        result_texts = [n.get_text() for n in result]
        expected_texts = [
            ' Before college the two main things I worked on, outside of school,',
            " outside of school, were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then,",  # noqa: E501
            ' wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot,',  # noqa: E501
            ' stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep.'  # noqa: E501
        ]
        assert result_texts == expected_texts

        trans = lazyllm.pipeline(lambda x: x, self.splitter)
        assert [n.get_text() for n in trans(docs[0])] == expected_texts

    def test_get_separator_pattern(self):
        splitter = CharacterSplitter(separator=' ')
        sep_pattern = splitter._get_separator_pattern(' ')
        assert sep_pattern == r'(?: )'
        splitter = CharacterSplitter(separator=' ', keep_separator=True)
        sep_pattern = splitter._get_separator_pattern(' ')
        assert sep_pattern == r'( )'

    def test_default_split(self):
        splitter = CharacterSplitter(separator=' ')
        text = 'Hello, world! This is a test.'
        splits = splitter._default_split(splitter._get_separator_pattern(' '), text)
        assert splits == ['Hello,', 'world!', 'This', 'is', 'a', 'test.']

        splitter = CharacterSplitter(separator=',', keep_separator=True)
        text = 'Hello, world! This is a test.'
        splits = splitter._default_split(splitter._get_separator_pattern(','), text)
        assert splits == ['Hello,', ' world! This is a test.']

        splitter = CharacterSplitter(separator=',', keep_separator=False)
        text = 'Hello, world! This is a test.'
        splits = splitter._default_split(splitter._get_separator_pattern(','), text)
        assert splits == ['Hello', ' world! This is a test.']

    def test_split_text(self):
        splitter = CharacterSplitter(separator=',', chunk_size=7, overlap=0)
        text = 'Hello, world! This is a test.'
        splits = splitter.split_text(text, metadata_size=0)
        assert splits == ['Hello', ' world! This is a test.']

    def test_split_text_with_split_fn(self):
        splitter = CharacterSplitter(separator=',', chunk_size=7, overlap=0)
        splitter.set_split_fns([lambda t: t.split(',')])
        text = 'Hello, world! This is a test.'
        splits = splitter.split_text(text, metadata_size=0)
        assert splits == ['Hello', ' world! This is a test.']
        splitter.add_split_fn(lambda t: t.split(' '), index=0)
        splits = splitter.split_text(text, metadata_size=0)
        assert splits == ['Hello,', 'world!', 'This', 'is', 'a', 'test.']
        splitter.clear_split_fns()
        splits = splitter.split_text(text, metadata_size=0)
        assert splits == ['Hello', ' world! This is a test.']

        def custom_split_with_separator(text, separator):
            chunks = text.split(separator)
            return [chunk.strip() for chunk in chunks if chunk.strip()]

        splitter = CharacterSplitter(separator=',', chunk_size=4, overlap=0)
        splitter.set_split_fns(custom_split_with_separator)
        text = 'apple, banana, cherry, date, happy, sad'
        splits = splitter.split_text(text, metadata_size=0)
        assert splits == ['apple', 'banana', 'cherry', 'date', 'happy', 'sad']

    def test_split_text_with_weak_split_fn(self):
        splitter = CharacterSplitter(separator=',', chunk_size=4, overlap=0)
        splitter.set_split_fns([lambda t: t.split('!')])
        text = 'Hello, world! This is a test.'
        splits = splitter.split_text(text, metadata_size=0)
        print(splits)
        assert splits == ['Hello, world', ' This is a test', '.']


class TestRecursiveSplitter:
    def setup_method(self):
        '''Setup for tests: initialize the RecursiveSplitter.'''
        self.splitter = RecursiveSplitter(chunk_size=30, overlap=10, separators=[',', '.', ' ', ''])

    def test_forward(self):
        text = ''' Before college the two main things I worked on, outside of school, were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep.'''  # noqa: E501
        docs = [DocNode(text=text)]
        result = self.splitter.batch_forward(docs, node_group='default')
        result_texts = [n.get_text() for n in result]
        expected_texts = [
            ' Before college the two main things I worked on outside of school',
            " outside of school were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then",  # noqa: E501
            ' I wrote what beginning writers were supposed to write then and probably still are: short stories. My stories were awful. They had hardly any plot',  # noqa: E501
            ' My stories were awful. They had hardly any plot just characters with strong feelings which I imagined made them deep.'  # noqa: E501
        ]

        assert result_texts == expected_texts

        trans = lazyllm.pipeline(lambda x: x, self.splitter)
        assert [n.get_text() for n in trans(docs[0])] == expected_texts

    def test_split_text(self):
        splitter = RecursiveSplitter(separators=['\n\n', '\n', '!', ' '], chunk_size=5, overlap=0)
        text = 'Hello\n\nworld! This\nis a test.'
        splits = splitter.split_text(text, metadata_size=0)
        assert splits == ['Hello', 'world! This', 'is a test.']

    def test_split_text_with_character_split_fn(self):
        splitter = RecursiveSplitter(separators=['\n\n', '\n', '!', ' '], chunk_size=9, overlap=0)
        splitter.set_split_fns([lambda t: t.split('\n\n')])
        text = 'Hello\n\nworld! This\nis a test.'
        splits = splitter.split_text(text, metadata_size=0)
        assert splits == ['Hello', 'world! This\nis a test.']
        splitter.add_split_fn(lambda t: t.split('\n'))
        splits = splitter.split_text(text, metadata_size=0)
        assert splits == ['Hello', 'world! This\nis a test.']
        splitter.clear_split_fns()
        splits = splitter.split_text(text, metadata_size=0)
        assert splits == ['Hello', 'world! This\nis a test.']


class TestMarkdownSplitter:
    def setup_method(self):
        '''Setup for tests: initialize the MarkdownSplitter.'''
        self.splitter = MarkdownSplitter(chunk_size=30, overlap=10)

    def test_get_heading_level(self):
        markdown = MarkdownSplitter()
        lines = [
            '# 标题1\n内容A',
            '## 标题2\n内容B',
            '### ###',
            '普通文本',
            '####### 非法标题',
            '## ### 这其实是标题文字'
        ]
        assert markdown._get_heading_level(lines[0]) == 1
        assert markdown._get_heading_level(lines[1]) == 2
        assert markdown._get_heading_level(lines[2]) == 3
        assert markdown._get_heading_level(lines[3]) == 0
        assert markdown._get_heading_level(lines[4]) == 0
        assert markdown._get_heading_level(lines[5]) == 2

    def test_split_markdown_by_semantics(self):
        text = '\n\n# 标题1\n内容A\n\n## 标题2\n内容B\n\n### 标题3\n内容C\n\n# 标题4\n内容D\n内容E'
        splits = self.splitter.split_markdown_by_semantics(text)
        assert splits == [
            _MdSplit(path=['标题1'], level=1, header='标题1', content='内容A', token_size=5, type='content'),
            _MdSplit(path=['标题1', '标题2'], level=2, header='标题2', content='内容B', token_size=5, type='content'),
            _MdSplit(path=['标题1', '标题2', '标题3'], level=3, header='标题3', content='内容C', token_size=5, type='content'),
            _MdSplit(path=['标题4'], level=1, header='标题4', content='内容D\n内容E', token_size=11, type='content')
        ]

    def test_split(self):
        text = '\n\n# 标题1\n内容A\n\n## 标题2\n内容B\n\n### 标题3\n内容C\n\n# 标题4\n内容D\n内容E'
        markdown = MarkdownSplitter(keep_headers=True, keep_trace=True)
        splits = markdown._split(text, 1024)
        assert splits == [
            _MdSplit(path=['标题1'], level=1, header='标题1', content='内容A', token_size=5, type='content'),
            _MdSplit(path=['标题1', '标题2'], level=2, header='标题2', content='内容B', token_size=5, type='content'),
            _MdSplit(path=['标题1', '标题2', '标题3'], level=3, header='标题3', content='内容C', token_size=5, type='content'),
            _MdSplit(path=['标题4'], level=1, header='标题4', content='内容D\n内容E', token_size=11, type='content')
        ]

        makrkdown = MarkdownSplitter(keep_headers=True, keep_trace=False)
        splits = makrkdown._split(text, 1024)
        assert splits == [
            _MdSplit(path=['标题1'], level=1, header='标题1', content='内容A', token_size=5, type='content'),
            _MdSplit(path=['标题1', '标题2'], level=2, header='标题2', content='内容B', token_size=5, type='content'),
            _MdSplit(path=['标题1', '标题2', '标题3'], level=3, header='标题3', content='内容C', token_size=5, type='content'),
            _MdSplit(path=['标题4'], level=1, header='标题4', content='内容D\n内容E', token_size=11, type='content')
        ]

        markdown = MarkdownSplitter(keep_headers=False, keep_trace=False)
        splits = markdown._split(text, 1024)
        assert splits == [
            _MdSplit(path=['标题1'], level=1, header='标题1', content='内容A', token_size=5, type='content'),
            _MdSplit(path=['标题1', '标题2'], level=2, header='标题2', content='内容B', token_size=5, type='content'),
            _MdSplit(path=['标题1', '标题2', '标题3'], level=3, header='标题3', content='内容C', token_size=5, type='content'),
            _MdSplit(path=['标题4'], level=1, header='标题4', content='内容D\n内容E', token_size=11, type='content')
        ]

    def test_merge(self):
        md_text = '\n\n# LinuxBoot on Ampere Mt. Jade Platform' \
                  '\nThe Ampere Altra Family processor based Mt. Jade platform is a high-performance ARM server platform, offering up to 256 processor cores in a ' \
                  'dual socket configuration. The Tianocore EDK2 firmware for the Mt. Jade platform has been fully upstreamed to the tianocore/edk2-platforms repository, '\
                  'enabling the community to build and experiment with the platform\'s firmware using entirely open-source code. It also supports LinuxBoot, an open-source ' \
                  'firmware framework that reduces boot time, enhances security, and increases flexibility compared to standard UEFI firmware.'\
                  '\n\nMt. Jade has also achieved a significant milestone by becoming [the first server certified under the Arm SystemReady LS certification program](https://community.arm.com/arm-community-blogs' \
                  '/b/architectures-and-processors-blog/posts/arm-systemready-ls). SystemReady LS ensures compliance with standardized boot and runtime environments for Linux-based ' \
                  'systems, enabling seamless deployment across diverse hardware. This certification further emphasizes Mt. Jade\'s readiness for enterprise and cloud-scale adoption '\
                  'by providing assurance of compatibility, performance, and reliability.' \
                  '\n\nThis case study explores the LinuxBoot implementation on the Ampere Mt. Jade platform, inspired by the approach used in [Google\'s LinuxBoot deployment](Google_study.md).' \
                  '\n\n## Ampere EDK2-LinuxBoot Components' \
                  '\nThe Mt. Jade platform embraces a hybrid firmware architecture, combining UEFI/EDK2 for hardware initialization and LinuxBoot for advanced boot functionalities. The platform aligns closely with step 6 in the LinuxBoot adoption model.' \
                  '\n\n<img src=\"../images/Case-study-Ampere.svg\">' \
                  '\n\nThe entire boot firmware stack for the Mt. Jade is open source and available in the Github.' \
                  '\n\n* **EDK2**: The PEI and minimal (stripped-down) DXE drivers, including both common and platform code, are fully open source and resides in Tianocore edk2-platforms and edk2 repositories.'\
                  '\n* **LinuxBoot**: The LinuxBoot binary ([flashkernel](../glossary.md)) for Mt. Jade is supported in the [linuxboot/linuxboot](https://github.com/linuxboot/linuxboot/tree/main/mainboards/ampere/jade) repository.' \
                  '\n\n## Ampere Solution for LinuxBoot as a Boot Device Selection'\
                  '\nAmpere has implemented and successfully upstreamed a solution for integrating LinuxBoot as a Boot Device Selection (BDS) option into the TianoCore EDK2 framework, as seen in commit [ArmPkg: Implement PlatformBootManagerLib for LinuxBoot](https://github.com/tianocore/edk2/commit/62540372230ecb5318a9c8a40580a14beeb9ded0). This innovation simplifies the boot process for the Mt. Jade platform and aligns with LinuxBoot\'s goals of efficiency and flexibility.'\
                  '\n\nUnlike the earlier practice that replaced the UEFI Shell with a LinuxBoot flashkernel, Ampere\'s solution introduces a custom BDS implementation that directly boots into the LinuxBoot environment as the active boot option. This approach bypasses the need to load the UEFI Shell or UiApp (UEFI Setup Menu), which depend on numerous unnecessary DXE drivers.'\
                  '\n\nTo further enhance flexibility, Ampere introduced a new GUID specifically for the LinuxBoot binary, ensuring clear separation from the UEFI Shell GUID. This distinction allows precise identification of LinuxBoot components in the firmware.'\
                  '\n\n## Build Process'\
                  '\nBuilding a flashable EDK2 firmware image with an integrated LinuxBoot flashkernel for the Ampere Mt. Jade platform involves two main steps: building the LinuxBoot flashkernel and integrating it into the EDK2 firmware build.'\
                  '\n\n### Step 1: Build the LinuxBoot Flashkernel'\
                  '\nThe LinuxBoot flash kernel is built as follows:'\
                  '\n\n```bash\ngit clone https://github.com/linuxboot/linuxboot.git'\
                  '\ncd linuxboot/mainboards/ampere/jade && make fetch flashkernel'\
                  '\n```' \
                  '\n\nAfter the build process completes, the flash kernel will be located at: linuxboot/mainboards/ampere/jade/flashkernel'\
                  '\n\n### Step 2: Build the EDK2 Firmware Image with the Flash Kernel'\
                  '\nThe EDK2 firmware image is built with the LinuxBoot flashkernel integrated into the flash image using the following steps:'\
                  '\n\n```bash'\
                  '\ngit clone https://github.com/tianocore/edk2-platforms.git'\
                  '\ngit clone https://github.com/tianocore/edk2.git'\
                  '\ngit clone https://github.com/tianocore/edk2-non-osi.git'\
                  '\n./edk2-platforms/Platform/Ampere/buildfw.sh -b RELEASE -t GCC -p Jade -l linuxboot/mainboards/ampere/jade/flashkernel'\
                  '\n```'\
                  '\n\nThe `buildfw.sh` script automatically integrates the LinuxBoot flash kernel (provided via the -l option) as part of the final EDK2 firmware image.'\
                  '\n\nThis process generates a flashable EDK2 firmware image with embedded LinuxBoot, ready for deployment on the Ampere Mt. Jade platform.'\
                  '\n\n## Booting with LinuxBoot\nWhen powered on, the system will boot into the u-root and automatically kexec to the target OS.'\
                  '\n\n```text'\
                  '\nRun /init as init process'\
                  '\n1970/01/01 00:00:10 Welcome to u-root!'\
                  '\n...'\
                  '\n```'\
                  '\n\n## Future Work'\
                  '\nWhile the LinuxBoot implementation on the Ampere Mt. Jade platform represents a significant milestone, several advanced features and improvements remain to be explored. These enhancements would extend the platform\'s capabilities, improve its usability, and reinforce its position as a leading open source firmware solution. Key areas for future development include:'\
                  '\n\n### Secure Boot with LinuxBoot'\
                  '\nOne of the critical areas for future development is enabling secure boot verification for the target operating system. In the LinuxBoot environment, the target OS is typically booted using kexec. However, it is unclear how Secure Boot operates in this context, as kexec bypasses traditional firmware-controlled secure boot mechanisms. Future work should investigate how to extend Secure Boot principles to kexec, ensuring that the OS kernel and its components are verified and authenticated before execution. This may involve implementing signature checks and utilizing trusted certificate chains directly within the LinuxBoot environment to mimic the functionality of UEFI Secure Boot during the kexec process.'\
                  '\n\n### TPM Support'\
                  '\nThe platform supports TPM, but its integration with LinuxBoot is yet to be defined. Future work could explore utilizing the TPM for secure boot measurements, and system integrity attestation.'\
                  '\n\n### Expanding Support for Additional Ampere Platforms'\
                  '\nBuilding on the success of LinuxBoot on Mt. Jade, future efforts should expand support to other Ampere platforms. This would ensure broader adoption and usability across different hardware configurations.'\
                  '\n\n### Optimizing the Transition Between UEFI and LinuxBoot'\
                  '\nImproving the efficiency of the handoff between UEFI and LinuxBoot could further reduce boot times. This optimization would involve refining the initialization process and minimizing redundant operations during the handoff.'\
                  '\n\n### Advanced Diagnostics and Monitoring Tools'\
                  '\nAdding more diagnostic and monitoring tools to the LinuxBoot u-root environment would enhance debugging and system management. These tools could provide deeper insights into system performance and potential issues, improving reliability and maintainability.'\
                  '\n\n## See Also'\
                  '\n* [LinuxBoot on Ampere Platforms: A new (old) approach to firmware](https://amperecomputing.com/blogs/linuxboot-on-ampere-platforms--a-new-old-approach-to-firmware)'  # noqa: E501
        markdown = MarkdownSplitter(keep_headers=True, keep_trace=True, overlap=30)
        splits = markdown._split(md_text, 300)
        merged = markdown._merge(splits, 300)
        expected_merged = [
            "The Ampere Altra Family processor based Mt. Jade platform is a high-performance ARM server platform, offering up to 256 processor cores in a dual socket configuration. The Tianocore EDK2 firmware for the Mt. Jade platform has been fully upstreamed to the tianocore/edk2-platforms repository, enabling the community to build and experiment with the platform's firmware using entirely open-source code. It also supports LinuxBoot, an open-source firmware framework that reduces boot time, enhances security, and increases flexibility compared to standard UEFI firmware.\n\nMt. Jade has also achieved a significant milestone by becoming [the first server certified under the Arm SystemReady LS certification program](https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/arm-systemready-ls). SystemReady LS ensures compliance with standardized boot and runtime environments for Linux-based systems, enabling seamless deployment across diverse hardware. This certification further emphasizes Mt. Jade's readiness for enterprise and cloud-scale adoption by providing assurance of compatibility, performance, and reliability.\n\nThis case study explores the LinuxBoot implementation on the Ampere Mt. Jade platform, inspired by the approach used in [Google's LinuxBoot deployment](Google_study.md).",  # noqa: E501

            'The Mt. Jade platform embraces a hybrid firmware architecture, combining UEFI/EDK2 for hardware initialization and LinuxBoot for advanced boot functionalities. The platform aligns '  # noqa: E501
            'closely with step 6 in the LinuxBoot adoption model.\n\n<img src="../images/Case-study-Ampere.svg">\n\nThe entire boot firmware stack for the Mt. '  # noqa: E501
            'Jade is open source and available in the Github.\n\n* **EDK2**: The PEI and minimal (stripped-down) DXE drivers, including both common and platform code, are fully open source and resides in Tianocore edk2-platforms and edk2 repositories.\n* **LinuxBoot**: The LinuxBoot binary ([flashkernel](../glossary.md)) for Mt. Jade is supported in the [linuxboot/linuxboot](https://github.com/linuxboot/linuxboot/tree/main/mainboards/ampere/jade) repository.',  # noqa: E501

            "Ampere has implemented and successfully upstreamed a solution for integrating LinuxBoot as a Boot Device Selection (BDS) option into the TianoCore EDK2 framework, as seen in commit [ArmPkg: Implement PlatformBootManagerLib for LinuxBoot](https://github.com/tianocore/edk2/commit/62540372230ecb5318a9c8a40580a14beeb9ded0). This innovation simplifies the boot process for the Mt. Jade platform and aligns with LinuxBoot's goals of efficiency and flexibility.\n\nUnlike the earlier practice that replaced the UEFI Shell with a LinuxBoot flashkernel, Ampere's solution introduces a custom BDS implementation that directly boots into the LinuxBoot environment as the active boot option. This approach bypasses the need to load the UEFI Shell or UiApp (UEFI Setup Menu), which depend on numerous unnecessary DXE drivers.\n\nTo further enhance flexibility, Ampere introduced a new GUID specifically for the LinuxBoot binary, ensuring clear separation from the UEFI Shell GUID. This distinction allows precise identification of LinuxBoot components in the firmware.",  # noqa: E501

            'Building a flashable EDK2 firmware image with an integrated LinuxBoot flashkernel for the Ampere Mt. Jade platform involves two main steps: building the LinuxBoot flashkernel and integrating it into the EDK2 firmware build.',  # noqa: E501

            'The LinuxBoot flash kernel is built as follows:\n\n```bash\ngit clone https://github.com/linuxboot/linuxboot.git\ncd linuxboot/mainboards/ampere/jade && make fetch flashkernel\n```\n\nAfter the build process completes, the flash kernel will be located at: linuxboot/mainboards/ampere/jade/flashkernel',  # noqa: E501

            'The EDK2 firmware image is built with the LinuxBoot flashkernel integrated into the flash image using the following steps:\n\n```bash\ngit clone https://github.com/tianocore/edk2-platforms.git\ngit clone https://github.com/tianocore/edk2.git\ngit clone https://github.com/tianocore/edk2-non-osi.git\n./edk2-platforms/Platform/Ampere/buildfw.sh -b RELEASE -t GCC -p Jade -l linuxboot/mainboards/ampere/jade/flashkernel\n```\n\nThe `buildfw.sh` script automatically integrates the LinuxBoot flash kernel (provided via the -l option) as part of the final EDK2 firmware image.\n\nThis process generates a flashable EDK2 firmware image with embedded LinuxBoot, ready for deployment on the Ampere Mt. Jade platform.',  # noqa: E501

            'When powered on, the system will boot into the u-root and automatically kexec to the target OS.\n\n```text\nRun /init as init process\n1970/01/01 00:00:10 Welcome to u-root!\n...\n```',  # noqa: E501

            "While the LinuxBoot implementation on the Ampere Mt. Jade platform represents a significant milestone, several advanced features and improvements remain to be explored. These enhancements would extend the platform's capabilities, improve its usability, and reinforce its position as a leading open source firmware solution. Key areas for future development include:",  # noqa: E501

            'One of the critical areas for future development is enabling secure boot verification for the target operating system. In the LinuxBoot environment, the target OS is typically booted using kexec. However, it is unclear how Secure Boot operates in this context, as kexec bypasses traditional firmware-controlled secure boot mechanisms. Future work should investigate how to extend Secure Boot principles to kexec, ensuring that the OS kernel and its components are verified and authenticated before execution. This may involve implementing signature checks and utilizing trusted certificate chains directly within the LinuxBoot environment to mimic the functionality of UEFI Secure Boot during the kexec process.',  # noqa: E501

            'The platform supports TPM, but its integration with LinuxBoot is yet to be defined. Future work could explore utilizing the TPM for secure boot measurements, and system integrity attestation.',  # noqa: E501

            'Building on the success of LinuxBoot on Mt. Jade, future efforts should expand support to other Ampere platforms. This would ensure broader adoption and usability across different hardware configurations.',  # noqa: E501

            'Improving the efficiency of the handoff between UEFI and LinuxBoot could further reduce boot times. This optimization would involve refining the initialization process and minimizing redundant operations during the handoff.',  # noqa: E501

            'Adding more diagnostic and monitoring tools to the LinuxBoot u-root environment would enhance debugging and system management. These tools could provide deeper insights into system performance and potential issues, improving reliability and maintainability.',  # noqa: E501

            '* [LinuxBoot on Ampere Platforms: A new (old) approach to firmware](https://amperecomputing.com/blogs/linuxboot-on-ampere-platforms--a-new-old-approach-to-firmware)']  # noqa: E501
        for i in range(len(merged)):
            assert merged[i].content == expected_merged[i]
            assert merged[i].metadata.get('path') is not None
            assert merged[i].metadata.get('header') is not None

        markdown = MarkdownSplitter(keep_headers=True, keep_trace=False, overlap=30)
        splits = markdown._split(md_text, 300)
        merged = markdown._merge(splits, 300)
        for i in range(len(merged)):
            assert merged[i].content == expected_merged[i]
            assert merged[i].metadata.get('path') is None
            assert merged[i].metadata.get('header') is not None

        markdown = MarkdownSplitter(keep_headers=False, keep_trace=False, overlap=30)
        splits = markdown._split(md_text, 300)
        merged = markdown._merge(splits, 300)
        for i in range(len(merged)):
            assert merged[i].content == expected_merged[i]
            assert merged[i].metadata.get('path') is None
            assert merged[i].metadata.get('header') is None

    def test_keep_code_blocks(self):
        md_text = '\n\n# LinuxBoot on Ampere Mt. Jade Platform\nThe Ampere Altra Family processor based Mt. Jade platform is a high-performance ARM server platform, offering up to 256 processor cores in a dual socket configuration. The Tianocore EDK2 firmware for the Mt. Jade platform has been fully upstreamed to the tianocore/edk2-platforms repository, enabling the community to build and experiment with the platform\'s firmware using entirely open-source code. It also supports LinuxBoot, an open-source firmware framework that reduces boot time, enhances security, and increases flexibility compared to standard UEFI firmware.\n\nMt. Jade has also achieved a significant milestone by becoming [the first server certified under the Arm SystemReady LS certification program](https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/arm-systemready-ls). SystemReady LS ensures compliance with standardized boot and runtime environments for Linux-based systems, enabling seamless deployment across diverse hardware. This certification further emphasizes Mt. Jade\'s readiness for enterprise and cloud-scale adoption by providing assurance of compatibility, performance, and reliability.\n\nThis case study explores the LinuxBoot implementation on the Ampere Mt. Jade platform, inspired by the approach used in [Google\'s LinuxBoot deployment](Google_study.md).\n\n## Ampere EDK2-LinuxBoot Components\nThe Mt. Jade platform embraces a hybrid firmware architecture, combining UEFI/EDK2 for hardware initialization and LinuxBoot for advanced boot functionalities. The platform aligns closely with step 6 in the LinuxBoot adoption model.\n\n<img src=\"../images/Case-study-Ampere.svg\">\n\nThe entire boot firmware stack for the Mt. Jade is open source and available in the Github.\n\n* **EDK2**: The PEI and minimal (stripped-down) DXE drivers, including both common and platform code, are fully open source and resides in Tianocore edk2-platforms and edk2 repositories.\n* **LinuxBoot**: The LinuxBoot binary ([flashkernel](../glossary.md)) for Mt. Jade is supported in the [linuxboot/linuxboot](https://github.com/linuxboot/linuxboot/tree/main/mainboards/ampere/jade) repository.\n\n## Ampere Solution for LinuxBoot as a Boot Device Selection\nAmpere has implemented and successfully upstreamed a solution for integrating LinuxBoot as a Boot Device Selection (BDS) option into the TianoCore EDK2 framework, as seen in commit [ArmPkg: Implement PlatformBootManagerLib for LinuxBoot](https://github.com/tianocore/edk2/commit/62540372230ecb5318a9c8a40580a14beeb9ded0). This innovation simplifies the boot process for the Mt. Jade platform and aligns with LinuxBoot\'s goals of efficiency and flexibility.\n\nUnlike the earlier practice that replaced the UEFI Shell with a LinuxBoot flashkernel, Ampere\'s solution introduces a custom BDS implementation that directly boots into the LinuxBoot environment as the active boot option. This approach bypasses the need to load the UEFI Shell or UiApp (UEFI Setup Menu), which depend on numerous unnecessary DXE drivers.\n\nTo further enhance flexibility, Ampere introduced a new GUID specifically for the LinuxBoot binary, ensuring clear separation from the UEFI Shell GUID. This distinction allows precise identification of LinuxBoot components in the firmware.\n\n## Build Process\nBuilding a flashable EDK2 firmware image with an integrated LinuxBoot flashkernel for the Ampere Mt. Jade platform involves two main steps: building the LinuxBoot flashkernel and integrating it into the EDK2 firmware build.\n\n### Step 1: Build the LinuxBoot Flashkernel\nThe LinuxBoot flash kernel is built as follows:\n\n```bash\ngit clone https://github.com/linuxboot/linuxboot.git\ncd linuxboot/mainboards/ampere/jade && make fetch flashkernel\n```\n\nAfter the build process completes, the flash kernel will be located at: linuxboot/mainboards/ampere/jade/flashkernel\n\n### Step 2: Build the EDK2 Firmware Image with the Flash Kernel\nThe EDK2 firmware image is built with the LinuxBoot flashkernel integrated into the flash image using the following steps:\n\n```bash\ngit clone https://github.com/tianocore/edk2-platforms.git\ngit clone https://github.com/tianocore/edk2.git\ngit clone https://github.com/tianocore/edk2-non-osi.git\n./edk2-platforms/Platform/Ampere/buildfw.sh -b RELEASE -t GCC -p Jade -l linuxboot/mainboards/ampere/jade/flashkernel\n```\n\nThe `buildfw.sh` script automatically integrates the LinuxBoot flash kernel (provided via the -l option) as part of the final EDK2 firmware image.\n\nThis process generates a flashable EDK2 firmware image with embedded LinuxBoot, ready for deployment on the Ampere Mt. Jade platform.\n\n## Booting with LinuxBoot\nWhen powered on, the system will boot into the u-root and automatically kexec to the target OS.\n\n```text\nRun /init as init process\n1970/01/01 00:00:10 Welcome to u-root!\n...\n```\n\n## Future Work\nWhile the LinuxBoot implementation on the Ampere Mt. Jade platform represents a significant milestone, several advanced features and improvements remain to be explored. These enhancements would extend the platform\'s capabilities, improve its usability, and reinforce its position as a leading open source firmware solution. Key areas for future development include:\n\n### Secure Boot with LinuxBoot\nOne of the critical areas for future development is enabling secure boot verification for the target operating system. In the LinuxBoot environment, the target OS is typically booted using kexec. However, it is unclear how Secure Boot operates in this context, as kexec bypasses traditional firmware-controlled secure boot mechanisms. Future work should investigate how to extend Secure Boot principles to kexec, ensuring that the OS kernel and its components are verified and authenticated before execution. This may involve implementing signature checks and utilizing trusted certificate chains directly within the LinuxBoot environment to mimic the functionality of UEFI Secure Boot during the kexec process.\n\n### TPM Support\nThe platform supports TPM, but its integration with LinuxBoot is yet to be defined. Future work could explore utilizing the TPM for secure boot measurements, and system integrity attestation.\n\n### Expanding Support for Additional Ampere Platforms\nBuilding on the success of LinuxBoot on Mt. Jade, future efforts should expand support to other Ampere platforms. This would ensure broader adoption and usability across different hardware configurations.\n\n### Optimizing the Transition Between UEFI and LinuxBoot\nImproving the efficiency of the handoff between UEFI and LinuxBoot could further reduce boot times. This optimization would involve refining the initialization process and minimizing redundant operations during the handoff.\n\n### Advanced Diagnostics and Monitoring Tools\nAdding more diagnostic and monitoring tools to the LinuxBoot u-root environment would enhance debugging and system management. These tools could provide deeper insights into system performance and potential issues, improving reliability and maintainability.\n\n## See Also\n* [LinuxBoot on Ampere Platforms: A new (old) approach to firmware](https://amperecomputing.com/blogs/linuxboot-on-ampere-platforms--a-new-old-approach-to-firmware)'  # noqa: E501
        markdown = MarkdownSplitter(keep_headers=False, keep_trace=False, overlap=30, keep_code_blocks=True)
        splits = markdown._split(md_text, 300)
        expected_splits = [
            _MdSplit(path=['LinuxBoot on Ampere Mt. Jade Platform'], level=1, header='LinuxBoot on Ampere Mt. Jade Platform', content="The Ampere Altra Family processor based Mt. Jade platform is a high-performance ARM server platform, offering up to 256 processor cores in a dual socket configuration. The Tianocore EDK2 firmware for the Mt. Jade platform has been fully upstreamed to the tianocore/edk2-platforms repository, enabling the community to build and experiment with the platform's firmware using entirely open-source code. It also supports LinuxBoot, an open-source firmware framework that reduces boot time, enhances security, and increases flexibility compared to standard UEFI firmware.\n\nMt. Jade has also achieved a significant milestone by becoming [the first server certified under the Arm SystemReady LS certification program](https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/arm-systemready-ls). SystemReady LS ensures compliance with standardized boot and runtime environments for Linux-based systems, enabling seamless deployment across diverse hardware. This certification further emphasizes Mt. Jade's readiness for enterprise and cloud-scale adoption by providing assurance of compatibility, performance, and reliability.\n\nThis case study explores the LinuxBoot implementation on the Ampere Mt. Jade platform, inspired by the approach used in [Google's LinuxBoot deployment](Google_study.md).", token_size=271, type='content'),  # noqa: E501

            _MdSplit(path=['LinuxBoot on Ampere Mt. Jade Platform', 'Ampere EDK2-LinuxBoot Components'], level=2, header='Ampere EDK2-LinuxBoot Components', content='The Mt. Jade platform embraces a hybrid firmware architecture, combining UEFI/EDK2 for hardware initialization and LinuxBoot for advanced boot functionalities. The platform aligns closely with step 6 in the LinuxBoot adoption model.\n\n<img src="../images/Case-study-Ampere.svg">\n\nThe entire boot firmware stack for the Mt. Jade is open source and available in the Github.\n\n* **EDK2**: The PEI and minimal (stripped-down) DXE drivers, including both common and platform code, are fully open source and resides in Tianocore edk2-platforms and edk2 repositories.\n* **LinuxBoot**: The LinuxBoot binary ([flashkernel](../glossary.md)) for Mt. Jade is supported in the [linuxboot/linuxboot](https://github.com/linuxboot/linuxboot/tree/main/mainboards/ampere/jade) repository.', token_size=204, type='content'),  # noqa: E501

            _MdSplit(path=['LinuxBoot on Ampere Mt. Jade Platform', 'Ampere Solution for LinuxBoot as a Boot Device Selection'], level=2, header='Ampere Solution for LinuxBoot as a Boot Device Selection', content="Ampere has implemented and successfully upstreamed a solution for integrating LinuxBoot as a Boot Device Selection (BDS) option into the TianoCore EDK2 framework, as seen in commit [ArmPkg: Implement PlatformBootManagerLib for LinuxBoot](https://github.com/tianocore/edk2/commit/62540372230ecb5318a9c8a40580a14beeb9ded0). This innovation simplifies the boot process for the Mt. Jade platform and aligns with LinuxBoot's goals of efficiency and flexibility.\n\nUnlike the earlier practice that replaced the UEFI Shell with a LinuxBoot flashkernel, Ampere's solution introduces a custom BDS implementation that directly boots into the LinuxBoot environment as the active boot option. This approach bypasses the need to load the UEFI Shell or UiApp (UEFI Setup Menu), which depend on numerous unnecessary DXE drivers.\n\nTo further enhance flexibility, Ampere introduced a new GUID specifically for the LinuxBoot binary, ensuring clear separation from the UEFI Shell GUID. This distinction allows precise identification of LinuxBoot components in the firmware.", token_size=240, type='content'),  # noqa: E501

            _MdSplit(path=['LinuxBoot on Ampere Mt. Jade Platform', 'Build Process'], level=2, header='Build Process', content='Building a flashable EDK2 firmware image with an integrated LinuxBoot flashkernel for the Ampere Mt. Jade platform involves two main steps: building the LinuxBoot flashkernel and integrating it into the EDK2 firmware build.', token_size=47, type='content'),  # noqa: E501

            _MdSplit(path=['LinuxBoot on Ampere Mt. Jade Platform', 'Build Process', 'Step 1: Build the LinuxBoot Flashkernel'], level=3, header='Step 1: Build the LinuxBoot Flashkernel', content='The LinuxBoot flash kernel is built as follows:', token_size=10, type='content'),  # noqa: E501

            _MdSplit(path=['LinuxBoot on Ampere Mt. Jade Platform', 'Build Process', 'Step 1: Build the LinuxBoot Flashkernel'], level=3, header='Step 1: Build the LinuxBoot Flashkernel', content='git clone https://github.com/linuxboot/linuxboot.git\ncd linuxboot/mainboards/ampere/jade && make fetch flashkernel', token_size=34, type='bash'),  # noqa: E501

            _MdSplit(path=['LinuxBoot on Ampere Mt. Jade Platform', 'Build Process', 'Step 1: Build the LinuxBoot Flashkernel'], level=3, header='Step 1: Build the LinuxBoot Flashkernel', content='After the build process completes, the flash kernel will be located at: linuxboot/mainboards/ampere/jade/flashkernel', token_size=29, type='content'),  # noqa: E501

            _MdSplit(path=['LinuxBoot on Ampere Mt. Jade Platform', 'Build Process', 'Step 2: Build the EDK2 Firmware Image with the Flash Kernel'], level=3, header='Step 2: Build the EDK2 Firmware Image with the Flash Kernel', content='The EDK2 firmware image is built with the LinuxBoot flashkernel integrated into the flash image using the following steps:', token_size=24, type='content'),  # noqa: E501

            _MdSplit(path=['LinuxBoot on Ampere Mt. Jade Platform', 'Build Process', 'Step 2: Build the EDK2 Firmware Image with the Flash Kernel'], level=3, header='Step 2: Build the EDK2 Firmware Image with the Flash Kernel', content='git clone https://github.com/tianocore/edk2-platforms.git\ngit clone https://github.com/tianocore/edk2.git\ngit clone https://github.com/tianocore/edk2-non-osi.git\n./edk2-platforms/Platform/Ampere/buildfw.sh -b RELEASE -t GCC -p Jade -l linuxboot/mainboards/ampere/jade/flashkernel', token_size=108, type='bash'),  # noqa: E501

            _MdSplit(path=['LinuxBoot on Ampere Mt. Jade Platform', 'Build Process', 'Step 2: Build the EDK2 Firmware Image with the Flash Kernel'], level=3, header='Step 2: Build the EDK2 Firmware Image with the Flash Kernel', content='The `buildfw.sh` script automatically integrates the LinuxBoot flash kernel (provided via the -l option) as part of the final EDK2 firmware image.\n\nThis process generates a flashable EDK2 firmware image with embedded LinuxBoot, ready for deployment on the Ampere Mt. Jade platform.', token_size=65, type='content'),  # noqa: E501

            _MdSplit(path=['LinuxBoot on Ampere Mt. Jade Platform', 'Booting with LinuxBoot'], level=2, header='Booting with LinuxBoot', content='When powered on, the system will boot into the u-root and automatically kexec to the target OS.', token_size=23, type='content'),  # noqa: E501

            _MdSplit(path=['LinuxBoot on Ampere Mt. Jade Platform', 'Booting with LinuxBoot'], level=2, header='Booting with LinuxBoot', content='Run /init as init process\n1970/01/01 00:00:10 Welcome to u-root!\n...', token_size=25, type='text'),  # noqa: E501

            _MdSplit(path=['LinuxBoot on Ampere Mt. Jade Platform', 'Future Work'], level=2, header='Future Work', content="While the LinuxBoot implementation on the Ampere Mt. Jade platform represents a significant milestone, several advanced features and improvements remain to be explored. These enhancements would extend the platform's capabilities, improve its usability, and reinforce its position as a leading open source firmware solution. Key areas for future development include:", token_size=61, type='content'),  # noqa: E501

            _MdSplit(path=['LinuxBoot on Ampere Mt. Jade Platform', 'Future Work', 'Secure Boot with LinuxBoot'], level=3, header='Secure Boot with LinuxBoot', content='One of the critical areas for future development is enabling secure boot verification for the target operating system. In the LinuxBoot environment, the target OS is typically booted using kexec. However, it is unclear how Secure Boot operates in this context, as kexec bypasses traditional firmware-controlled secure boot mechanisms. Future work should investigate how to extend Secure Boot principles to kexec, ensuring that the OS kernel and its components are verified and authenticated before execution. This may involve implementing signature checks and utilizing trusted certificate chains directly within the LinuxBoot environment to mimic the functionality of UEFI Secure Boot during the kexec process.', token_size=126, type='content'),  # noqa: E501

            _MdSplit(path=['LinuxBoot on Ampere Mt. Jade Platform', 'Future Work', 'TPM Support'], level=3, header='TPM Support', content='The platform supports TPM, but its integration with LinuxBoot is yet to be defined. Future work could explore utilizing the TPM for secure boot measurements, and system integrity attestation.', token_size=37, type='content'),  # noqa: E501

            _MdSplit(path=['LinuxBoot on Ampere Mt. Jade Platform', 'Future Work', 'Expanding Support for Additional Ampere Platforms'], level=3, header='Expanding Support for Additional Ampere Platforms', content='Building on the success of LinuxBoot on Mt. Jade, future efforts should expand support to other Ampere platforms. This would ensure broader adoption and usability across different hardware configurations.', token_size=36, type='content'),  # noqa: E501

            _MdSplit(path=['LinuxBoot on Ampere Mt. Jade Platform', 'Future Work', 'Optimizing the Transition Between UEFI and LinuxBoot'], level=3, header='Optimizing the Transition Between UEFI and LinuxBoot', content='Improving the efficiency of the handoff between UEFI and LinuxBoot could further reduce boot times. This optimization would involve refining the initialization process and minimizing redundant operations during the handoff.', token_size=37, type='content'),  # noqa: E501

            _MdSplit(path=['LinuxBoot on Ampere Mt. Jade Platform', 'Future Work', 'Advanced Diagnostics and Monitoring Tools'], level=3, header='Advanced Diagnostics and Monitoring Tools', content='Adding more diagnostic and monitoring tools to the LinuxBoot u-root environment would enhance debugging and system management. These tools could provide deeper insights into system performance and potential issues, improving reliability and maintainability.', token_size=40, type='content'),  # noqa: E501

            _MdSplit(path=['LinuxBoot on Ampere Mt. Jade Platform', 'See Also'], level=2, header='See Also', content='* [LinuxBoot on Ampere Platforms: A new (old) approach to firmware](https://amperecomputing.com/blogs/linuxboot-on-ampere-platforms--a-new-old-approach-to-firmware)', token_size=59, type='content')]  # noqa: E501
        assert splits == expected_splits
        merged = markdown._merge(splits, 300)
        expected_merged = [
            "The Ampere Altra Family processor based Mt. Jade platform is a high-performance ARM server platform, offering up to 256 processor cores in a dual socket configuration. The Tianocore EDK2 firmware for the Mt. Jade platform has been fully upstreamed to the tianocore/edk2-platforms repository, enabling the community to build and experiment with the platform's firmware using entirely open-source code. It also supports LinuxBoot, an open-source firmware framework that reduces boot time, enhances security, and increases flexibility compared to standard UEFI firmware.\n\nMt. Jade has also achieved a significant milestone by becoming [the first server certified under the Arm SystemReady LS certification program](https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/arm-systemready-ls). SystemReady LS ensures compliance with standardized boot and runtime environments for Linux-based systems, enabling seamless deployment across diverse hardware. This certification further emphasizes Mt. Jade's readiness for enterprise and cloud-scale adoption by providing assurance of compatibility, performance, and reliability.\n\nThis case study explores the LinuxBoot implementation on the Ampere Mt. Jade platform, inspired by the approach used in [Google's LinuxBoot deployment](Google_study.md).",  # noqa: E501

            'The Mt. Jade platform embraces a hybrid firmware architecture, combining UEFI/EDK2 for hardware initialization and LinuxBoot for advanced boot functionalities. The platform aligns closely with step 6 in the LinuxBoot adoption model.\n\n<img src="../images/Case-study-Ampere.svg">\n\nThe entire boot firmware stack for the Mt. Jade is open source and available in the Github.\n\n* **EDK2**: The PEI and minimal (stripped-down) DXE drivers, including both common and platform code, are fully open source and resides in Tianocore edk2-platforms and edk2 repositories.\n* **LinuxBoot**: The LinuxBoot binary ([flashkernel](../glossary.md)) for Mt. Jade is supported in the [linuxboot/linuxboot](https://github.com/linuxboot/linuxboot/tree/main/mainboards/ampere/jade) repository.',  # noqa: E501

            "Ampere has implemented and successfully upstreamed a solution for integrating LinuxBoot as a Boot Device Selection (BDS) option into the TianoCore EDK2 framework, as seen in commit [ArmPkg: Implement PlatformBootManagerLib for LinuxBoot](https://github.com/tianocore/edk2/commit/62540372230ecb5318a9c8a40580a14beeb9ded0). This innovation simplifies the boot process for the Mt. Jade platform and aligns with LinuxBoot's goals of efficiency and flexibility.\n\nUnlike the earlier practice that replaced the UEFI Shell with a LinuxBoot flashkernel, Ampere's solution introduces a custom BDS implementation that directly boots into the LinuxBoot environment as the active boot option. This approach bypasses the need to load the UEFI Shell or UiApp (UEFI Setup Menu), which depend on numerous unnecessary DXE drivers.\n\nTo further enhance flexibility, Ampere introduced a new GUID specifically for the LinuxBoot binary, ensuring clear separation from the UEFI Shell GUID. This distinction allows precise identification of LinuxBoot components in the firmware.", 'Building a flashable EDK2 firmware image with an integrated LinuxBoot flashkernel for the Ampere Mt. Jade platform involves two main steps: building the LinuxBoot flashkernel and integrating it into the EDK2 firmware build.', 'The LinuxBoot flash kernel is built as follows:',  # noqa: E501

            'git clone https://github.com/linuxboot/linuxboot.git\ncd linuxboot/mainboards/ampere/jade && make fetch flashkernel',  # noqa: E501

            'After the build process completes, the flash kernel will be located at: linuxboot/mainboards/ampere/jade/flashkernel',  # noqa: E501

            'The EDK2 firmware image is built with the LinuxBoot flashkernel integrated into the flash image using the following steps:',  # noqa: E501

            'git clone https://github.com/tianocore/edk2-platforms.git\ngit clone https://github.com/tianocore/edk2.git\ngit clone https://github.com/tianocore/edk2-non-osi.git\n./edk2-platforms/Platform/Ampere/buildfw.sh -b RELEASE -t GCC -p Jade -l linuxboot/mainboards/ampere/jade/flashkernel',  # noqa: E501

            'The `buildfw.sh` script automatically integrates the LinuxBoot flash kernel (provided via the -l option) as part of the final EDK2 firmware image.\n\nThis process generates a flashable EDK2 firmware image with embedded LinuxBoot, ready for deployment on the Ampere Mt. Jade platform.',  # noqa: E501

            'When powered on, the system will boot into the u-root and automatically kexec to the target OS.',

            'Run /init as init process\n1970/01/01 00:00:10 Welcome to u-root!\n...',

            "While the LinuxBoot implementation on the Ampere Mt. Jade platform represents a significant milestone, several advanced features and improvements remain to be explored. These enhancements would extend the platform's capabilities, improve its usability, and reinforce its position as a leading open source firmware solution. Key areas for future development include:",  # noqa: E501

            'One of the critical areas for future development is enabling secure boot verification for the target operating system. In the LinuxBoot environment, the target OS is typically booted using kexec. However, it is unclear how Secure Boot operates in this context, as kexec bypasses traditional firmware-controlled secure boot mechanisms. Future work should investigate how to extend Secure Boot principles to kexec, ensuring that the OS kernel and its components are verified and authenticated before execution. This may involve implementing signature checks and utilizing trusted certificate chains directly within the LinuxBoot environment to mimic the functionality of UEFI Secure Boot during the kexec process.',  # noqa: E501

            'The platform supports TPM, but its integration with LinuxBoot is yet to be defined. Future work could explore utilizing the TPM for secure boot measurements, and system integrity attestation.',  # noqa: E501

            'Building on the success of LinuxBoot on Mt. Jade, future efforts should expand support to other Ampere platforms. This would ensure broader adoption and usability across different hardware configurations.',  # noqa: E501

            'Improving the efficiency of the handoff between UEFI and LinuxBoot could further reduce boot times. This optimization would involve refining the initialization process and minimizing redundant operations during the handoff.',  # noqa: E501

            'Adding more diagnostic and monitoring tools to the LinuxBoot u-root environment would enhance debugging and system management. These tools could provide deeper insights into system performance and potential issues, improving reliability and maintainability.',  # noqa: E501

            '* [LinuxBoot on Ampere Platforms: A new (old) approach to firmware](https://amperecomputing.com/blogs/linuxboot-on-ampere-platforms--a-new-old-approach-to-firmware)']  # noqa: E501
        for i in range(len(merged)):
            assert merged[i].content == expected_merged[i]

class TestCodeSplitter:
    def test_split_xml(self):
        splitter = CodeSplitter(chunk_size=40, overlap=0, filetype='xml', keep_trace=True, keep_tags=True)
        text = '''
        <libraries>
            <library name="My Library">
                <book id="1">
                    <title>1984</title>
                    <author>Orwell</author>
                </book>
                <book id="2">
                    <title>Animal Farm</title>
                    <author>Orwell</author>
                </book>
            </library>
            <library name="My Library 2">
                <book id="1">
                    <title>1984</title>
                    <author>Orwell</author>
                </book>
                <book id="2">
                    <title>Animal Farm</title>
                    <author>Orwell</author>
                </book>
            </library>
        </libraries>
        '''
        splits = splitter._split_xml(text, chunk_size=40)
        assert len(splits) == 8

    def test_split_code(self):
        splitter = CodeSplitter(chunk_size=40, overlap=0, filetype='python', keep_trace=True, keep_tags=True)
        text = """
            class CodeSplitter(_TextSplitterBase):
                def __init__(self, chunk_size: int = 1024, overlap: int = 200, num_workers: int = 0,
                            keep_trace: bool = False, keep_tags: bool = False, **kwargs):
                    super().__init__(chunk_size=chunk_size, overlap=overlap, num_workers=num_workers)
                    self.token_splitter = _TokenTextSplitter(chunk_size=chunk_size, overlap=overlap)
                    self._keep_trace = keep_trace
                    self._keep_tags = keep_tags
                    self._dispatch_list = {
                        'xml': self._split_xml,
                        'json': self._split_json,
                        'yaml': self._split_yaml,
                        'html': self._split_html,
                        'python': self._split_code,
                        'c': self._split_code,
                        'c++': self._split_code,
                    }
                    self._filetype = filetype

                def transform(self, node: DocNode, **kwargs) -> List[DocNode]:
                    return self.split_text(
                        node.get_text(),
                        metadata_size=self._get_metadata_size(node)
                    )

                def split_text(self, text: str, metadata_size: int) -> List[DocNode]:
                    if text == '':
                        return [DocNode(text='')]
                    effective_chunk_size = self._chunk_size - metadata_size
                    if not self._filetype:
                        LOG.warning("Filetype not specified, cannot determine split method")
                        return [DocNode(text=text, metadata={'tag': 'unknown_type'})]

                    filetype = self._filetype.lower()
                    if filetype not in self._dispatch_list:
                        LOG.warning(f"Unsupported file type: {filetype}, fallback to default code splitter")
                        _split = self._split_code
                    else:
                        _split = self._dispatch_list[filetype]
                    splits = _split(text, effective_chunk_size)

                    if filetype == 'xml':
                        return splits

                    chunks = self._merge(splits, effective_chunk_size)
                    return chunks

                def _split_xml(self, text: str, chunk_size: int) -> List[DocNode]:
                    try:
                        root = ET.fromstring(text)
                    except ET.ParseError as e:
                        LOG.warning(f"Failed to parse XML: {e}. Returning original text as a single DocNode.")
                        return [DocNode(text=text, metadata={'tag': 'xml_error', 'error': str(e), 'trace': []})]

                    def _format_tag_with_attrs(tag_name: str, attributes: dict) -> str:
                        if not attributes:
                            return tag_name

                        attr_strs = []
                        for attr_name, attr_value in attributes.items():
                            attr_strs.append(f'{attr_name}="{attr_value}"')

                        return f'{tag_name} {" ".join(attr_strs)}'
        """
        splits = splitter._split_code(text, chunk_size=40)
        assert len(splits) == 68

    def test_batch_forward(self):
        splitter = CodeSplitter(chunk_size=40, overlap=0, filetype='python')
        text = """
        def main():
            print("Hello, world!")
        """
        doc_node = DocNode(text=text)
        nodes = splitter.batch_forward([doc_node], node_group='test')
        assert len(nodes) == 2


class TestTextSplitterBase:
    def test_token_size(self):
        splitter = _TextSplitterBase(chunk_size=5, overlap=0)
        text = 'Hello, world! This is a test.'
        token_size = splitter._token_size(text)
        assert token_size == 9

    def test_invalid_chunk_overlap(self):
        with pytest.raises(ValueError):
            _TextSplitterBase(chunk_size=2, overlap=10)

    def test_split(self):
        splitter = _TextSplitterBase(chunk_size=5, overlap=0)
        text = 'Hello, world! This is a test.'
        splits = splitter._split(text, chunk_size=5)
        assert splits == [
            _Split(text='Hello, world!', is_sentence=True, token_size=4),
            _Split(text='This is a test.', is_sentence=True, token_size=5),
        ]

    def test_merge(self):
        splitter = _TextSplitterBase(chunk_size=5, overlap=0)
        splits = [
            _Split(text='Hello, world!', is_sentence=True, token_size=4),
            _Split(text='This is a test.', is_sentence=True, token_size=5),
        ]
        merged = splitter._merge(splits, chunk_size=5)
        assert merged == ['Hello, world!', 'This is a test.']

    def test_split_text(self):
        splitter = _TextSplitterBase(chunk_size=5, overlap=0)
        text = 'Hello, world! This is a test.'
        splits = splitter.split_text(text, metadata_size=0)
        assert splits == ['Hello, world!', 'This is a test.']

    def test_empty_text(self):
        splitter = _TextSplitterBase(chunk_size=20, overlap=10)
        chunks = splitter.split_text('', metadata_size=0)
        assert chunks == ['']

    def test_overlap_behavior(self):
        splitter = _TextSplitterBase(chunk_size=10, overlap=2)
        text = 'abcdefghijabcdefghij'
        splits = [
            _Split(text[:10], is_sentence=True, token_size=len(splitter.token_encoder(text[:10]))),
            _Split(text[10:14], is_sentence=True, token_size=len(splitter.token_encoder(text[10:14]))),
            _Split(text[14:], is_sentence=True, token_size=len(splitter.token_encoder(text[14:])))
        ]
        chunks = splitter._merge(splits, chunk_size=10)
        assert len(chunks) >= 2
        assert splitter.token_encoder(chunks[0])[-2:] == splitter.token_encoder(chunks[1])[:2]

    def test_metadata_size_limit(self, doc_node):
        splitter = _TextSplitterBase(chunk_size=20, overlap=10)
        doc_node.get_metadata_str.return_value = 'x' * 100
        with pytest.raises(ValueError):
            splitter.split_text('短文本', metadata_size=200)

    def test_get_splits_by_fns(self):
        splitter = _TextSplitterBase(chunk_size=5, overlap=0)
        text = 'Hello, world! This is a test.'
        splits, is_sentence = splitter._get_splits_by_fns(text)
        assert splits == ['Hello, world!', 'This is a test.']
        assert is_sentence is True

    def test_get_metadata_size(self):
        splitter = _TextSplitterBase(chunk_size=20, overlap=10)
        node = DocNode(text='Hello, world! This is a test.')
        metadata_size = splitter._get_metadata_size(node)
        assert metadata_size == 0

    def test_transform_returns_chunks(self, doc_node):
        splitter = _TextSplitterBase(chunk_size=20, overlap=10)
        chunks = splitter.transform(doc_node)
        assert isinstance(chunks, list)
        assert all(isinstance(c, str) for c in chunks)

    def test_batch_forward_single(self, doc_node):
        splitter = _TextSplitterBase(chunk_size=20, overlap=10)
        doc_node.children = {}
        result = splitter.batch_forward(doc_node, node_group='test')
        assert isinstance(result, list)
        assert all(hasattr(c, 'text') for c in result)
        assert 'test' in doc_node.children

    def test_from_tiktoken_encoder_basic(self):
        splitter = _TextSplitterBase(chunk_size=100, overlap=20)

        result = splitter.from_tiktoken_encoder(encoding_name='gpt2')
        assert result is splitter
        assert splitter.token_encoder is not None
        assert splitter.token_decoder is not None

    def test_from_tiktoken_encoder_with_model_name(self):
        splitter = _TextSplitterBase(chunk_size=100, overlap=20)

        result = splitter.from_tiktoken_encoder(model_name='gpt-3.5-turbo')

        assert result is splitter
        assert splitter.token_encoder is not None
        assert splitter.token_decoder is not None

    def test_from_tiktoken_encoder_encoding_decoding(self):
        splitter = _TextSplitterBase(chunk_size=100, overlap=20)
        splitter.from_tiktoken_encoder(encoding_name='gpt2')
        text = 'Hello, world!'
        encoded = splitter.token_encoder(text)
        assert isinstance(encoded, list)
        assert len(encoded) > 0

    def test_from_tiktoken_encoder_token_size(self):
        splitter = _TextSplitterBase(chunk_size=100, overlap=20)
        splitter.from_tiktoken_encoder(encoding_name='gpt2')

        text = 'This is a test sentence.'
        token_size = splitter._token_size(text)

        assert isinstance(token_size, int)
        assert token_size > 0

    def test_from_tiktoken_encoder_chaining(self):
        splitter = _TextSplitterBase(chunk_size=100, overlap=20)

        result = splitter.from_tiktoken_encoder(encoding_name='gpt2')

        text = 'Test text'
        chunks = result.split_text(text, metadata_size=0)
        assert chunks == ['Test text']


class TestTokenTextSplitter:
    def test_token_splitter_basic(self):
        token_splitter = _TokenTextSplitter(chunk_size=10, overlap=3)
        text = 'hello world'
        splits = token_splitter._split(text, chunk_size=10)
        assert isinstance(splits, list)
        assert all(isinstance(s, _Split) for s in splits)

    def test_token_splitter_overlap_behavior(self):
        token_splitter = _TokenTextSplitter(chunk_size=10, overlap=3)
        text = 'abcdefghijabcdefghij'
        splits = token_splitter._split(text, chunk_size=10)
        chunks = token_splitter._merge(splits, chunk_size=10)
        assert len(chunks) == 1
        assert chunks[0] == 'abcdefghijabcdefghij'

    def test_token_splitter_exact_overlap(self):
        token_splitter = _TokenTextSplitter(chunk_size=10, overlap=3)
        text = 'abcdefghijabcdefghij'
        chunks = token_splitter.split_text(text, metadata_size=0)
        assert len(chunks) >= 1
        assert chunks[0] == 'abcdefghijabcdefghij'

    def test_token_splitter_short_text(self):
        token_splitter = _TokenTextSplitter(chunk_size=10, overlap=3)
        text = 'short'
        chunks = token_splitter.split_text(text, metadata_size=0)
        assert len(chunks) == 1
        assert chunks[0] == 'short'

    def test_token_splitter_exact_chunk_size(self):
        token_splitter = _TokenTextSplitter(chunk_size=10, overlap=3)
        text = 'a' * 10
        chunks = token_splitter.split_text(text, metadata_size=0)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_token_splitter_large_text(self):
        token_splitter = _TokenTextSplitter(chunk_size=10, overlap=3)
        text = 'a' * 50
        chunks = token_splitter.split_text(text, metadata_size=0)
        assert len(chunks) > 1

        for i in range(len(chunks) - 1):
            assert chunks[i][-3:] == chunks[i + 1][:3]

    def test_token_splitter_merge_returns_text_only(self):
        token_splitter = _TokenTextSplitter(chunk_size=10, overlap=3)
        text = 'abcdefghijklmnopqrst'
        splits = token_splitter._split(text, chunk_size=10)
        merged = token_splitter._merge(splits, chunk_size=10)

        assert isinstance(merged, list)
        assert all(isinstance(m, str) for m in merged)

    def test_token_splitter_transform_with_docnode(self, doc_node):
        token_splitter = _TokenTextSplitter(chunk_size=10, overlap=3)
        chunks = token_splitter.transform(doc_node)
        assert isinstance(chunks, list)
        assert all(isinstance(c, str) for c in chunks)

    def test_token_text_splitter_with_tiktoken(self):
        splitter = _TokenTextSplitter(chunk_size=10, overlap=5)
        splitter.from_tiktoken_encoder(encoding_name='gpt2')

        text = 'This is a test sentence that needs to be split into multiple chunks.'
        chunks = splitter.split_text(text, metadata_size=0)

        assert isinstance(chunks, list)
        assert len(chunks) > 1

        for i in range(len(chunks) - 1):
            tokens1 = splitter.token_encoder(chunks[i])
            tokens2 = splitter.token_encoder(chunks[i + 1])

            overlap_size = min(5, len(tokens1), len(tokens2))
            if overlap_size > 0:
                assert tokens1[-overlap_size:] == tokens2[:overlap_size]


class TestDocumentSplit:
    def setup_method(self):
        document = Document(
            dataset_path='rag_master',
            manager=False
        )
        document.create_node_group(
            name='sentence_test',
            transform=SentenceSplitter,
            chunk_size=128,
            chunk_overlap=10
        )
        document.create_node_group(
            name='character_test',
            transform=CharacterSplitter,
            chunk_size=128,
            overlap=0,
            separator=',',
            keep_separator=True
        )
        document.create_node_group(
            name='recursive_test',
            transform=RecursiveSplitter,
            chunk_size=128,
            overlap=0,
            separators=['\n\n', '\n', '.', ' ']
        )

        query = '何为天道？'

        self.document = document
        self.query = query

    def test_sentence_split(self):
        document = self.document
        document.activate_groups('sentence_test')
        document.start()
        retriever = Retriever(document, group_name='sentence_test', similarity='bm25', topk=3)
        doc_node_list = retriever(query=self.query)
        assert len(doc_node_list) == 3

    def test_character_split(self):
        document = self.document
        document.activate_groups('character_test')
        document.start()
        retriever = Retriever(document, group_name='character_test', similarity='bm25', topk=3)
        doc_node_list = retriever(query=self.query)
        assert len(doc_node_list) == 3

    def test_recursive_split(self):
        document = self.document
        document.activate_groups('recursive_test')
        document.start()
        retriever = Retriever(document, group_name='recursive_test', similarity='bm25', topk=3)
        doc_node_list = retriever(query=self.query)
        assert len(doc_node_list) == 3


class TestDocumentChainSplit:
    def setup_method(self):
        document = Document(
            dataset_path='rag_master',
            manager=False
        )
        document.create_node_group(
            name='sentence_test',
            transform=SentenceSplitter,
            chunk_size=128,
            chunk_overlap=10
        )
        document.create_node_group(
            name='recursive_test',
            transform=RecursiveSplitter,
            chunk_size=128,
            overlap=0,
            parent='sentence_test'
        )
        document.create_node_group(
            name='character_test',
            transform=CharacterSplitter,
            chunk_size=128,
            overlap=0,
            separator=' ',
            parent='recursive_test'
        )

        query = '何为天道？'

        self.document = document
        self.query = query

    def test_sentence_split(self):
        document = self.document
        document.activate_groups('sentence_test')
        document.start()
        retriever = Retriever(document, group_name='sentence_test', similarity='bm25', topk=3)
        doc_node_list = retriever(query=self.query)
        assert len(doc_node_list) == 3

    def test_recursive_split(self):
        document = self.document
        document.activate_groups('recursive_test')
        document.start()
        retriever = Retriever(document, group_name='recursive_test', similarity='bm25', topk=3)
        doc_node_list = retriever(query=self.query)
        assert len(doc_node_list) == 3

    def test_character_split(self):
        document = self.document
        document.activate_groups('character_test')
        document.start()
        retriever = Retriever(document, group_name='character_test', similarity='bm25', topk=3)
        doc_node_list = retriever(query=self.query)
        assert len(doc_node_list) == 3


class TestDIYDocumentSplit:
    def setup_method(self):
        document = Document(
            dataset_path='rag_master',
            manager=False
        )
        document.create_node_group(
            name='sentence_test',
            transform=SentenceSplitter,
            chunk_size=128,
            chunk_overlap=10
        )
        splitter = CharacterSplitter(chunk_size=128, overlap=10, separator=' ')
        splitter.set_split_fns([lambda x: x.split(' ')])
        document.create_node_group(
            name='character_test',
            transform=splitter,
            parent='sentence_test')

        query = '何为天道？'

        self.document = document
        self.query = query

    def test_character_split(self):
        document = self.document
        document.activate_groups('character_test')
        document.start()
        retriever = Retriever(document, group_name='character_test', similarity='bm25', topk=3)
        doc_node_list = retriever(query=self.query)
        assert len(doc_node_list) == 3
