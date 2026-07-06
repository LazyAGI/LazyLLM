CONTENT_SUMMARY_PROMPT = '''\
You are summarizing a draft document section for a writer agent.

The content below is extracted from a draft. The title may appear first, followed by the body text.

Summarize the main topic and key points in ONE sentence so the writer agent knows what this draft covers.

Content:
---
{content}
---

Return JSON: {{"summary": "one-sentence summary of the draft's main topic and key content"}}
'''
