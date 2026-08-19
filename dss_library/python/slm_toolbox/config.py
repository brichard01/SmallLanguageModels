"""Which LLM the toolbox runs on.

Held in the project variable `method_llm_id` so an SME can swap the whole
toolbox between a frontier model and a local SLM in one place, without touching
any agent code. Falls back to gpt-5 if the variable is absent.
"""

import dataiku

DEFAULT_LLM_ID = "openai:openai:gpt-5"


def llm_id():
    try:
        return dataiku.get_custom_variables().get("method_llm_id") or DEFAULT_LLM_ID
    except Exception:
        return DEFAULT_LLM_ID


def get_llm(llm_id_override=None):
    project = dataiku.api_client().get_default_project()
    return project.get_llm(llm_id_override or llm_id())
