import logging
from typing import Any

import yaml
from pathlib import Path
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


def load_prompt(name: str, template_type: str = "generic", use_history: bool = False) -> PromptTemplate:
    """Load a prompt template from a YAML file in prompts/."""
    path = _PROMPTS_DIR / f"{name}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)

    templates = data.get("templates", {})
    variant = f"{template_type}_history" if use_history else template_type

    # Fall back: specific variant -> template_type -> generic -> generic without history
    template_str = (
        templates.get(variant)
        or templates.get(template_type)
        or templates.get("generic_history" if use_history else "generic")
        or templates.get("generic")
    )

    if template_str is None:
        raise ValueError(f"No template found for variant '{variant}' in {path}")

    input_variables = data.get("input_variables", ["context", "question"])
    if use_history and "history" not in input_variables:
        input_variables = ["history"] + input_variables
    if "system_prompt" in template_str and "system_prompt" not in input_variables:
        input_variables = ["system_prompt"] + input_variables

    return PromptTemplate(input_variables=input_variables, template=template_str)


class Generator:
    """Generates answers from retrieved context using an LLM."""

    def __init__(
        self,
        llm: Any,
        system_prompt: str = "",
        prompt_name: str = "qa",
        template_type: str = "generic",
        use_history: bool = False,
    ):
        self._llm = llm
        self._prompt = load_prompt(prompt_name, template_type, use_history)
        self._use_history = use_history
        self._system_prompt = system_prompt

    def generate(
        self,
        query: str,
        documents: list[Document],
        history: str = "",
    ) -> dict:
        """Generate an answer with source citations.

        Returns:
            dict with keys: answer (str), sources (list[dict])
        """
        context = "\n\n".join(doc.page_content for doc in documents)

        prompt_kwargs = {"context": context, "question": query, "system_prompt": self._system_prompt}
        if self._use_history:
            prompt_kwargs["history"] = history

        formatted = self._prompt.format(**prompt_kwargs)
        logger.info(f"Generating answer for query (prompt length: {len(formatted)} chars)")

        answer = self._llm.invoke(formatted)

        sources = []
        for doc in documents:
            source_entry = {
                "file": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
                "excerpt": doc.page_content[:200],
            }
            sources.append(source_entry)

        return {"answer": answer, "sources": sources}
