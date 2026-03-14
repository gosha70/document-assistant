"""Tests for prompt_info.py — verifies bug fixes and prompt generation."""
import pytest
from prompt_info import PromptInfo, LLAMA_MODEL_NAME, MISTRAL_MODEL_NAME


class TestPromptInfoBugFixes:
    """Verify the bug fixes in prompt_info.py."""

    def test_template_type_setter_works(self):
        """Bug fix: setter was named 'c' instead of 'template_type'."""
        pi = PromptInfo("test prompt", LLAMA_MODEL_NAME, False)
        pi.template_type = MISTRAL_MODEL_NAME
        assert pi.template_type == MISTRAL_MODEL_NAME

    def test_use_history_setter_stores_value(self):
        """Bug fix: setter assigned ValueError class instead of value."""
        pi = PromptInfo("test prompt", LLAMA_MODEL_NAME, False)
        pi.use_history = True
        assert pi.use_history is True

    def test_use_history_setter_stores_false(self):
        """Ensure use_history setter works for False too."""
        pi = PromptInfo("test prompt", LLAMA_MODEL_NAME, True)
        pi.use_history = False
        assert pi.use_history is False


class TestPromptInfoInit:
    def test_defaults_to_llama_template(self):
        pi = PromptInfo("prompt", None, None)
        assert pi.template_type == LLAMA_MODEL_NAME
        assert pi.use_history is False

    def test_explicit_values(self):
        pi = PromptInfo("prompt", MISTRAL_MODEL_NAME, True)
        assert pi.template_type == MISTRAL_MODEL_NAME
        assert pi.use_history is True
        assert pi.system_prompt == "prompt"


class TestPromptTemplateGeneration:
    def test_llama_no_history(self):
        pi = PromptInfo("You are helpful.", LLAMA_MODEL_NAME, False)
        prompt, memory = pi.get_prompt_template()
        assert "context" in prompt.input_variables
        assert "question" in prompt.input_variables
        assert "history" not in prompt.input_variables
        assert "You are helpful." in prompt.template

    def test_llama_with_history(self):
        pi = PromptInfo("You are helpful.", LLAMA_MODEL_NAME, True)
        prompt, memory = pi.get_prompt_template()
        assert "history" in prompt.input_variables
        assert "context" in prompt.input_variables
        assert "question" in prompt.input_variables

    def test_mistral_no_history(self):
        pi = PromptInfo("You are helpful.", MISTRAL_MODEL_NAME, False)
        prompt, memory = pi.get_prompt_template()
        assert "context" in prompt.input_variables
        assert "question" in prompt.input_variables
        assert "history" not in prompt.input_variables

    def test_mistral_with_history(self):
        pi = PromptInfo("You are helpful.", MISTRAL_MODEL_NAME, True)
        prompt, memory = pi.get_prompt_template()
        assert "history" in prompt.input_variables

    def test_generic_template(self):
        pi = PromptInfo("You are helpful.", "unknown_model", False)
        prompt, memory = pi.get_prompt_template()
        assert "context" in prompt.input_variables
        assert "question" in prompt.input_variables
        assert "Answer:" in prompt.template

    def test_memory_is_returned_when_history_enabled(self):
        pi = PromptInfo("test", LLAMA_MODEL_NAME, True)
        prompt, memory = pi.get_prompt_template()
        assert memory is not None

    def test_memory_is_base_memory_subclass_when_history_enabled(self):
        """Memory must be a BaseMemory subclass for RetrievalQA compatibility."""
        from langchain_core.memory import BaseMemory
        pi = PromptInfo("test", LLAMA_MODEL_NAME, True)
        prompt, memory = pi.get_prompt_template()
        assert isinstance(memory, BaseMemory)

    def test_memory_is_none_when_history_disabled(self):
        pi = PromptInfo("test", LLAMA_MODEL_NAME, False)
        prompt, memory = pi.get_prompt_template()
        assert memory is None

    def test_str_representation(self):
        pi = PromptInfo("test", LLAMA_MODEL_NAME, False)
        s = str(pi)
        assert "PromptInfo" in s
        assert "test" in s
