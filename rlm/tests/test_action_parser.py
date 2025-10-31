"""
Unit tests for action_schemas module.

Tests all three action types and edge cases for the parser.
"""

import pytest
from rlm.action_schemas import (
    Action,
    PyAction,
    FinalAction,
    FinalVarAction,
    parse_action,
    format_action,
)


class TestPyAction:
    """Tests for PY(...) action parsing."""
    
    def test_simple_py_action(self):
        """Test parsing a simple Python action."""
        text = 'PY("""x = 1 + 1""")'
        action = parse_action(text)
        assert isinstance(action, PyAction)
        assert action.code == "x = 1 + 1"
    
    def test_py_action_with_newlines(self):
        """Test parsing Python action with multiple lines."""
        text = '''PY("""
import re
x = context['document']
print(x[:100])
""")'''
        action = parse_action(text)
        assert isinstance(action, PyAction)
        assert "import re" in action.code
        assert "context['document']" in action.code
        assert "print(x[:100])" in action.code
    
    def test_py_action_with_nested_quotes(self):
        """Test parsing Python action with nested single quotes."""
        text = '''PY("""print(context['document'])""")'''
        action = parse_action(text)
        assert isinstance(action, PyAction)
        assert action.code == "print(context['document'])"
    
    def test_py_action_with_special_chars(self):
        """Test parsing Python action with special characters."""
        text = r'''PY("""text = "Hello\nWorld\t!"
pattern = r"\d+"
matches = re.findall(pattern, text)""")'''
        action = parse_action(text)
        assert isinstance(action, PyAction)
        assert "Hello\\nWorld\\t!" in action.code
        assert r"\d+" in action.code
    
    def test_py_action_in_context(self):
        """Test parsing PY action when surrounded by other text."""
        text = "Let me inspect the context first. PY(\"\"\"print(len(context))\"\"\")"
        action = parse_action(text)
        assert isinstance(action, PyAction)
        assert action.code == "print(len(context))"


class TestFinalAction:
    """Tests for FINAL(...) action parsing."""
    
    def test_simple_final_action(self):
        """Test parsing a simple FINAL action."""
        text = 'FINAL("""The answer is 42""")'
        action = parse_action(text)
        assert isinstance(action, FinalAction)
        assert action.answer == "The answer is 42"
    
    def test_final_action_multiline(self):
        """Test parsing FINAL action with multiline answer."""
        text = '''FINAL("""The capital of France is Paris.
It is located in the northern part of the country.""")'''
        action = parse_action(text)
        assert isinstance(action, FinalAction)
        assert "Paris" in action.answer
        assert "northern part" in action.answer
    
    def test_final_action_with_quotes(self):
        """Test parsing FINAL action containing single quotes."""
        text = '''FINAL("""The author's main point is clear.""")'''
        action = parse_action(text)
        assert isinstance(action, FinalAction)
        assert action.answer == "The author's main point is clear."
    
    def test_final_action_empty(self):
        """Test parsing FINAL action with empty answer."""
        text = 'FINAL("""""")'
        action = parse_action(text)
        assert isinstance(action, FinalAction)
        assert action.answer == ""
    
    def test_final_action_in_context(self):
        """Test parsing FINAL action surrounded by explanatory text."""
        text = 'Based on my analysis, FINAL("""Paris""")'
        action = parse_action(text)
        assert isinstance(action, FinalAction)
        assert action.answer == "Paris"


class TestFinalVarAction:
    """Tests for FINAL_VAR(...) action parsing."""
    
    def test_final_var_with_double_quotes(self):
        """Test parsing FINAL_VAR with double quotes."""
        text = 'FINAL_VAR(var_name="result")'
        action = parse_action(text)
        assert isinstance(action, FinalVarAction)
        assert action.var_name == "result"
    
    def test_final_var_with_single_quotes(self):
        """Test parsing FINAL_VAR with single quotes."""
        text = "FINAL_VAR(var_name='answer')"
        action = parse_action(text)
        assert isinstance(action, FinalVarAction)
        assert action.var_name == "answer"
    
    def test_final_var_with_underscore(self):
        """Test parsing FINAL_VAR with underscored variable name."""
        text = 'FINAL_VAR(var_name="final_count")'
        action = parse_action(text)
        assert isinstance(action, FinalVarAction)
        assert action.var_name == "final_count"
    
    def test_final_var_in_context(self):
        """Test parsing FINAL_VAR surrounded by text."""
        text = "The result is stored in the variable. FINAL_VAR(var_name=\"count\")"
        action = parse_action(text)
        assert isinstance(action, FinalVarAction)
        assert action.var_name == "count"


class TestParserEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_none_input(self):
        """Test that None input returns None."""
        assert parse_action(None) is None
    
    def test_empty_string(self):
        """Test that empty string returns None."""
        assert parse_action("") is None
    
    def test_invalid_action(self):
        """Test that invalid action format returns None."""
        text = "INVALID_ACTION(something)"
        assert parse_action(text) is None
    
    def test_malformed_py(self):
        """Test that malformed PY action returns None."""
        text = 'PY("single quotes only")'
        assert parse_action(text) is None
    
    def test_malformed_final(self):
        """Test that malformed FINAL action returns None."""
        text = 'FINAL("single quotes")'
        assert parse_action(text) is None
    
    def test_malformed_final_var(self):
        """Test that malformed FINAL_VAR returns None."""
        text = 'FINAL_VAR(name="wrong_param")'
        assert parse_action(text) is None
    
    def test_text_without_action(self):
        """Test that regular text returns None."""
        text = "This is just some regular text without any action."
        assert parse_action(text) is None
    
    def test_unclosed_triple_quotes(self):
        """Test that unclosed triple quotes returns None."""
        text = 'PY("""incomplete'
        assert parse_action(text) is None
    
    def test_multiple_actions_returns_first(self):
        """Test that when multiple actions present, first match is returned."""
        text = 'FINAL_VAR(var_name="x") and also PY("""y=1""")'
        action = parse_action(text)
        # FINAL_VAR is checked first in parse_action
        assert isinstance(action, FinalVarAction)
        assert action.var_name == "x"


class TestFormatAction:
    """Tests for format_action function."""
    
    def test_format_py_action(self):
        """Test formatting PyAction back to string."""
        action = PyAction(code="x = 1 + 1")
        formatted = format_action(action)
        assert formatted == 'PY("""x = 1 + 1""")'
    
    def test_format_final_action(self):
        """Test formatting FinalAction back to string."""
        action = FinalAction(answer="The answer is 42")
        formatted = format_action(action)
        assert formatted == 'FINAL("""The answer is 42""")'
    
    def test_format_final_var_action(self):
        """Test formatting FinalVarAction back to string."""
        action = FinalVarAction(var_name="result")
        formatted = format_action(action)
        assert formatted == 'FINAL_VAR(var_name="result")'
    
    def test_format_roundtrip_py(self):
        """Test that format -> parse roundtrip works for PY."""
        original = PyAction(code="print('hello')")
        formatted = format_action(original)
        parsed = parse_action(formatted)
        assert isinstance(parsed, PyAction)
        assert parsed.code == original.code
    
    def test_format_roundtrip_final(self):
        """Test that format -> parse roundtrip works for FINAL."""
        original = FinalAction(answer="42")
        formatted = format_action(original)
        parsed = parse_action(formatted)
        assert isinstance(parsed, FinalAction)
        assert parsed.answer == original.answer
    
    def test_format_roundtrip_final_var(self):
        """Test that format -> parse roundtrip works for FINAL_VAR."""
        original = FinalVarAction(var_name="result")
        formatted = format_action(original)
        parsed = parse_action(formatted)
        assert isinstance(parsed, FinalVarAction)
        assert parsed.var_name == original.var_name


class TestRealWorldExamples:
    """Tests with realistic model outputs."""
    
    def test_model_output_with_explanation(self):
        """Test parsing action from model output with explanation."""
        text = """Let me first check what's in the context variable.
        
PY(\"\"\"
print(type(context))
print(list(context.keys()))
\"\"\")

This will help me understand the structure."""
        action = parse_action(text)
        assert isinstance(action, PyAction)
        assert "print(type(context))" in action.code
    
    def test_model_output_final_with_reasoning(self):
        """Test parsing FINAL from output with reasoning."""
        text = """Based on my analysis of the document, I can see that the capital 
is mentioned in the third paragraph. The answer is:

FINAL(\"\"\"Paris\"\"\")"""
        action = parse_action(text)
        assert isinstance(action, FinalAction)
        assert action.answer == "Paris"
    
    def test_model_output_with_markdown(self):
        """Test parsing action when model uses markdown formatting."""
        text = """I'll execute this code:

```
PY(\"\"\"count = len(context['items'])\"\"\")
```"""
        action = parse_action(text)
        assert isinstance(action, PyAction)
        assert action.code == "count = len(context['items'])"
    
    def test_verbose_model_output(self):
        """Test parsing from verbose model response."""
        text = """Certainly! I'll count the occurrences for you. First, I'll 
write some Python code to count the word in the document:

PY(\"\"\"
word = 'python'
count = context['document'].lower().count(word)
print(f"Found {count} occurrences of '{word}'")
\"\"\")

Let me know what happens!"""
        action = parse_action(text)
        assert isinstance(action, PyAction)
        assert "count = context['document'].lower().count(word)" in action.code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
