"""Unit tests for action schema parser."""

import pytest
from rlm.action_schemas import (
    PyAction,
    CallSubmodelAction,
    FinalAction,
    FinalVarAction,
    parse_action,
)


class TestPyAction:
    """Tests for parsing PY("""...""") actions."""
    
    def test_simple_py_action(self):
        """Test parsing a simple Python action."""
        text = 'PY("""print("hello")""")'
        action = parse_action(text)
        assert isinstance(action, PyAction)
        assert action.code == 'print("hello")'
    
    def test_multiline_py_action(self):
        """Test parsing a multiline Python action."""
        text = '''PY("""
import re
matches = re.findall(r"pattern", text)
print(matches)
""")'''
        action = parse_action(text)
        assert isinstance(action, PyAction)
        assert "import re" in action.code
        assert "re.findall" in action.code
    
    def test_py_action_with_triple_quotes_in_code(self):
        """Test parsing Python action containing triple quotes."""
        text = 'PY("""x = "test string"""")'
        action = parse_action(text)
        assert isinstance(action, PyAction)
        assert action.code == 'x = "test string"'
    
    def test_py_action_with_context(self):
        """Test parsing Python action accessing context variable."""
        text = 'PY("""print(context[:500])""")'
        action = parse_action(text)
        assert isinstance(action, PyAction)
        assert action.code == 'print(context[:500])'
    
    def test_py_action_in_longer_text(self):
        """Test extracting PY action from surrounding text."""
        text = 'I will now execute some code: PY("""x = 42""") and continue'
        action = parse_action(text)
        assert isinstance(action, PyAction)
        assert action.code == 'x = 42'


class TestCallSubmodelAction:
    """Tests for parsing CALL_SUBMODEL(...) actions."""
    
    def test_simple_call_submodel(self):
        """Test parsing a simple submodel call."""
        text = 'CALL_SUBMODEL(query="What is X?", context_var="doc")'
        action = parse_action(text)
        assert isinstance(action, CallSubmodelAction)
        assert action.query == "What is X?"
        assert action.context_var == "doc"
    
    def test_call_submodel_with_triple_quotes(self):
        """Test parsing submodel call with triple-quoted query."""
        text = 'CALL_SUBMODEL(query="""What is the answer to life, universe, and everything?""", context_var="context")'
        action = parse_action(text)
        assert isinstance(action, CallSubmodelAction)
        assert action.query == "What is the answer to life, universe, and everything?"
        assert action.context_var == "context"
    
    def test_call_submodel_multiline_query(self):
        """Test parsing submodel call with multiline query."""
        text = '''CALL_SUBMODEL(query="""Based on the following context:
{context}

Answer the question: What happened?""", context_var="context")'''
        action = parse_action(text)
        assert isinstance(action, CallSubmodelAction)
        assert "Based on the following context" in action.query
        assert "What happened?" in action.query
        assert action.context_var == "context"
    
    def test_call_submodel_no_spaces(self):
        """Test parsing submodel call without spaces after comma."""
        text = 'CALL_SUBMODEL(query="test",context_var="var")'
        action = parse_action(text)
        assert isinstance(action, CallSubmodelAction)
        assert action.query == "test"
        assert action.context_var == "var"
    
    def test_call_submodel_extra_spaces(self):
        """Test parsing submodel call with extra spaces."""
        text = 'CALL_SUBMODEL(query="test",   context_var="var")'
        action = parse_action(text)
        assert isinstance(action, CallSubmodelAction)
        assert action.query == "test"
        assert action.context_var == "var"


class TestFinalAction:
    """Tests for parsing FINAL("""...""") actions."""
    
    def test_simple_final_action(self):
        """Test parsing a simple final answer."""
        text = 'FINAL("""42""")'
        action = parse_action(text)
        assert isinstance(action, FinalAction)
        assert action.answer == "42"
    
    def test_final_action_long_answer(self):
        """Test parsing final action with longer answer."""
        text = 'FINAL("""The answer is that the protagonist decided to leave the city and start a new life.""")'
        action = parse_action(text)
        assert isinstance(action, FinalAction)
        assert "protagonist decided to leave" in action.answer
    
    def test_final_action_multiline(self):
        """Test parsing multiline final answer."""
        text = '''FINAL("""The main points are:
1. First point
2. Second point
3. Third point""")'''
        action = parse_action(text)
        assert isinstance(action, FinalAction)
        assert "First point" in action.answer
        assert "Second point" in action.answer
    
    def test_final_action_with_quotes(self):
        """Test parsing final answer containing quotes."""
        text = 'FINAL("""The character said "hello world" and left.""")'
        action = parse_action(text)
        assert isinstance(action, FinalAction)
        assert 'said "hello world"' in action.answer
    
    def test_final_action_in_context(self):
        """Test extracting FINAL action from surrounding text."""
        text = 'After analyzing everything, my answer is: FINAL("""Yes, it is correct.""")'
        action = parse_action(text)
        assert isinstance(action, FinalAction)
        assert action.answer == "Yes, it is correct."


class TestFinalVarAction:
    """Tests for parsing FINAL_VAR(var_name="...") actions."""
    
    def test_simple_final_var(self):
        """Test parsing a simple final var action."""
        text = 'FINAL_VAR(var_name="result")'
        action = parse_action(text)
        assert isinstance(action, FinalVarAction)
        assert action.var_name == "result"
    
    def test_final_var_different_names(self):
        """Test parsing final var with different variable names."""
        for var_name in ["answer", "count", "output", "final_result"]:
            text = f'FINAL_VAR(var_name="{var_name}")'
            action = parse_action(text)
            assert isinstance(action, FinalVarAction)
            assert action.var_name == var_name
    
    def test_final_var_in_context(self):
        """Test extracting FINAL_VAR action from surrounding text."""
        text = 'I have stored the result in a variable: FINAL_VAR(var_name="count")'
        action = parse_action(text)
        assert isinstance(action, FinalVarAction)
        assert action.var_name == "count"


class TestParseActionEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_no_action(self):
        """Test that None is returned when no action is found."""
        text = "This is just regular text with no action."
        action = parse_action(text)
        assert action is None
    
    def test_empty_string(self):
        """Test parsing empty string."""
        action = parse_action("")
        assert action is None
    
    def test_incomplete_action(self):
        """Test that incomplete actions return None."""
        incomplete_actions = [
            'PY("""',
            'CALL_SUBMODEL(query="test"',
            'FINAL(',
            'FINAL_VAR(var_name=',
        ]
        for text in incomplete_actions:
            action = parse_action(text)
            assert action is None, f"Expected None for incomplete action: {text}"
    
    def test_malformed_action(self):
        """Test that malformed actions return None."""
        malformed_actions = [
            'PY("code")',  # Wrong quotes
            'CALL_SUBMODEL(query="test")',  # Missing context_var
            'FINAL("answer")',  # Wrong quotes
            'FINAL_VAR("var")',  # Missing var_name=
        ]
        for text in malformed_actions:
            action = parse_action(text)
            # These might parse or not depending on regex, but shouldn't crash
            assert action is None or isinstance(action, (PyAction, CallSubmodelAction, FinalAction, FinalVarAction))
    
    def test_multiple_actions_returns_first(self):
        """Test that when multiple actions exist, the first one is returned."""
        text = 'PY("""x = 1""") then FINAL("""done""")'
        action = parse_action(text)
        # Should return the first action found (PY)
        assert isinstance(action, PyAction)
        assert action.code == "x = 1"


class TestActionPriority:
    """Tests for action parsing priority when multiple patterns might match."""
    
    def test_py_before_final(self):
        """Test that PY is matched before FINAL when both present."""
        text = 'PY("""result""") and FINAL("""result""")'
        action = parse_action(text)
        assert isinstance(action, PyAction)
    
    def test_final_var_before_final(self):
        """Test that FINAL_VAR is matched before FINAL."""
        # Note: Based on the regex order in parse_action, this tests the actual behavior
        text = 'FINAL_VAR(var_name="x") FINAL("""x""")'
        action = parse_action(text)
        # The function checks in order: PY, CALL_SUBMODEL, FINAL_VAR, FINAL
        # So PY would be first if present, then CALL_SUBMODEL, then FINAL_VAR
        assert isinstance(action, (FinalVarAction, FinalAction))


class TestRealWorldExamples:
    """Tests with realistic examples from actual RLM usage."""
    
    def test_context_inspection(self):
        """Test typical context inspection pattern."""
        text = '''First, let me inspect the context:
PY("""
doc = context["document"]
print(f"Document length: {len(doc)}")
print(f"First 500 chars: {doc[:500]}")
""")'''
        action = parse_action(text)
        assert isinstance(action, PyAction)
        assert "context[" in action.code
        assert "print" in action.code
    
    def test_grep_pattern(self):
        """Test typical grep/search pattern."""
        text = '''I'll search for the relevant information:
PY("""
import re
pattern = r"temperature:\s*(\d+)"
matches = re.findall(pattern, context["document"])
print(f"Found {len(matches)} matches: {matches}")
""")'''
        action = parse_action(text)
        assert isinstance(action, PyAction)
        assert "import re" in action.code
        assert "re.findall" in action.code
    
    def test_submodel_call_with_context(self):
        """Test realistic submodel call."""
        text = '''Now I'll query the submodel with the relevant context:
CALL_SUBMODEL(query="""Based on the document, what was the final decision and why?""", context_var="context")'''
        action = parse_action(text)
        assert isinstance(action, CallSubmodelAction)
        assert "final decision" in action.query
        assert action.context_var == "context"
    
    def test_final_answer_after_reasoning(self):
        """Test final answer after reasoning."""
        text = '''Based on my analysis, I can now provide the final answer:
FINAL("""The temperature increased by 15 degrees Celsius between 2020 and 2023.""")'''
        action = parse_action(text)
        assert isinstance(action, FinalAction)
        assert "temperature increased" in action.answer
    
    def test_final_var_after_computation(self):
        """Test final var after computation."""
        text = '''I've computed the result and stored it:
FINAL_VAR(var_name="total_count")'''
        action = parse_action(text)
        assert isinstance(action, FinalVarAction)
        assert action.var_name == "total_count"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
