"""Tests for vocabulary loading and validation."""

import json

import pytest

from uitag.vocab import load_vocab


class TestLoadBuiltin:
    def test_load_leith_17(self):
        vocab = load_vocab("leith-17")
        assert vocab.name == "leith-17"
        assert len(vocab.types) == 17
        assert "button" in vocab.types
        assert "text_field" in vocab.types
        assert "text_label" in vocab.types
        assert vocab.fallback_type == "other"
        assert vocab.padding_pct == 25

    def test_load_rico_25(self):
        vocab = load_vocab("rico-25")
        assert vocab.name == "rico-25"
        assert len(vocab.types) == 25

    def test_load_screenvlm_55(self):
        vocab = load_vocab("screenvlm-55")
        assert vocab.name == "screenvlm-55"
        assert len(vocab.types) == 56

    def test_unknown_builtin_raises(self):
        with pytest.raises(FileNotFoundError, match="not-a-vocab"):
            load_vocab("not-a-vocab")


class TestLoadCustom:
    def test_load_from_path(self, tmp_path):
        vocab_data = {
            "name": "custom",
            "version": "1.0",
            "description": "Test vocab",
            "types": ["a", "b", "c"],
            "prompt_template": 'Pick one: {types_list}. JSON: {"element_type": "<type>"}',
            "fallback_type": "c",
            "padding_pct": 10,
        }
        vocab_path = tmp_path / "custom.json"
        vocab_path.write_text(json.dumps(vocab_data))
        vocab = load_vocab(str(vocab_path))
        assert vocab.name == "custom"
        assert vocab.types == ["a", "b", "c"]

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_vocab("/nonexistent/path.json")


class TestValidation:
    def _write_vocab(self, tmp_path, overrides):
        base = {
            "name": "test",
            "version": "1.0",
            "description": "test",
            "types": ["a", "b"],
            "prompt_template": "Pick: {types_list}",
            "fallback_type": "b",
            "padding_pct": 25,
        }
        base.update(overrides)
        path = tmp_path / "test.json"
        path.write_text(json.dumps(base))
        return str(path)

    def test_empty_types_raises(self, tmp_path):
        path = self._write_vocab(tmp_path, {"types": []})
        with pytest.raises(ValueError, match="types"):
            load_vocab(path)

    def test_missing_types_list_placeholder_raises(self, tmp_path):
        path = self._write_vocab(tmp_path, {"prompt_template": "No placeholder here"})
        with pytest.raises(ValueError, match="types_list"):
            load_vocab(path)

    def test_fallback_not_in_types_raises(self, tmp_path):
        path = self._write_vocab(tmp_path, {"fallback_type": "z"})
        with pytest.raises(ValueError, match="fallback_type"):
            load_vocab(path)


class TestPromptGeneration:
    def test_build_prompt(self):
        vocab = load_vocab("leith-17")
        prompt = vocab.build_prompt()
        assert "button" in prompt
        assert "text_field" in prompt
        assert "{types_list}" not in prompt
