"""
Unit tests for the shared preprocessing module. These run on every push
as part of the CI workflow (03-ci-tests.yaml).
"""
import pytest

from training.preprocess import chunk_text, clean_text, is_valid_length


class TestCleanText:
    def test_collapses_whitespace(self):
        assert clean_text("hello   \n\t world") == "hello world"

    def test_strips_control_chars(self):
        assert clean_text("hello\x00world\u200b") == "helloworld"

    def test_strips_leading_trailing(self):
        assert clean_text("   hello   ") == "hello"

    def test_raises_on_non_string(self):
        with pytest.raises(TypeError):
            clean_text(123)


class TestValidLength:
    def test_short_input_rejected(self):
        assert is_valid_length("too short") is False

    def test_long_input_accepted(self):
        assert is_valid_length("a" * 25) is True

    def test_padding_only_rejected(self):
        assert is_valid_length("   short   ") is False


class TestChunkText:
    def test_short_text_returns_one_chunk(self):
        out = chunk_text("hello world")
        assert len(out) == 1
        assert out[0] == "hello world"

    def test_long_text_chunked(self):
        long = "x" * 3500
        out = chunk_text(long, chunk_chars=1500)
        assert len(out) >= 2
        # Each chunk should be at most chunk_chars
        assert all(len(c) <= 1500 for c in out)

    def test_chunks_overlap(self):
        long = "abcdefg" * 300  # 2100 chars
        out = chunk_text(long, chunk_chars=1500)
        # Second chunk should start before the first chunk ends (overlap)
        assert len(out) == 2
        first_end = out[0][-100:]
        assert first_end in long
