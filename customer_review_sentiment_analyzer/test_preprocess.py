import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from preprocess import (
    remove_html_tags,
    remove_urls,
    remove_special_characters,
    normalize_whitespace,
    clean_text,
    map_rating_to_sentiment,
)

class TestRemoveHtmlTags:
    def test_removes_paragraph_tags(self):
        assert remove_html_tags("<p>Hello world</p>") == "Hello world"

    def test_removes_br_tags(self):
        assert remove_html_tags("Line one<br>Line two") == "Line oneLine two"

    def test_removes_nested_tags(self):
        assert remove_html_tags("<div><span>text</span></div>") == "text"

    def test_no_html_returns_unchanged(self):
        text = "Plain text with no HTML"
        assert remove_html_tags(text) == text

    def test_empty_string(self):
        assert remove_html_tags("") == ""


class TestRemoveUrls:
    def test_removes_https_url(self):
        result = remove_urls("Visit https://example.com for details")
        assert "https://example.com" not in result

    def test_removes_http_url(self):
        result = remove_urls("See http://buy.com/product")
        assert "http://buy.com/product" not in result

    def test_removes_www_url(self):
        result = remove_urls("Check www.site.com today")
        assert "www.site.com" not in result

    def test_no_url_unchanged(self):
        text = "No URLs here at all"
        assert remove_urls(text) == text


class TestRemoveSpecialCharacters:
    def test_removes_exclamation(self):
        result = remove_special_characters("Amazing!!!")
        assert "!" not in result

    def test_removes_at_sign(self):
        result = remove_special_characters("Hello @user")
        assert "@" not in result

    def test_keeps_alphabets(self):
        result = remove_special_characters("Hello World")
        assert "Hello" in result and "World" in result

    def test_keeps_digits(self):
        result = remove_special_characters("100 percent")
        assert "100" in result


class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self):
        assert normalize_whitespace("too   many   spaces") == "too many spaces"

    def test_strips_leading_trailing(self):
        assert normalize_whitespace("  hello  ") == "hello"

    def test_handles_tabs(self):
        assert normalize_whitespace("word\t\tword") == "word word"

    def test_handles_newlines(self):
        assert normalize_whitespace("line1\nline2") == "line1 line2"


class TestCleanText:
    def test_positive_review(self):
        text = "This product is AMAZING and works perfectly!"
        result = clean_text(text)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "!" not in result
        assert result == result.lower()

    def test_negative_review(self):
        text = "Terrible quality. I hate this product completely."
        result = clean_text(text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_html_in_review(self):
        text = "<b>Great product</b> highly recommended!"
        result = clean_text(text)
        assert "<" not in result
        assert ">" not in result

    def test_url_in_review(self):
        text = "Buy this at https://amazon.com - great deal!"
        result = clean_text(text)
        assert "https" not in result
        assert "amazon.com" not in result

    def test_non_string_input(self):
        result = clean_text(12345)
        assert isinstance(result, str)

    def test_empty_string(self):
        result = clean_text("")
        assert isinstance(result, str)

    def test_all_special_chars(self):
        result = clean_text("!!!@@@###$$$")
        assert isinstance(result, str)


class TestMapRatingToSentiment:
    def test_five_star_is_positive(self):
        assert map_rating_to_sentiment(5) == "Positive"

    def test_four_star_is_positive(self):
        assert map_rating_to_sentiment(4) == "Positive"

    def test_three_star_is_neutral(self):
        assert map_rating_to_sentiment(3) == "Neutral"

    def test_two_star_is_negative(self):
        assert map_rating_to_sentiment(2) == "Negative"

    def test_one_star_is_negative(self):
        assert map_rating_to_sentiment(1) == "Negative"

    def test_all_ratings_covered(self):
        valid_sentiments = {"Positive", "Neutral", "Negative"}
        for rating in range(1, 6):
            assert map_rating_to_sentiment(rating) in valid_sentiments

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
