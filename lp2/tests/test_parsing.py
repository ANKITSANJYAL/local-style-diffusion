from lpa2.core.parsing import split_style_content

def test_basic_parsing():
    c, s = split_style_content("a cat on a car in vaporwave style", backend="naive")
    assert "cat" in [w.lower() for w in c]
    assert "vaporwave" in [w.lower() for w in s]
