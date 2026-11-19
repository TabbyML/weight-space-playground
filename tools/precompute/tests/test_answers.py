from tools.precompute.answers import is_correct, last_boxed, normalize


def test_last_boxed_nested_braces():
    assert last_boxed(r"so \boxed{1} then \boxed{\frac{1}{2}}.") == r"\frac{1}{2}"


def test_last_boxed_unclosed_and_missing():
    assert last_boxed("no box here") is None
    assert last_boxed(r"\boxed{12} and \boxed{3") == "12"


def test_normalize_latex_variants():
    assert normalize(r"\dfrac{3}{4}") == r"\frac{3}{4}"
    assert normalize(r"x = 5") == "5"
    assert normalize(r"1{,}048") == "1048"
    assert normalize(r"\text{ 42 }.") == "42"
    assert normalize(r"30^\circ") == "30"


def test_is_correct_numeric_and_string():
    assert is_correct("1048", "1048")
    assert is_correct("1,048", "1048")
    assert is_correct(r"\frac{4}{2}", "2")
    assert is_correct("2.0", "2")
    assert not is_correct("1028", "1048")
    assert not is_correct(None, "1048")
