def test_credit_to_income_ratio():
    income = 100000
    credit = 300000
    ratio = credit / income
    assert ratio == 3.0

def test_annuity_to_income_ratio():
    income = 100000
    annuity = 25000
    ratio = annuity / income
    assert ratio == 0.25

def test_ratio_zero_income():
    income = 0
    credit = 300000
    try:
        ratio = credit / income
        assert False, "Should have raised ZeroDivisionError"
    except ZeroDivisionError:
        pass