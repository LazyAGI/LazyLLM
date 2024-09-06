from lazyllm.tools.tools import Calculator

class TestCalculator(object):
    def setup_method(self):
        self._calc = Calculator()

    def test_calculator(self):
        res = self._calc('(12*13)/6')
        assert res == 26

    def test_invalid_import(self):
        try:
            value = 123
            self._calc('import(os)')
            value = 456
        except Exception:
            value = 789
        finally:
            assert value == 789

    def test_math_func(self):
        res = self._calc('fabs(-5)')
        assert res == 5
