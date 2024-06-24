import lazyllm

class TestOption(object):

    def test_option(self):
        l1 = [1, 2]
        l2 = [3, 4, 5]
        o1 = lazyllm.Option(l1)
        o2 = lazyllm.Option(l2)

        expected_output = [[1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]]
        assert list(lazyllm.OptionIter([o1, o2])) == expected_output

    def test_test(self):

        def get_options(x):
            if isinstance(x, lazyllm.Option):
                return [x]
            else:
                return []
        o1 = lazyllm.Option([1, 2])
        o2 = lazyllm.Option([o1, 3, 4])
        o3 = lazyllm.Option([5, 6])

        expected_output = ('[[<Option options="[1, 2]" curr="1">, 5, 1], [<Option options="[1, 2]" curr="1">, 5, 2], '
                           '[<Option options="[1, 2]" curr="1">, 6, 1], [<Option options="[1, 2]" curr="1">, 6, 2], '
                           '[3, 5], [3, 6], [4, 5], [4, 6]]')

        assert str(list(lazyllm.OptionIter([o2, o3], get_options))) == expected_output
