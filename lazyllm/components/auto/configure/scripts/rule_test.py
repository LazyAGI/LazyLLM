import csv
from core import Rule, SearchMode, Configurations
import unittest
from io import StringIO

class TestRule(unittest.TestCase):

    def test_parse_and_query(self):
        rules = [
            Rule.from_indexed("A", [0, 2, 4, 6, 8, 10], matches=SearchMode.BINARY_EXACTLY),
            Rule.from_indexed("B", [0, 2, 4, 6, 8, 10], matches=SearchMode.BINARY_CEIL),
            Rule.from_indexed("C", [0, 2, 4, 6, 8, 10], matches=SearchMode.BINARY_FLOOR),
            Rule.from_indexed("D", ["0", "1", "2", "3"]),
            Rule.from_indexed("E", [False, True]),
            Rule.from_options("F", ["A", "B", "C", "D", "E", "F"]),
            Rule.from_type("G", int),
            Rule.from_type("H", str),
        ]

        content = """A,B,C,D,E,F,G,H
0,0,0,0,FALSE,A,0,"Hello, world!"
0,0,0,0,FALSE,A,1,"Hi"
2,8,4,2,TRUE,A,2,"Good job!"
2,8,4,2,TRUE,A,3,"Well done!"
4,8,4,2,FALSE,C,4,"Excellent!"
"""
        query = {"A": 0, "B": 0, "C": 1, "D": "0", "E": False}
        expect = [
            {"A": 0, "B": 0, "C": 0, "D": "0", "E": False, "F": "A", "G": 0, "H": 'Hello, world!'},
            {"A": 0, "B": 0, "C": 0, "D": "0", "E": False, "F": "A", "G": 1, "H": 'Hi'},
        ]

        c = Configurations(rules)
        with self.assertRaises(ValueError):
            c.parse_header([])
        with self.assertRaises(AssertionError):
            c.parse_values(iter([]))

        reader = csv.reader(StringIO(content))
        c = Configurations(rules)
        c.parse_header(next(reader))
        c.parse_values(reader)
        self.assertEqual(c.lookup(query), expect)

if __name__ == '__main__':
    unittest.main()
