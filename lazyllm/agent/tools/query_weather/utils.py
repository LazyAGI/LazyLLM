import pandas as pd
try:
    import ahocorasick
except ImportError:
    raise ImportError("Please use `pip install pyahocorasick` to install ahocorasick")
import os

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

class CityCodeMatcher:
    def __init__(self, data_path:str=f"{current_dir}/data/citycode.xlsx") -> None:
        self.data = pd.read_excel(data_path)
        self.automaton = self.build_automaton(self.data)

    @staticmethod
    def build_automaton(data:pd.DataFrame):
        A = ahocorasick.Automaton()
        for idx in data.index:
            A.add_word(data.loc[idx, "simple_name"], (idx, data.loc[idx, "adcode"]))
            A.add_word(data.loc[idx, "pinyin"], (idx, data.loc[idx, "adcode"]))
        A.make_automaton()
        return A

    def get_adcode_and_fullname(self, city_name:str):
        for end_idx, (idx, adcode) in self.automaton.iter(city_name.lower()):
            return adcode, self.data.loc[idx, "full_name"]
        return None, None