from pypinyin import pinyin, Style
import pandas as pd
import ahocorasick

ethnic_groups_of_china = [
    "汉族", "壮族", "满族", "回族", "苗族", "维吾尔族", "土家族", "彝族", "蒙古族", "藏族", 
    "布依族", "侗族", "瑶族", "朝鲜族", "白族", "哈尼族", "哈萨克族", "黎族", "傣族", "畲族", 
    "傈僳族", "仡佬族", "东乡族", "高山族", "拉祜族", "水族", "佤族", "纳西族", "羌族", "土族", 
    "仫佬族", "锡伯族", "柯尔克孜族", "达斡尔族", "景颇族", "毛南族", "撒拉族", "布朗族", "塔吉克族", 
    "阿昌族", "普米族", "鄂温克族", "怒族", "京族", "基诺族", "德昂族", "保安族", "俄罗斯族", "裕固族", 
    "乌孜别克族", "门巴族", "鄂伦春族", "独龙族", "塔塔尔族", "赫哲族", "珞巴族"
]

def build_automaton(keywords):
    A = ahocorasick.Automaton()
    for word in keywords:
        A.add_word(word, word)
    A.make_automaton()
    return A

automaton = build_automaton(ethnic_groups_of_china)

def cut_city_name(city_name:str):
    city_name_copy = city_name[:]
    for end_idx, matched_kw in automaton.iter(city_name):
        city_name = city_name.replace(matched_kw, '')

    if city_name.endswith(("特别行政区")):
        city_name = city_name[:-5]
    elif city_name.endswith(("自治区","自治县","自治州","自治市","自治旗","直辖市","联合旗")):
        city_name = city_name[:-3]
    elif city_name.endswith(("矿区","地区")):
        city_name = city_name[:-2]
    elif (len(city_name)>2 
          and not city_name.endswith(("前旗","中旗","后旗","左旗","右旗")) 
          and city_name.endswith(("省","区","县","州","市","旗"))):
        city_name = city_name[:-1]
    return city_name if city_name else city_name_copy

def trans_city_name(city_name:str):
    pinyin_result = pinyin(city_name, style=Style.NORMAL, heteronym=False)
    pinyin_text = ''.join([item[0] for item in pinyin_result])
    return pinyin_text

if __name__ == '__main__':
    # 读取城市名数据，从 https://lbs.amap.com/api/webservice/download 下载
    date_path = "path_to_download_data/AMap_adcode_citycode.xlsx"
    df = pd.read_excel(date_path)
    result_df = pd.DataFrame(columns=['full_name','simple_name', 'pinyin', 'adcode'])
    result_df['full_name'] = df['中文名']
    result_df['simple_name'] = df['中文名'].apply(cut_city_name)
    result_df['pinyin'] = result_df['simple_name'].apply(trans_city_name)
    result_df['adcode'] = df['adcode']
    result_df.dropna()

    # 保存结果
    result_df.to_excel("./citycode.xlsx", index=False)