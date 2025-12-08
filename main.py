import pandas as pd
import numpy as np
import re   #正则表达式模块，用于文本匹配、搜索、替换等操作。
from sklearn.model_selection import train_test_split    #用于将数据集分为训练集和测试集
from sklearn.ensemble import RandomForestRegressor  #随机森林回归模型
from sklearn.pipeline import Pipeline   #将多个处理步骤（如预处理 + 模型训练）串起来，形成一个工作流，便于管理和复用。
from sklearn.compose import ColumnTransformer   #对不同类型的列（数值列、类别列）应用不同的预处理方法。
from sklearn.preprocessing import OneHotEncoder, StandardScaler #OneHotEncoder：将类别特征转换为独热编码（0/1矩阵）。
# StandardScaler：对数值特征进行标准化（均值为0，方差为1）。
from sklearn.impute import SimpleImputer    #用于填补缺失值，例如用均值、中位数或常数填充。
from sklearn.metrics import mean_absolute_error, r2_score   #mean_absolute_error：平均绝对误差，衡量回归预测误差。
# r2_score：决定系数，衡量回归模型的拟合优度。
import joblib   #用于保存和加载模型或其他 Python 对象，例如训练好的机器学习模型。
import plotly.express as px #Plotly 的简易接口，用于绘制交互式图表（折线图、柱状图、散点图等）
from wordcloud import WordCloud #生成词云图，可用于展示文本中词频的重要性。
import matplotlib.pyplot as plt #静态绘图库，可绘制各种图形（折线图、柱状图、散点图等）
import os   #操作系统相关功能，如文件路径操作、目录遍历、环境变量访问等。
import streamlit as st  #Streamlit 框架，用于快速搭建数据可视化 Web 应用。
from streamlit_echarts import st_echarts    #在 Streamlit 中嵌入 ECharts 图表，支持交互式可视化。
from pyecharts.charts import Map    #用于绘制地图图表（省市分布、热力图等）。
from pyecharts import options as opts   #设置 ECharts 图表的配置项（标题、坐标轴、颜色、标签等）。
from pyecharts.globals import ThemeType #设置图表主题（如深色、浅色风格等）。
import json #处理 JSON 数据


st.set_page_config(layout="wide", page_title="招聘数据洞察与薪资预测")  #页面配置函数。
# page_title：设置网页标签栏显示的标题。
# layout="wide"：设置页面布局为宽屏模式，比默认的居中模式显示更多内容。

st.title("📊 招聘数据洞察平台 & 💰 智能薪资预测系统")   #实际页面中显示的标题

st.markdown("上传招聘数据即可自动可视化并构建薪资预测模型。")   #用 Markdown 显示一段说明文字

#st.session_state 是 Streamlit 用来在不同用户操作间保存状态的字典
if "use_demo" not in st.session_state: st.session_state.use_demo = False    #是否使用示例数据的标志
if "data_loaded_df" not in st.session_state: st.session_state.data_loaded_df = None #用户上传的数据
if "model" not in st.session_state: st.session_state.model = None   #训练好的薪资预测模型

winsor_switch = True    #是否对数据进行极值截断（Winsorization）
low_q, high_q = 0.01, 0.99  #截断的低分位数和高分位数
top_cities, top_inds, min_combo = 30, 12, 3 #可视化或分析时的筛选阈值
sample_3d, jitter_seed = 10, 42 #3D 可视化抽样数量和随机扰动种子

def filter_keywords(text):
    #停用词：这些词会被过滤掉，不作为关键词保留
    stop_words = {
        "工程师", "开发", "专员", "主管", "经理", "总监", "负责人", "专家", "高级", "资深",
        "初级", "中级", "助理", "实习", "兼职", "全职", "岗位", "职位", "招聘", "应聘",
        "任职", "工作", "职责", "要求", "相关", "专业", "熟练", "掌握", "具备", "经验",
        "能力", "团队", "合作", "沟通", "负责", "参与", "主导", "协助", "支持", "保障"
    }

    keywords = re.findall(r"[\u4e00-\u9fa5]{2,}", text) #提取所有长度 >=2 的中文词语（使用正则表达式）
    filtered = [kw for kw in keywords if kw not in stop_words and len(kw) >= 2] #过滤掉停用词
    return " ".join(filtered)   #把列表中的词用空格拼接成一个字符串返回。


def parse_salary(s):
    if pd.isna(s): return np.nan    #如果薪资信息缺失，返回 NaN

    s = str(s).lower().replace(" ", "").replace("，", ",")  #全部转成小写，去掉空格，中文逗号转为英文逗号

    #使用的数据只考虑直接表示和使用k做单位，不考虑可能存在的w或中文单位
    try:
        #考虑k结尾的情况
        nums = re.findall(r"(\d+\.?\d*)k", s)   #查找形如 10k 或 10.5k 的数字，提取数字部分
        if len(nums) == 1: return float(nums[0])    #只有一个数字，直接返回该值
        if len(nums) >= 2: return (float(nums[0]) + float(nums[1])) / 2 #有两个及以上数字，返回平均值

        #考虑不是k结尾的情况
        nums2 = re.findall(r"(\d+\.?\d*)", s)
        if len(nums2) == 1:
            v = float(nums2[0])
            return v / 1000 if v > 1000 else v  #转换为k单位
        if len(nums2) >= 2:
            a, b = float(nums2[0]), float(nums2[1])
            if a > 1000: a /= 1000; b /= 1000
            return (a + b) / 2
        
    except:
        return np.nan
    return np.nan


def parse_workyear(w):  #解析工作年限要求
    if pd.isna(w): return np.nan

    s = str(w)  #将输入转换为字符串

    if "应届" in s: return 0
    if "不限" in s: return np.nan

    m = re.findall(r"(\d+)\D+(\d+)", s) #年限要求为区间形式
    if m: return (float(m[0][0]) + float(m[0][1])) / 2

    m2 = re.search(r"(\d+)", s) #单个数字
    if m2:
        v = float(m2.group(1))
        return v / 2 if "以下" in s else v
    
    return np.nan


def normalize_education(ed):
    if pd.isna(ed): return 0
    s = str(ed)
    if "博" in s: return 5
    if "硕" in s or "master" in s: return 4
    if "本" in s: return 3
    if "专" in s: return 2
    if "高" in s: return 1
    return 0


COLUMN_MAPPING_BANK = {
    # 岗位名称：覆盖官方命名、口语化表达、英文变体
    "positionName": [
        "positionName", "position", "岗位", "职位", "岗位名称", "招聘岗位",
        "职位名称", "应聘岗位", "工作岗位", "岗位类型", "职位类型", "岗位名称",
        "job", "jobtitle", "job_title", "position_title","job_name", "岗位名", "职位名",
        "招聘职位", "求职岗位", "任职岗位", "岗位说明", "职位说明", "岗位称呼"
    ],
    # 工作城市：覆盖地点相关所有表述
    "city": [
        "city", "工作城市", "城市", "地点", "工作地点", "所在城市", "上班地点",
        "工作地点 ", "城市名称", "地区", "所在地区", "工作地区", "办公地点",
        "location", "work_location", "city_name", "地区名称", "省市", "所在省市",
        "工作省市", "办公城市", "常驻城市", "base城市", "base地", "驻地","area"
    ],
    # 行业领域：覆盖行业分类、方向、赛道等表述
    "industryField": [
        "industryField", "行业", "所属行业", "行业方向", "行业领域", "行业类型",
        "行业分类", "所在行业", "行业赛道", "产业", "所属产业", "产业领域",
        "industry", "industry_type", "industry_category", "business_field",
        "行业属性", "行业板块", "行业细分", "细分行业", "领域", "业务领域",'job_type'
    ],
    # 岗位大类：覆盖分类、类别、层级等表述
    "bigcategory": [
        "bigcategory", "岗位大类", "职位大类", "类别", "岗位分类", "职位分类",
        "岗位类别", "职位类别", "大类", "分类", "岗位层级", "职位层级",
        "job_category", "position_category", "job_type", "position_type",
        "岗位群", "职位群", "职业大类", "职业分类", "岗位体系", "职位体系",
        "分类名称", "类别名称", "一级分类", "二级分类", "岗位归属", "职位归属"
    ],
    # 工作年限：覆盖经验、年限、资历等表述
    "workYear": [
        "workYear", "工作年限", "经验", "经验要求", "工作经验", "从业年限",
        "任职年限", "工作时长", "经验年限", "资历要求", "工作资历", "经验水平",
        "work_experience", "experience", "years_of_experience", "work_years",
        "经验范围", "年限要求", "最低工作年限", "最高工作年限", "经验年限要求",
        "从业经验", "职场经验", "工作经历年限", "经验时长", "任职经验","job_exp","workingexp"
    ],
    # 学历要求：覆盖学历、文化程度、教育背景等表述
    "education": [
        "education", "学历", "最低学历", "学历要求", "文化程度", "教育背景",
        "学历层次", "学历水平", "教育程度", "毕业学历", "学历条件", "学历标准",
        "education_level", "education_background", "academic_background",
        "学历要求 ", "最低学历要求", "最高学历要求", "学历门槛", "学历资质",
        "文化水平", "教育程度要求", "学历背景", "学术背景", "毕业院校要求",  # 关联学历的延伸表述
        "学历档次", "学历等级", "学位要求", "学历学位","job_edu",'edu'
    ],
    # 薪资待遇：覆盖所有薪酬相关表述
    "salary": [
        "salary", "薪资", "工资", "薪水", "待遇", "薪酬", "薪资水平", "薪资范围",
        "工资水平", "工资范围", "薪水范围", "薪酬水平", "薪酬范围", "薪资待遇",
        "薪酬待遇", "工资待遇", "薪水待遇", "月薪", "月薪资", "月工资", "月薪酬",
        "salary_range", "salary_level", "pay", "wage", "remuneration", "compensation",
        "薪资标准", "工资标准", "薪酬标准", "待遇标准", "薪资结构", "薪酬结构",
        "税前薪资", "税后薪资", "税前工资", "税后工资", "月薪范围", "年薪",
        "年薪范围", "年薪水平", "时薪", "日薪", "周薪", "薪酬福利", "薪资福利",
        "待遇水平", "收入水平", "薪资区间", "工资区间", "薪酬区间","money"
    ]
}


def auto_map_columns(df, mapping_bank): #更新列名为标准列名
    rename_dict = {}
    for new, cands in mapping_bank.items():
        for col in df.columns:
            if col.strip() in cands or col.lower() in [x.lower() for x in cands]:
                rename_dict[col] = new
                break
    return df.rename(columns=rename_dict)


# 城市标准化    解决"上海/上海市/上海-浦东新区等城市名称混乱"问题
# 薪资解析      支持 10k-20k / 1万-2万 / 年薪100k / 面议 等
# 薪资截尾      修复薪资直方图与核密度图"极端飞线问题"
# 行业聚合      TopN 行业聚合，其余归类"其他"
# 输出 salary_k_raw / salary_k_clean / salary_k_log 三套数据

city_alias_map = {
        "北京": "北京", "上海": "上海", "天津": "天津", "重庆": "重庆",
        "石家庄": "石家庄", "唐山": "唐山", "秦皇岛": "秦皇岛", "邯郸": "邯郸",
        "邢台": "邢台", "保定": "保定", "张家口": "张家口", "承德": "承德",
        "沧州": "沧州", "廊坊": "廊坊", "衡水": "衡水",
        "太原": "太原", "大同": "大同", "阳泉": "阳泉", "长治": "长治",
        "晋城": "晋城", "朔州": "朔州", "晋中": "晋中", "运城": "运城",
        "忻州": "忻州", "临汾": "临汾", "吕梁": "吕梁",
        "呼和浩特": "呼和浩特", "包头": "包头", "乌海": "乌海", "赤峰": "赤峰",
        "通辽": "通辽", "鄂尔多斯": "鄂尔多斯", "呼伦贝尔": "呼伦贝尔",
        "巴彦淖尔": "巴彦淖尔", "乌兰察布": "乌兰察布", "兴安盟": "兴安盟",
        "锡林郭勒盟": "锡林郭勒盟", "阿拉善盟": "阿拉善盟",
        "沈阳": "沈阳", "大连": "大连", "鞍山": "鞍山", "抚顺": "抚顺",
        "本溪": "本溪", "丹东": "丹东", "锦州": "锦州", "营口": "营口",
        "阜新": "阜新", "辽阳": "辽阳", "盘锦": "盘锦", "铁岭": "铁岭",
        "朝阳": "朝阳", "葫芦岛": "葫芦岛",
        "长春": "长春", "吉林": "吉林", "四平": "四平", "辽源": "辽源",
        "通化": "通化", "白山": "白山", "松原": "松原", "白城": "白城",
        "延边": "延边",
        "哈尔滨": "哈尔滨", "齐齐哈尔": "齐齐哈尔", "鸡西": "鸡西",
        "鹤岗": "鹤岗", "双鸭山": "双鸭山", "大庆": "大庆", "伊春": "伊春",
        "佳木斯": "佳木斯", "七台河": "七台河", "牡丹江": "牡丹江",
        "黑河": "黑河", "绥化": "绥化", "大兴安岭": "大兴安岭",
        "南京": "南京", "无锡": "无锡", "徐州": "徐州", "常州": "常州",
        "苏州": "苏州", "南通": "南通", "连云港": "连云港", "淮安": "淮安",
        "盐城": "盐城", "扬州": "扬州", "镇江": "镇江", "泰州": "泰州",
        "宿迁": "宿迁",
        "杭州": "杭州", "宁波": "宁波", "温州": "温州", "嘉兴": "嘉兴",
        "湖州": "湖州", "绍兴": "绍兴", "金华": "金华", "衢州": "衢州",
        "舟山": "舟山", "台州": "台州", "丽水": "丽水",
        "合肥": "合肥", "芜湖": "芜湖", "蚌埠": "蚌埠", "淮南": "淮南",
        "马鞍山": "马鞍山", "淮北": "淮北", "铜陵": "铜陵", "安庆": "安庆",
        "黄山": "黄山", "滁州": "滁州", "阜阳": "阜阳", "宿州": "宿州",
        "六安": "六安", "亳州": "亳州", "池州": "池州", "宣城": "宣城",
        "福州": "福州", "厦门": "厦门", "莆田": "莆田", "三明": "三明",
        "泉州": "泉州", "漳州": "漳州", "南平": "南平", "龙岩": "龙岩",
        "宁德": "宁德",
        "南昌": "南昌", "景德镇": "景德镇", "萍乡": "萍乡", "九江": "九江",
        "新余": "新余", "鹰潭": "鹰潭", "赣州": "赣州", "吉安": "吉安",
        "宜春": "宜春", "抚州": "抚州", "上饶": "上饶",
        "济南": "济南", "青岛": "青岛", "淄博": "淄博", "枣庄": "枣庄",
        "东营": "东营", "烟台": "烟台", "潍坊": "潍坊", "济宁": "济宁",
        "泰安": "泰安", "威海": "威海", "日照": "日照", "莱芜": "莱芜",
        "临沂": "临沂", "德州": "德州", "聊城": "聊城", "滨州": "滨州",
        "菏泽": "菏泽",
        "郑州": "郑州", "开封": "开封", "洛阳": "洛阳", "平顶山": "平顶山",
        "安阳": "安阳", "鹤壁": "鹤壁", "新乡": "新乡", "焦作": "焦作",
        "濮阳": "濮阳", "许昌": "许昌", "漯河": "漯河", "三门峡": "三门峡",
        "南阳": "南阳", "商丘": "商丘", "信阳": "信阳", "周口": "周口",
        "驻马店": "驻马店", "济源": "济源",
        "武汉": "武汉", "黄石": "黄石", "十堰": "十堰", "宜昌": "宜昌",
        "襄阳": "襄阳", "鄂州": "鄂州", "荆门": "荆门", "孝感": "孝感",
        "荆州": "荆州", "黄冈": "黄冈", "咸宁": "咸宁", "随州": "随州",
        "恩施": "恩施", "天门": "天门", "潜江": "潜江",
        "仙桃": "仙桃", "神农架": "神农架",
        "长沙": "长沙", "株洲": "株洲", "湘潭": "湘潭", "衡阳": "衡阳",
        "邵阳": "邵阳", "岳阳": "岳阳", "常德": "常德", "张家界": "张家界",
        "益阳": "益阳", "郴州": "郴州", "永州": "永州", "怀化": "怀化",
        "娄底": "娄底", "湘西": "湘西",
        "广州": "广州", "深圳": "深圳", "珠海": "珠海", "汕头": "汕头",
        "佛山": "佛山", "韶关": "韶关", "湛江": "湛江", "肇庆": "肇庆",
        "江门": "江门", "茂名": "茂名", "惠州": "惠州", "梅州": "梅州",
        "汕尾": "汕尾", "河源": "河源", "阳江": "阳江", "清远": "清远",
        "东莞": "东莞", "中山": "中山", "潮州": "潮州", "揭阳": "揭阳",
        "云浮": "云浮",
        "南宁": "南宁", "柳州": "柳州", "桂林": "桂林", "梧州": "梧州",
        "北海": "北海", "防城港": "防城港", "钦州": "钦州", "贵港": "贵港",
        "玉林": "玉林", "百色": "百色", "贺州": "贺州", "河池": "河池",
        "来宾": "来宾", "崇左": "崇左",
        "海口": "海口", "三亚": "三亚", "三沙": "三沙", "儋州": "儋州",
        "五指山": "五指山", "琼海": "琼海", "文昌": "文昌", "万宁": "万宁",
        "东方": "东方", "定安": "定安", "屯昌": "屯昌", "澄迈": "澄迈",
        "临高": "临高", "白沙": "白沙", "昌江": "昌江",
        "乐东": "乐东", "陵水": "陵水", "保亭": "保亭",
        "琼中": "琼中",
        "重庆": "重庆",
        "成都": "成都", "自贡": "自贡", "攀枝花": "攀枝花", "泸州": "泸州",
        "德阳": "德阳", "绵阳": "绵阳", "广元": "广元", "遂宁": "遂宁",
        "内江": "内江", "乐山": "乐山", "南充": "南充", "眉山": "眉山",
        "宜宾": "宜宾", "广安": "广安", "达州": "达州", "雅安": "雅安",
        "巴中": "巴中", "资阳": "资阳", "阿坝": "阿坝", "甘孜": "甘孜",
        "凉山": "凉山",
        "贵阳": "贵阳", "六盘水": "六盘水", "遵义": "遵义", "安顺": "安顺",
        "毕节": "毕节", "铜仁": "铜仁", "黔西南": "黔西南",
        "黔东南": "黔东南", "黔南": "黔南",
        "昆明": "昆明", "曲靖": "曲靖", "玉溪": "玉溪", "保山": "保山",
        "昭通": "昭通", "丽江": "丽江", "普洱": "普洱", "临沧": "临沧",
        "楚雄": "楚雄", "红河": "红河", "文山": "文山",
        "西双版纳": "西双版纳", "大理": "大理", "德宏": "德宏",
        "怒江": "怒江", "迪庆": "迪庆",
        "拉萨": "拉萨", "昌都": "昌都", "山南": "山南", "日喀则": "日喀则",
        "那曲": "那曲", "阿里": "阿里", "林芝": "林芝",
        "西宁": "西宁", "海东": "海东", "海北": "海北", "黄南": "黄南",
        "海南": "海南", "果洛": "果洛", "玉树": "玉树",
        "海西": "海西",
        "银川": "银川", "石嘴山": "石嘴山", "吴忠": "吴忠", "固原": "固原",
        "中卫": "中卫",
        "乌鲁木齐": "乌鲁木齐", "克拉玛依": "克拉玛依", "吐鲁番": "吐鲁番",
        "哈密": "哈密", "昌吉": "昌吉", "博尔塔拉": "博尔塔拉",
        "巴音郭楞": "巴音郭楞", "阿克苏": "阿克苏", "克孜勒苏": "克孜勒苏",
        "喀什": "喀什", "和田": "和田", "伊犁": "伊犁",
        "塔城": "塔城", "阿勒泰": "阿勒泰", "石河子": "石河子",
        "阿拉尔": "阿拉尔", "图木舒克": "图木舒克", "五家渠": "五家渠",
        "北屯": "北屯",
        "香港": "香港", "澳门": "澳门", "台湾": "台湾"
    }


def standardize_city(city): #标准化城市名称函数
    if pd.isna(city):
        return np.nan
    
    city = str(city).strip()    #去除前后空白字符
    city = re.sub(r"\s+", "", city) #去除所有空白字符（去除字符串中间可能存在的空白字符）
    city = re.sub(r"(省|市|区|县)$", "", city)  #去除末尾的省/市/区/县等行政区划后缀

    return city_alias_map.get(city, city)   #返回标准化后的城市名称，若无匹配则返回原始名称


def apply_salary_clean(df, low_q=0.01, high_q=0.99, winsor=True):
    s = pd.to_numeric(df["salary_k_raw"], errors="coerce")  #将薪资列转换为数值类型，无法转换的值设为 NaN

    q1, q2 = s.quantile(low_q), s.quantile(high_q)  #计算指定分位数的薪资值

    if winsor:
        df["salary_k_clean"] = s.clip(lower=q1, upper=q2)   #将极值点截断到分位数范围内
    else:
        df["salary_k_clean"] = s.mask((s < q1) | (s > q2), np.nan)  #将极值点设为 NaN

    df["salary_k_log"] = df["salary_k_clean"].apply(lambda x: np.log1p(x) if pd.notna(x) else x)    #薪资取对数

    df["salary_is_outlier"] = (s < q1) | (s > q2)  #标记是否为极值点

    return df


def aggregate_industry(df, col, topN):
    df = df.copy()  #创建数据副本，避免修改原始数据

    if col not in df.columns:
        df["industry_agg"] = "其他"
        return df
    
    df[col] = df[col].fillna("其他")  #将缺失值填充为“其他”

    top = df[col].value_counts().nlargest(topN).index.tolist()  #获取出现频率最高的 topN 个类别

    df["industry_agg"] = df[col].where(df[col].isin(top), other="其他") #刷新列的每个元素的值
    
    return df


#Streamlit 的缓存装饰器，表示该函数的返回结果会被缓存。
# 作用：当数据不变时，多次执行不会重新跑一遍数据清洗流程，而是直接使用缓存，大幅提升速度。
@st.cache_data(show_spinner="⚙️ 正在处理和清洗数据...")
def process_and_engineer_data(raw_df):
    """统一的数据清洗和特征工程函数，使用缓存"""
    df = raw_df.copy()

    # 1. 字段映射
    df = auto_map_columns(df, COLUMN_MAPPING_BANK)

    # 2. 确保所有必需的列都存在，否则创建空列
    required_cols = ["positionName", "city", "industryField", "bigcategory", "workYear", "education", "salary"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # 3. 工程字段
    df["salary_k"] = df["salary"].apply(parse_salary)
    df["work_year_num"] = df["workYear"].apply(parse_workyear)
    df["edu_level"] = df["education"].apply(normalize_education)

    return df


@st.cache_data
def load_file_data(file, name):
    """根据文件类型加载原始数据"""
    if name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)


st.sidebar.header("📁 数据加载")    #在侧边栏显示标题

uploaded_file = st.sidebar.file_uploader(   #显示一个文件上传控件
    "上传招聘数据（Excel / CSV）",  #控件的 标题/提示文本
    type=["xlsx", "xls", "csv"],    #文件类型
    key="file_uploader" #为控件指定一个 唯一标识符（key）
)

data_to_process = None
is_demo_mode = False    #标记当前是否使用示例数据

if uploaded_file:
    # 文件上传，重新加载数据
    data_to_process = load_file_data(uploaded_file, uploaded_file.name)
    is_demo_mode = False
elif st.sidebar.button("使用示例数据（拉勾招聘）"):
    # 点击示例按钮，重新加载数据
    st.session_state.use_demo = True
    try:
        data_to_process = pd.read_excel("拉勾招聘.xlsx")
        is_demo_mode = True
    except FileNotFoundError:
        st.error("示例数据文件 '拉勾招聘.xlsx' 未找到！")
        st.stop()   #立即终止当前脚本的执行，并 停止向下继续运行后续代码。
elif st.session_state.use_demo or st.session_state.data_loaded_df is None:  #修改
    # Streamlit 每次交互会 Rerun 脚本，如果处于 demo 模式或数据丢失，尝试再次加载 demo 数据
    try:
        data_to_process = pd.read_excel("拉勾招聘.xlsx")
        is_demo_mode = True
    except:
        pass  # 找不到文件，等待用户操作

# 如果有新数据需要处理，则进行处理并更新 Session State
if data_to_process is not None:
    st.session_state.data_loaded_df = process_and_engineer_data(data_to_process)
    st.session_state.use_demo = is_demo_mode


# 检查数据是否可用
df = st.session_state.data_loaded_df
if df is None:
    st.info("请通过左侧边栏上传数据或使用示例数据。")
    st.stop()
else:
    st.sidebar.success(f"数据加载成功，共 {len(df)} 条记录")


# 列名容错
df.columns = [c.strip() for c in df.columns]


# 找到地点列并标准化
city_col = next((c for c in df.columns if "市" in c or "city" in c.lower() or "地点" in c), None)
df["city_standard"] = df[city_col].apply(standardize_city) if city_col else np.nan

# 薪资
salary_col = next((c for c in df.columns if "薪" in c or "salary" in c.lower()), None)
df["salary_k_raw"] = df[salary_col].apply(parse_salary) if salary_col else np.nan

# 截尾
df = apply_salary_clean(df, low_q, high_q, True)

# 行业聚合
ind_col = next((c for c in df.columns if "行业" in c or "industry" in c.lower()), None)
df = aggregate_industry(df, ind_col, top_inds)


# 全局薪资模式选择器
salary_mode = st.radio( #Streamlit 的单选按钮控件，让用户从一组选项中选择一个
    "选择薪资显示模式",
    ["原始 (raw)", "截尾 (clean)", "log(截尾)"],
    horizontal=True,    #按钮水平排列
    index=1 #默认选中第二个选项
)

if salary_mode == "原始 (raw)":
    sal_col = "salary_k_raw"
    sal_name = "薪资(k) - 原始"
elif salary_mode == "截尾 (clean)":
    sal_col = "salary_k_clean"
    sal_name = "薪资(k) - 截尾"
else:
    sal_col = "salary_k_log"
    sal_name = "薪资 log(1+x)"

tab1, tab2, tab3 = st.tabs(["📌 数据概览", "📊 可视化分析", "💰 在线薪资预测"]) #创建多标签页界面


with tab1:
    st.subheader("数据样例")    #二级标题（subheader）
    st.dataframe(df.head(20), width='stretch')  #显示数据框的前20行，宽度自适应
    st.metric("记录数量", len(df))  #展示一个指标
    st.metric("有薪资记录数量", df[sal_col].notna().sum())  #sal_col 列非空记录数

with tab2:
    st.markdown("### 📊 全方位招聘数据可视化分析")
    st.markdown("---")
    
    # 薪资分布分析（使用全局薪资模式）
    st.header("📌 薪资分布分析")
    
    df_sal = df.dropna(subset=[sal_col])    #从原始数据 df 中删除在薪资列 sal_col 中为空的行，只保留有薪资的记录。

    if df_sal.empty:
        st.warning("当前选择无薪资数据可展示")
    else:
        col_sal1, col_sal2 = st.columns(2)  #创建两个并排的列布局
        
        # 1. 薪资分布直方图+箱线图
        with col_sal1:
            fig_hist = px.histogram(
                df_sal, #数据源
                x=sal_col,  #以薪资列为横轴
                nbins=40,   #将薪资数据分成 40 个柱子
                marginal="box", #在直方图上方显示对应的箱线图
                title=f"薪资分布直方图 + 箱线图 ({salary_mode})",
                labels={sal_col: sal_name}
            )
            st.plotly_chart(fig_hist, use_container_width=True) #显示图表，自适应宽度

        # 2. 薪资小提琴图
        with col_sal2:
            fig_violin = px.violin(
                df_sal,
                y=sal_col,
                box=True,   #在小提琴图中显示箱线图
                points="outliers",  #只显示异常值点
                title=f"薪资小提琴图 ({salary_mode})",
                labels={sal_col: sal_name}
            )
            st.plotly_chart(fig_violin, use_container_width=True)
    
    st.markdown("---")


    # 第二组：城市&行业分析
    st.subheader("🌆 城市&行业竞争力分析")

    col_city1, col_city2 = st.columns(2)

    # 4. 城市平均薪资横向条形图
    with col_city1:
        required_cols = ["city", sal_col]

        if all(col in df.columns for col in required_cols) and all(df[col].notna().any() for col in required_cols):
            city_salary = df.groupby("city")[sal_col].agg(
                mean_salary="mean", #创建列 mean_salary，计算每个城市的平均薪资
                count="count"   #创建列 count，计算每个城市的样本数量
            ).dropna().query("count >= 5").sort_values("mean_salary", ascending=True)

            if not city_salary.empty:
                fig_city = px.bar(
                    city_salary.tail(15),
                    y=city_salary.tail(15).index,
                    x="mean_salary",
                    orientation="h",    #横向条形图
                    title=f"城市平均薪资 Top15（样本量≥5） ({salary_mode})",
                    labels={"mean_salary": f"平均薪资 ({sal_name})", "y": "城市"},
                    color="count",  #按样本量着色
                    color_continuous_scale="Viridis",   #颜色渐变表
                    hover_data=["count"]    #悬停显示样本量
                )

                st.plotly_chart(fig_city, config={"displayModeBar": True, "responsive": True})
                #显示 mode bar（工具条）、告知 Plotly 图在容器大小改变（浏览器窗口调整）时应自适应大小
            else:
                st.info("📌 城市薪资数据不足（需每个城市≥5条样本），无法绘制")
        else:
            missing = [col for col in required_cols if col not in df.columns or not df[col].notna().any()]
            st.info(f"📌 缺少 {missing} 数据，无法绘制城市薪资图")

    # 5. 行业平均薪资柱状图
    with col_city2:
        required_cols = ["industryField", sal_col]

        if all(col in df.columns for col in required_cols) and all(df[col].notna().any() for col in required_cols):
            ind_mean = df.groupby("industryField")[sal_col].agg(
                mean_salary="mean",
                count="count"
            ).dropna().query("count >= 5").sort_values("mean_salary", ascending=False).head(15)

            if not ind_mean.empty:
                fig_ind = px.bar(
                    ind_mean,
                    x=ind_mean.index,
                    y="mean_salary",
                    title=f"行业平均薪资 Top15（样本量≥5） ({salary_mode})",
                    labels={"mean_salary": f"平均薪资 ({sal_name})", "x": "行业领域"},
                    color="count",
                    color_continuous_scale="Blues",
                    hover_data=["count"]
                )
                st.plotly_chart(fig_ind, config={"displayModeBar": True, "responsive": True})
            else:
                st.info("📌 行业薪资数据不足（需每个行业≥5条样本），无法绘制")
        else:
            missing = [col for col in required_cols if col not in df.columns or not df[col].notna().any()]
            st.info(f"📌 缺少 {missing} 数据，无法绘制行业薪资图")

    st.header("📌 城市 × 行业 薪资气泡图（二维）")

    df_bubble = (
        df.dropna(subset=["city_standard", "industry_agg", sal_col])    #删除任意一列为空的行
        .groupby(["city_standard", "industry_agg"], as_index=False)
        .agg(
            mean_salary=(sal_col, "mean"),
            job_count=(sal_col, "count")
        )
    )

    # 过滤样本数太少的
    df_bubble = df_bubble[df_bubble["job_count"] >= int(min_combo)]

    # 保留 Top 城市
    city_toplist = df["city_standard"].value_counts().nlargest(top_cities).index.tolist()
    df_bubble = df_bubble[df_bubble["city_standard"].isin(city_toplist)]

    if df_bubble.empty:
        st.info("数据量不足，无法绘制气泡图。请降低阈值或增加展示城市数。")
    else:
        # 城市映射为整数并 jitter（抖动）
        np.random.seed(int(jitter_seed))

        ordered_city = sorted(df_bubble["city_standard"].unique())
        city_to_num = {c: i for i, c in enumerate(ordered_city)}    
        #enumerate 把 ordered_city 中的每个城市 c 与一个整数索引 i（从 0 开始）配对

        df_bubble["city_num"] = df_bubble["city_standard"].map(city_to_num) #将城市名称映射为对应的整数索引，写入新列 city_num

        df_bubble["city_jitter"] = df_bubble["city_num"] + np.random.normal(0, 0.18, len(df_bubble))    #在 city_num 的基础上加上一个服从正态分布的随机偏移（“抖动”），生成浮点数列 city_jitter。

        # 气泡大小 log 缩放
        df_bubble["bubble_size"] = np.log1p(df_bubble["job_count"]) #避免过大的尺寸

        mn, mx = df_bubble["bubble_size"].min(), df_bubble["bubble_size"].max() #获取气泡大小的最小值和最大值

        df_bubble["bubble_scaled"] = 6 + (df_bubble["bubble_size"] - mn) / (mx - mn + 1e-9) * 34  #缩放气泡大小到 6-40 之间,1e-9 防止除以零

        fig_bubble = px.scatter(
            df_bubble,
            x="city_jitter",
            y="mean_salary",
            color="industry_agg",
            size="bubble_scaled",
            hover_data=["city_standard", "industry_agg", "mean_salary", "job_count"],
            title=f"城市 × 行业 气泡图（Top {top_cities} 城市 + Top {top_inds} 行业） ({salary_mode})",
            labels={"mean_salary": f"平均薪资 ({sal_name})"}
        )

        fig_bubble.update_layout(   #将x轴刻度替换为城市名称
            xaxis=dict( #字典
                tickmode="array",
                tickvals=list(range(len(ordered_city))),    #设置刻度值和对应的标签
                ticktext=ordered_city,
                tickangle=-45,  #逆时针旋转45度，避免标签重叠
                title="城市"
            ),
            yaxis=dict(title=f"平均薪资 ({sal_name})")
        )
        st.plotly_chart(fig_bubble, use_container_width=True)


    # 第三组：经验&学历分析
    st.subheader("🎓 经验&学历影响分析")
    col_exp1, col_exp2 = st.columns(2)

    # 7. 工作经验-薪资趋势图
    with col_exp1:
        required_cols = ["work_year_num", sal_col]

        if all(col in df.columns for col in required_cols) and all(df[col].notna().any() for col in required_cols):
            exp_data = df.dropna(subset=required_cols)

            exp_data = exp_data[exp_data["work_year_num"] <= 20]  # 过滤异常值

            exp_grouped = exp_data.groupby("work_year_num").agg({
                sal_col: ["mean", "count"]
            }).reset_index()

            exp_grouped.columns = ["work_year_num", "mean_salary", "count"]

            exp_grouped = exp_grouped[exp_grouped["count"] >= 3]

            if not exp_grouped.empty:
                fig_exp = px.scatter(
                    exp_grouped,
                    x="work_year_num",
                    y="mean_salary",
                    size="count",
                    title=f"工作经验与薪资增长趋势 ({salary_mode})",
                    labels={"work_year_num": "工作经验(年)", "mean_salary": f"平均薪资 ({sal_name})"},
                    trendline="lowess", #添加一条 LOWESS 平滑趋势线
                    trendline_options=dict(frac=0.3),   #调节平滑程度（越大越平滑）
                    hover_data=["count"]
                )
                st.plotly_chart(fig_exp, config={"displayModeBar": True, "responsive": True})
            else:
                st.info("📌 工作经验-薪资数据不足（需每组≥3条样本），无法绘制")
        else:
            missing = [col for col in required_cols if col not in df.columns or not df[col].notna().any()]
            st.info(f"📌 缺少 {missing} 数据，无法绘制经验-薪资趋势图")

    # 8. 学历-薪资箱线图
    with col_exp2:
        required_cols = ["edu_level", sal_col]

        if all(col in df.columns for col in required_cols) and all(df[col].notna().any() for col in required_cols):
            edu_salary = df.dropna(subset=required_cols)

            edu_mapping = {0: "未知", 1: "高中/中专", 2: "大专", 3: "本科", 4: "硕士", 5: "博士"}

            edu_salary["edu_name"] = edu_salary["edu_level"].map(edu_mapping)

            edu_count = edu_salary["edu_name"].value_counts()

            valid_edu = edu_count[edu_count >= 3].index.tolist()

            edu_salary = edu_salary[edu_salary["edu_name"].isin(valid_edu)]

            if not edu_salary.empty:
                fig_edu_box = px.box(
                    edu_salary,
                    x="edu_name",
                    y=sal_col,
                    title=f"不同学历的薪资分布差异 ({salary_mode})",
                    labels={sal_col: sal_name, "edu_name": "学历"},
                    color="edu_name",
                    color_discrete_sequence=px.colors.qualitative.D3    #使用 D3 颜色序列
                )
                st.plotly_chart(fig_edu_box, config={"displayModeBar": True, "responsive": True})
            else:
                st.info("📌 学历-薪资数据不足（需每个学历≥3条样本），无法绘制")
        else:
            missing = [col for col in required_cols if col not in df.columns or not df[col].notna().any()]
            st.info(f"📌 缺少 {missing} 数据，无法绘制学历-薪资箱线图")

    # 9. 学历×经验薪资热力图
    required_cols = ["edu_level", "work_year_num", sal_col]

    if all(col in df.columns for col in required_cols) and all(df[col].notna().any() for col in required_cols):
        pivot_data = df.dropna(subset=required_cols)

        pivot_exp_edu = pivot_data.pivot_table( #创建数据透视表
            index="edu_level",
            columns="work_year_num",
            values=sal_col,
            aggfunc="mean"
        )

        # 行数列数均要大于等于2才能绘制热力图
        if pivot_exp_edu.shape[0] >= 2 and pivot_exp_edu.shape[1] >= 2:
            edu_mapping = {0: "未知", 1: "高中/中专", 2: "大专", 3: "本科", 4: "硕士", 5: "博士"}

            y_labels = [edu_mapping.get(idx, f"等级{idx}") for idx in pivot_exp_edu.index]

            fig_heat = px.imshow(   #绘制矩形热力图
                pivot_exp_edu,
                labels=dict(x="工作经验（年）", y="学历等级", color=f"平均薪资 ({sal_name})"),
                title=f"学历 × 经验 — 薪资热力图 ({salary_mode})",
                x=pivot_exp_edu.columns.astype(str),
                y=y_labels,
                color_continuous_scale="YlOrRd" #颜色渐变表，黄橙红
            )

            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("📌 学历/经验薪资数据不足（需至少2×2维度），无法绘制热力图")
    else:
        missing = [col for col in required_cols if col not in df.columns or not df[col].notna().any()]
        st.info(f"📌 缺少 {missing} 数据，无法绘制学历×经验薪资热力图")

    st.markdown("---")


    # 第四组：岗位分析
    st.subheader("💼 岗位需求分析")
    col_job1, col_job2 = st.columns(2)

    # 10. 岗位数量Top10表格
    with col_job1:
        if "positionName" in df.columns and df["positionName"].notna().any():

            position_count = df["positionName"].value_counts().head(10) #数量前十的岗位

            if not position_count.empty:
                st.subheader("岗位数量 Top10")

                st.table(position_count.reset_index().rename(columns={"index": "岗位名称", "positionName": "招聘数量"}))    #表格
            else:
                st.info("📌 岗位数据为空，无法显示岗位数量排名")
        else:
            st.info("📌 缺少岗位名称（positionName）数据，无法显示岗位数量排名")

    # 11. 热门岗位条形图
    with col_job2:
        if "positionName" in df.columns and df["positionName"].notna().any():
            position_count = df["positionName"].value_counts().head(10).reset_index()
            position_count.columns = ["岗位名称", "招聘数量"]

            if not position_count.empty:
                fig_job_bar = px.bar(
                    position_count,
                    x="招聘数量",
                    y="岗位名称",
                    orientation="h",    #水平条形图
                    title="热门岗位招聘需求 Top10",
                    color="招聘数量",
                    color_continuous_scale="Viridis"    #颜色方案
                )

                st.plotly_chart(fig_job_bar, config={"displayModeBar": True, "responsive": True})
            else:
                st.info("📌 岗位数据为空，无法绘制热门岗位条形图")
        else:
            st.info("📌 缺少岗位名称（positionName）数据，无法绘制热门岗位条形图")

    # 12. 岗位词云
    if "positionName" in df.columns and df["positionName"].notna().any():
        st.subheader("岗位关键词词云")
        raw_text = " ".join(df["positionName"].dropna().astype(str))    #把所有岗位名称连接成一个长字符串，用空格分隔

        #似乎可以改一下
        text = filter_keywords(raw_text) if 'filter_keywords' in locals() else raw_text #去除停用词

        if text:
            font_path = "msyh.ttc" if os.name == "nt" else None #如果系统是 Windows（os.name=="nt"），使用微软雅黑字体 msyh.ttc。

            wc = WordCloud(
                width=800, height=400,
                background_color="white",
                font_path=font_path,
                max_words=50,
                min_font_size=8,
                max_font_size=48,
                collocations=False, #禁用词组合并
                random_state=42
            )

            fig_wc = plt.figure(figsize=(10, 5))    
            wc_image = wc.generate(text)    #生成词云图像
            plt.imshow(wc_image, interpolation="bilinear")  #显示词云图，平滑图像显示
            plt.axis("off") #关闭坐标轴显示

            st.pyplot(fig_wc)   #显示 Matplotlib 图像
            plt.close(fig_wc)   #关闭这个 Matplotlib 图像对象，释放内存资源。
        else:
            st.warning("📌 暂无有效岗位关键词可生成词云")
    else:
        st.info("📌 缺少岗位名称（positionName）数据，无法生成词云")

    st.set_page_config(layout="wide")   #置当前 Streamlit 页面（Web界面）的整体配置，页面采用宽屏布局


    st.subheader("🔍 深度进阶分析")

    # 14. 行业-岗位薪资热力图
    required_cols = ["industryField", "bigcategory", "salary_k"]

    if all(col in df.columns for col in required_cols) and all(df[col].notna().any() for col in required_cols):
        top_industries = df["industryField"].value_counts().head(10).index
        top_categories = df["bigcategory"].value_counts().head(10).index

        heat_data = df[
            df["industryField"].isin(top_industries) &
            df["bigcategory"].isin(top_categories)
            ].groupby(["industryField", "bigcategory"])["salary_k"].mean().unstack()    #转为透视表格式，默认第一个分组键作为行索引，第二个分组键作为列索引，值为平均薪资

        if not heat_data.empty and heat_data.shape[0] >= 3 and heat_data.shape[1] >= 3:
            fig_heat_ind = px.imshow(
                heat_data,
                title="行业×岗位类别 平均薪资热力图",
                labels={"x": "岗位大类", "y": "行业领域", "color": "平均薪资(k/月)"},
                color_continuous_scale="YlOrRd"
            )

            st.plotly_chart(fig_heat_ind, use_container_width=True)
        else:
            st.info("📌 行业-岗位数据不足（需至少3×3维度），无法绘制热力图")
    else:
        missing = [col for col in required_cols if col not in df.columns or not df[col].notna().any()]
        st.info(f"📌 缺少 {missing} 数据，无法绘制行业-岗位薪资热力图")

    # 15. 三维散点图（城市-行业-薪资）
    st.header("📌 城市 × 行业 × 薪资 三维分布（3D）")

    city3 = df["city_standard"].value_counts().nlargest(sample_3d).index.tolist()
    ind3 = df["industry_agg"].value_counts().nlargest(sample_3d).index.tolist()

    df_3d = df.dropna(subset=["city_standard", "industry_agg", "salary_k_clean"])
    df_3d = df_3d[df_3d["city_standard"].isin(city3) & df_3d["industry_agg"].isin(ind3)]

    if df_3d.empty:
        st.info("数据不足，无法绘制 3D 图")
    else:
        # label encode
        c_map = {c: i for i, c in enumerate(city3)}
        i_map = {i: j for j, i in enumerate(ind3)}

        df_3d["city_id"] = df_3d["city_standard"].map(c_map)
        df_3d["industry_id"] = df_3d["industry_agg"].map(i_map)

        df3d_plot = df_3d.groupby(
            ["city_standard", "industry_agg", "city_id", "industry_id"], as_index=False 
            #默认 groupby 会把分组键变成结果 DataFrame 的索引（index）；设置为False 则把这些分组键保留为普通列
        ).agg(
            mean_salary=("salary_k_clean", "mean"),
            job_count=("salary_k_clean", "count")
        )

        df3d_plot["size3d"] = np.log1p(df3d_plot["job_count"])
        smin, smax = df3d_plot["size3d"].min(), df3d_plot["size3d"].max()
        df3d_plot["size_scaled"] = 5 + (df3d_plot["size3d"] - smin) / (smax - smin + 1e-9) * 20

        fig3d = px.scatter_3d(
            df3d_plot,
            x="city_id",
            y="industry_id",
            z="mean_salary",
            size="size_scaled",
            color="mean_salary",
            hover_data=["city_standard", "industry_agg", "mean_salary", "job_count"],
            title=f"城市 × 行业 × 薪资 3D 分布（Top {sample_3d} 城市 × Top {sample_3d} 行业）",
        )

        fig3d.update_layout(
            scene=dict(     #Plotly 的 3D 图使用 scene 来配置 3D 坐标轴
                xaxis=dict(
                    tickmode="array",   #自定义数组
                    tickvals=list(range(len(city3))),   #位置（刻度）
                    ticktext=city3, #标签文本
                    title="城市"
                ),
                yaxis=dict(
                    tickmode="array",
                    tickvals=list(range(len(ind3))),
                    ticktext=ind3,
                    title="行业"
                ),
                zaxis=dict(title="平均薪资 (k)")
            )
        )

        st.plotly_chart(fig3d, use_container_width=True)

    # 16. 薪资正态性检验QQ图（Quantile-Quantile Plot，分位数-分位数图）
    if "salary_k" in df.columns and df["salary_k"].notna().any():
        sal_data = df["salary_k"].dropna()

        if len(sal_data) > 100:
            from scipy import stats

            sal_data_zscore = stats.zscore(sal_data)

            # qq_data = stats.probplot(sal_data, dist="norm") #与正态分布进行比较
            qq_data = stats.probplot(sal_data_zscore, dist="norm") #与标准正态分布进行比较

            qq_df = pd.DataFrame({
                "理论分位数": qq_data[0][0],
                "实际分位数": qq_data[0][1]
            })

            fig_qq = px.scatter(
                qq_df,
                x="理论分位数",
                y="实际分位数",
                title="薪资分布正态性检验QQ图（越贴近红线越符合正态分布）"
            )

            slope, intercept, r = qq_data[1]

            fig_qq.add_shape(
                type="line",
                x0=qq_df["理论分位数"].min(),
                # y0=qq_df["理论分位数"].min(),
                y0=slope * qq_df["理论分位数"].min() + intercept,
                x1=qq_df["理论分位数"].max(),
                # y1=qq_df["理论分位数"].max(),
                y1=slope * qq_df["理论分位数"].max() + intercept,
                line=dict(color="red", dash="dash")
            )

            st.plotly_chart(fig_qq, use_container_width=True)
        else:
            st.info("📌 样本量不足（需至少100条薪资数据），无法进行正态性检验")
    else:
        st.info("📌 缺少薪资数据（salary_k），无法进行正态性检验")


with tab3:
    model_path = "salary_predictor_rf.joblib"

    # 核心模型加载 (使用 st.cache_resource)
    @st.cache_resource(show_spinner="载入本地模型中...")
    def load_predictor_model(path):
        """从磁盘加载模型，并进行资源缓存"""
        try:
            model = joblib.load(path)
            return model
        except FileNotFoundError:
            return None
        except Exception as e:
            st.error(f"加载模型时发生错误: {e}")
            return None

    if st.session_state.model is None:
        cached_model = load_predictor_model(model_path)
        if cached_model is not None:
            st.session_state.model = cached_model
            st.subheader("模型状态")
            st.success("✅ 已高效载入本地缓存模型")
        elif st.session_state.data_loaded_df is not None:
            st.subheader("模型状态")  # 只有在有数据时，才显示状态
            st.warning("⚠️ 尚未加载或训练模型，请勾选 **训练新模型** 选项进行训练。")


    st.markdown("---")


    st.subheader("模型训练与预测")

    # 模型训练逻辑
    if st.checkbox("训练新模型（推荐）", key='train_model_checkbox'):   #复选框
        with st.spinner('⏳ 正在训练模型，请稍候...'):

            # 1. 准备数据 (使用 df，它指向 session state 的数据)
            train_df = df.dropna(
                subset=["salary_k", "city", "industryField", "bigcategory", "positionName", "work_year_num","edu_level"])

            if len(train_df) < 10:
                st.error("❌ 有效训练数据不足，无法训练模型。")
            else:
                if len(train_df) < 100:
                    st.warning(f"⚠️ 有效训练数据不足（仅 {len(train_df)} 条），模型准确度可能较低。")

                X = train_df[["city", "industryField", "bigcategory", "positionName", "work_year_num", "edu_level"]]
                y = train_df["salary_k"]

                pipe = Pipeline([
                    ("pre", ColumnTransformer([

                        ("num", Pipeline([          #变换步骤的名字：num
                            ("imp", SimpleImputer(strategy="median")),  #用中位数填补缺失值
                            ("scaler", StandardScaler())    #对数据进行标准化：变为均值 0、方差 1
                            ]),
                         ["work_year_num", "edu_level"] #应用该 pipeline 的列名列表
                        ),
                         
                        ("cat", Pipeline([
                            ("imp", SimpleImputer(strategy="constant", fill_value="未知")), #填补策略使用常量，缺失值替换为 "未知" 字符串
                            ("ohe", OneHotEncoder(handle_unknown="ignore")) #对这些字段做 One-Hot 编码，忽略训练集中未见过的类别（在预测时）
                            ]),
                         ["city", "industryField", "bigcategory", "positionName"]
                        ),

                    ])),

                    ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
                    # 随机森林回归模型，使用 100 棵树，随机种子 42，利用所有可用 CPU 核心进行训练

                ])

                # 2. 训练与评估
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=max(0.2, 10 / len(X)), random_state=42)   #测试集比例，取 0.2 和 10 / len(X) 中较大的

                pipe.fit(Xtr, ytr)

                pred = pipe.predict(Xte)

                # 3. 存储和缓存管理
                st.session_state.model = pipe  # 将训练好的模型存入 session_state
                
                joblib.dump(pipe, model_path)   # 保存模型到磁盘

                load_predictor_model.clear()  # 清除 st.cache_resource 缓存
                #保障下次加载时能读取最新模型

                st.success(     #显示成功提示框
                    f"✅ 模型训练与保存完成。**MAE**={mean_absolute_error(yte, pred):.3f} k/月，**R²**={r2_score(yte, pred):.3f}")


    # 模型状态提示 (只在没有模型且未勾选训练时显示)
    elif st.session_state.model is None and st.session_state.data_loaded_df is not None:
        st.subheader("模型状态")
        st.warning("⚠️ 尚未加载或训练模型，请勾选 **训练新模型** 选项进行训练。")


    # 在线单条薪资预测
    st.markdown("---")
    st.subheader("在线单条薪资预测")

    #Streamlit 中的 表单 (Form) 机制，用于把多个输入控件组合在一起，并且只在点击提交按钮时一次性运行
    with st.form("pred"):   #创建一个 ID 为 "pred" 的表单
        c1, c2 = st.columns(2)

        # 选项必须从 df 中获取 (df 指向 st.session_state.data_loaded_df)
        cities = sorted(df["city"].dropna().unique().tolist())
        industries = sorted(df["industryField"].dropna().unique().tolist())
        big_categories = sorted(df["bigcategory"].dropna().unique().tolist())
        educations = sorted(df["education"].dropna().unique().tolist())

        default_city = '北京' if '北京' in cities else (cities[0] if cities else '')
        default_ind = '互联网' if '互联网' in industries else (industries[0] if industries else '')
        default_big = '技术' if '技术' in big_categories else (big_categories[0] if big_categories else '')
        default_edu = '本科' if '本科' in educations else (educations[0] if educations else '')

        with c1:
            city_in = st.selectbox("工作城市", cities, index=cities.index(default_city) if default_city else 0) #下拉选择框组件
            ind_in = st.selectbox("行业领域", industries, index=industries.index(default_ind) if default_ind else 0)
            job_in = st.text_input("岗位名称", value="Python开发工程师")

        with c2:
            big_in = st.selectbox("岗位大类", big_categories,
                                  index=big_categories.index(default_big) if default_big else 0)
            work_in = st.text_input("工作年限", value="3-5年")
            edu_in = st.selectbox("学历", educations, index=educations.index(default_edu) if default_edu else 0)

        submit = st.form_submit_button("预测薪资")

        if submit:
            if st.session_state.model is None:
                st.error("❌ 模型未加载或未训练，无法预测薪资。")
            else:
                # 预测逻辑
                sample_data = {
                    "city": city_in,
                    "industryField": ind_in,
                    "bigcategory": big_in,
                    "positionName": job_in,
                    "work_year_num": parse_workyear(work_in),
                    "edu_level": normalize_education(edu_in)
                }

                if pd.isna(sample_data["work_year_num"]) or sample_data["edu_level"] == 0:
                    st.warning("⚠️ 工作年限或学历解析失败，请检查输入格式。")

                sample = pd.DataFrame([sample_data])

                p = st.session_state.model.predict(sample)[0]   #返回结果为列表，取第一个元素

                st.balloons()   #放出彩色气球动画
                st.success(f"🎉 预测结果：**💰 {p:.2f} k/月** ≈ **{p * 1000:.0f} 元/月**")


st.markdown("---")
st.markdown("")
