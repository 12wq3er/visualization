import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import streamlit as st
from streamlit_echarts import st_echarts
from pyecharts.charts import Map
from pyecharts import options as opts
from pyecharts.globals import ThemeType
import json

st.set_page_config(page_title="招聘数据可视化系统", layout="wide")

# ==========================================================
# ================ 1. 原始变量与旧功能保留区 ================
# ==========================================================
st.set_page_config(layout="wide", page_title="招聘数据洞察与薪资预测")
st.title("📊 招聘数据洞察平台 & 💰 智能薪资预测系统")
st.markdown("上传招聘数据即可自动可视化并构建薪资预测模型。")
if "use_demo" not in st.session_state: st.session_state.use_demo = False
if "data_loaded_df" not in st.session_state: st.session_state.data_loaded_df = None
if "model" not in st.session_state: st.session_state.model = None
winsor_switch = True
low_q, high_q = 0.01, 0.99
top_cities, top_inds, min_combo = 30, 12, 3
sample_3d, jitter_seed = 10, 42

def filter_keywords(text):
    stop_words = {
        "工程师", "开发", "专员", "主管", "经理", "总监", "负责人", "专家", "高级", "资深",
        "初级", "中级", "助理", "实习", "兼职", "全职", "岗位", "职位", "招聘", "应聘",
        "任职", "工作", "职责", "要求", "相关", "专业", "熟练", "掌握", "具备", "经验",
        "能力", "团队", "合作", "沟通", "负责", "参与", "主导", "协助", "支持", "保障"
    }
    keywords = re.findall(r"[\u4e00-\u9fa5]{2,}", text)
    filtered = [kw for kw in keywords if kw not in stop_words and len(kw) >= 2]
    return " ".join(filtered)


def parse_salary(s):
    # ... (函数体不变)
    if pd.isna(s): return np.nan
    s = str(s).lower().replace(" ", "").replace("，", ",")
    try:
        nums = re.findall(r"(\d+\.?\d*)k", s)
        if len(nums) == 1: return float(nums[0])
        if len(nums) >= 2: return (float(nums[0]) + float(nums[1])) / 2
        nums2 = re.findall(r"(\d+\.?\d*)", s)
        if len(nums2) == 1:
            v = float(nums2[0])
            return v / 1000 if v > 1000 else v
        if len(nums2) >= 2:
            a, b = float(nums2[0]), float(nums2[1])
            if a > 1000: a /= 1000; b /= 1000
            return (a + b) / 2
    except:
        return np.nan
    return np.nan


def parse_workyear(w):
    # ... (函数体不变)
    if pd.isna(w): return np.nan
    s = str(w)
    if "应届" in s: return 0
    if "不限" in s: return np.nan
    m = re.findall(r"(\d+)\D+(\d+)", s)
    if m: return (float(m[0][0]) + float(m[0][1])) / 2
    m2 = re.search(r"(\d+)", s)
    if m2:
        v = float(m2.group(1))
        return v / 2 if "以下" in s else v
    return np.nan


def normalize_education(ed):
    # ... (函数体不变)
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


def auto_map_columns(df, mapping_bank):
    # ... (函数体不变)
    rename_dict = {}
    for new, cands in mapping_bank.items():
        for col in df.columns:
            if col.strip() in cands or col.lower() in [x.lower() for x in cands]:
                rename_dict[col] = new
                break
    return df.rename(columns=rename_dict)
# ==========================================================
# ================ 2. 增强：数据清洗模块 ====================
# ==========================================================
# 🔹 城市标准化    解决"上海/上海市/上海-浦东新区等城市名称混乱"问题
# 🔹 薪资解析      支持 10k-20k / 1万-2万 / 年薪100k / 面议 等
# 🔹 薪资截尾      修复薪资直方图与核密度图"极端飞线问题"
# 🔹 行业聚合      TopN 行业聚合，其余归类"其他"
# 🔹 输出 salary_k_raw / salary_k_clean / salary_k_log 三套数据

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


def standardize_city(city):
    if pd.isna(city):
        return np.nan
    city = str(city).strip()
    city = re.sub(r"\s+", "", city)
    city = re.sub(r"(省|市|区|县)$", "", city)
    return city_alias_map.get(city, city)


def apply_salary_clean(df, low_q=0.01, high_q=0.99, winsor=True):
    s = pd.to_numeric(df["salary_k_raw"], errors="coerce")
    q1, q2 = s.quantile(low_q), s.quantile(high_q)
    if winsor:
        df["salary_k_clean"] = s.clip(lower=q1, upper=q2)
    else:
        df["salary_k_clean"] = s.mask((s < q1) | (s > q2), np.nan)
    df["salary_k_log"] = df["salary_k_clean"].apply(lambda x: np.log1p(x) if pd.notna(x) else x)
    df["salary_is_outlier"] = (s < q1) | (s > q2)
    return df

def aggregate_industry(df, col, topN):
    df = df.copy()
    if col not in df.columns:
        df["industry_agg"] = "其他"
        return df
    df[col] = df[col].fillna("其他")
    top = df[col].value_counts().nlargest(topN).index.tolist()
    df["industry_agg"] = df[col].where(df[col].isin(top), other="其他")
    return df
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

# ==========================================================
# ================ 3. 数据侧边栏参数 & 加载区 ================
# ==========================================================
st.sidebar.header("📁 数据加载")
uploaded_file = st.sidebar.file_uploader(
    "上传招聘数据（Excel / CSV）",
    type=["xlsx", "xls", "csv"],
    key="file_uploader"
)
data_to_process = None
is_demo_mode = False

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
        st.stop()
elif st.session_state.use_demo and st.session_state.data_loaded_df is None:
    # 第一次 Rerun，如果处于 demo 模式且数据丢失，尝试再次加载 demo 数据
    try:
        data_to_process = pd.read_excel("拉勾招聘.xlsx")
        is_demo_mode = True
    except:
        pass  # 找不到文件，等待用户操作

# 如果有新数据需要处理，则进行处理并更新 Session State
if data_to_process is not None:
    st.session_state.data_loaded_df = process_and_engineer_data(data_to_process)
    st.session_state.use_demo = is_demo_mode

# --------------------------------------------------------------------------------
# 检查数据是否可用
# --------------------------------------------------------------------------------
df = st.session_state.data_loaded_df

if df is None:
    st.info("请通过左侧边栏上传数据或使用示例数据。")
    st.stop()
else:
    st.sidebar.success(f"数据加载成功，共 {len(df)} 条记录")



# 列名容错
df.columns = [c.strip() for c in df.columns]


# ==========================================================
# ========== 4. 数据清洗应用到当前 df （合并完成至此）========
# ==========================================================

# 城市
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

# ========== 数据清洗阶段结束 / 准备进入可视化部分 ==========
# ⚠⚠⚠ 下面即将开始绘图区域，将在第 2 次输出继续 ⚠⚠⚠
# ==========================================================
# ========== 5. 可视化增强模块（开始绘图） ==================
# ==========================================================

# 全局薪资模式选择器
salary_mode = st.radio(
    "选择薪资显示模式",
    ["原始 (raw)", "截尾 (clean)", "log(截尾)"],
    horizontal=True,
    index=1
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

tab1, tab2, tab3 = st.tabs(["📌 数据概览", "📊 可视化分析", "💰 在线薪资预测"])

# ########################################################
# # TAB1 — 数据概览 ########################################################
with tab1:
    st.subheader("数据样例")
    st.dataframe(df.head(20), width='stretch')
    st.metric("记录数量", len(df))
    st.metric("有薪资记录数量", df[sal_col].notna().sum())

with tab2:
    st.markdown("### 📊 全方位招聘数据可视化分析")
    st.markdown("---")
    
    # ======================================
    # 薪资分布分析（使用全局薪资模式）
    # ======================================
    st.header("📌 薪资分布分析")
    
    df_sal = df.dropna(subset=[sal_col])
    if df_sal.empty:
        st.warning("当前选择无薪资数据可展示")
    else:
        col_sal1, col_sal2 = st.columns(2)
        
        # 1. 薪资分布直方图+箱线图
        with col_sal1:
            fig_hist = px.histogram(
                df_sal,
                x=sal_col,
                nbins=40,
                marginal="box",
                title=f"薪资分布直方图 + 箱线图 ({salary_mode})",
                labels={sal_col: sal_name}
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # 2. 薪资小提琴图
        with col_sal2:
            fig_violin = px.violin(
                df_sal,
                y=sal_col,
                box=True,
                points="outliers",
                title=f"薪资小提琴图 ({salary_mode})",
                labels={sal_col: sal_name}
            )
            st.plotly_chart(fig_violin, use_container_width=True)
    
    st.markdown("---")

    # ======================================
    # 第二组：城市&行业分析
    # ======================================
    st.subheader("🌆 城市&行业竞争力分析")
    col_city1, col_city2 = st.columns(2)

    # 4. 城市平均薪资横向条形图
    with col_city1:
        required_cols = ["city", sal_col]
        if all(col in df.columns for col in required_cols) and all(df[col].notna().any() for col in required_cols):
            city_salary = df.groupby("city")[sal_col].agg(
                mean_salary="mean",
                count="count"
            ).dropna().query("count >= 5").sort_values("mean_salary", ascending=True)

            if not city_salary.empty:
                fig_city = px.bar(
                    city_salary.tail(15),
                    y=city_salary.tail(15).index,
                    x="mean_salary",
                    orientation="h",
                    title=f"城市平均薪资 Top15（样本量≥5） ({salary_mode})",
                    labels={"mean_salary": f"平均薪资 ({sal_name})", "y": "城市"},
                    color="count",
                    color_continuous_scale="Viridis",
                    hover_data=["count"]
                )
                st.plotly_chart(fig_city, config={"displayModeBar": True, "responsive": True})
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
        df.dropna(subset=["city_standard", "industry_agg", sal_col])
        .groupby(["city_standard", "industry_agg"], as_index=False)
        .agg(
            mean_salary=(sal_col, "mean"),
            job_count=(sal_col, "count")
        )
    )

    # 过滤样本数
    df_bubble = df_bubble[df_bubble["job_count"] >= int(min_combo)]

    # 保留 Top 城市
    city_toplist = df["city_standard"].value_counts().nlargest(top_cities).index.tolist()
    df_bubble = df_bubble[df_bubble["city_standard"].isin(city_toplist)]

    if df_bubble.empty:
        st.info("数据量不足，无法绘制气泡图。请降低阈值或增加展示城市数。")
    else:
        # 城市映射为整数并 jitter
        np.random.seed(int(jitter_seed))
        ordered_city = sorted(df_bubble["city_standard"].unique())
        city_to_num = {c: i for i, c in enumerate(ordered_city)}
        df_bubble["city_num"] = df_bubble["city_standard"].map(city_to_num)
        df_bubble["city_jitter"] = df_bubble["city_num"] + np.random.normal(0, 0.18, len(df_bubble))

        # 气泡大小 log 缩放
        df_bubble["bubble_size"] = np.log1p(df_bubble["job_count"])
        mn, mx = df_bubble["bubble_size"].min(), df_bubble["bubble_size"].max()
        df_bubble["bubble_scaled"] = 6 + (df_bubble["bubble_size"] - mn) / (mx - mn + 1e-9) * 34

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

        fig_bubble.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(len(ordered_city))),
                ticktext=ordered_city,
                tickangle=-45,
                title="城市"
            ),
            yaxis=dict(title=f"平均薪资 ({sal_name})")
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

    # ======================================
    # 第三组：经验&学历分析
    # ======================================
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
                    trendline="lowess",
                    trendline_options=dict(frac=0.3),
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
                    color_discrete_sequence=px.colors.qualitative.D3
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
        pivot_exp_edu = pivot_data.pivot_table(
            index="edu_level",
            columns="work_year_num",
            values=sal_col,
            aggfunc="mean"
        )
        # 过滤有效数据（至少2极行2列）
        if pivot_exp_edu.shape[0] >= 2 and pivot_exp_edu.shape[1] >= 2:
            edu_mapping = {0: "未知", 1: "高中/中专", 2: "大专", 3: "本科", 4: "硕士", 5: "博士"}
            y_labels = [edu_mapping.get(idx, f"等级{idx}") for idx in pivot_exp_edu.index]

            fig_heat = px.imshow(
                pivot_exp_edu,
                labels=dict(x="工作经验（年）", y="学历等级", color=f"平均薪资 ({sal_name})"),
                title=f"学历 × 经验 — 薪资热力图 ({salary_mode})",
                x=pivot_exp_edu.columns.astype(str),
                y=y_labels,
                color_continuous_scale="YlOrRd"
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("📌 学历/经验薪资数据不足（需至少2×2维度），无法绘制热力图")
    else:
        missing = [col for col in required_cols if col not in df.columns or not df[col].notna().any()]
        st.info(f"📌 缺少 {missing} 数据，无法绘制学历×经验薪资热力图")

    st.markdown("---")

    # ======================================
    # 第四组：岗位分析
    # ======================================
    st.subheader("💼 岗位需求分析")
    col_job1, col_job2 = st.columns(2)

    # 10. 岗位数量Top10表格
    with col_job1:
        if "positionName" in df.columns and df["positionName"].notna().any():
            position_count = df["positionName"].value_counts().head(10)
            if not position_count.empty:
                st.subheader("岗位数量 Top10")
                st.table(position_count.reset_index().rename(columns={"index": "岗位名称", "positionName": "招聘数量"}))
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
                    orientation="h",
                    title="热门岗位招聘需求 Top10",
                    color="招聘数量",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_job_bar, config={"displayModeBar": True, "responsive": True})
            else:
                st.info("📌 岗位数据为空，无法绘制热门岗位条形图")
        else:
            st.info("📌 缺少岗位名称（positionName）数据，无法绘制热门岗位条形图")

    # 12. 岗位词云
    if "positionName" in df.columns and df["positionName"].notna().any():
        st.subheader("岗位关键词词云")
        raw_text = " ".join(df["positionName"].dropna().astype(str))
        text = filter_keywords(raw_text) if 'filter_keywords' in locals() else raw_text

        if text:
            font_path = "msyh.ttc" if os.name == "nt" else None
            wc = WordCloud(
                width=800, height=400,
                background_color="white",
                font_path=font_path,
                max_words=50,
                min_font_size=8,
                max_font_size=48,
                collocations=False,
                random_state=42
            )
            fig_wc = plt.figure(figsize=(10, 5))
            wc_image = wc.generate(text)
            plt.imshow(wc_image, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(fig_wc)
            plt.close(fig_wc)
        else:
            st.warning("📌 暂无有效岗位关键词可生成词云")
    else:
        st.info("📌 缺少岗位名称（positionName）数据，无法生成词云")
    st.set_page_config(layout="wide")



# ---- 4. 标准化城市名 ----
#



st.subheader("🔍 深度进阶分析")

    # 14. 行业-岗位薪资热力图
required_cols = ["industryField", "bigcategory", "salary_k"]
if all(col in df.columns for col in required_cols) and all(df[col].notna().any() for col in required_cols):
        top_industries = df["industryField"].value_counts().head(10).index
        top_categories = df["bigcategory"].value_counts().head(10).index

        heat_data = df[
            df["industryField"].isin(top_industries) &
            df["bigcategory"].isin(top_categories)
            ].groupby(["industryField", "bigcategory"])["salary_k"].mean().unstack()

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
            scene=dict(
                xaxis=dict(
                    tickmode="array",
                    tickvals=list(range(len(city3))),
                    ticktext=city3,
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

    # 16. 薪资正态性检验QQ图
if "salary_k" in df.columns and df["salary_k"].notna().any():
        sal_data = df["salary_k"].dropna()
        if len(sal_data) > 100:
            from scipy import stats

            qq_data = stats.probplot(sal_data, dist="norm")
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
            fig_qq.add_shape(
                type="line",
                x0=qq_df["理论分位数"].min(),
                y0=qq_df["理论分位数"].min(),
                x1=qq_df["理论分位数"].max(),
                y1=qq_df["理论分位数"].max(),
                line=dict(color="red", dash="dash")
            )
            st.plotly_chart(fig_qq, use_container_width=True)
        else:
            st.info("📌 样本量不足（需至少100条薪资数据），无法进行正态性检验")
else:
        st.info("📌 缺少薪资数据（salary_k），无法进行正态性检验")

# 第 3 部分将紧跟此处继续拼接
# ==========================================================
# ============ 6. 导出、下载与数据诊断功能 ==================
# ==========================================================


# ==========================================================
# ============ 7. 保留原项目功能的挂载接口 =================
# ==========================================================
# ⚠ 若你原始 0.py 有以下功能，这里已预留兼容接口：
#  ✔ 岗位推荐 / AI 推荐算法
#  ✔ 求职预测模型 / 矩阵评分模块
#  ✔ 简历评分或岗位匹配
#  ✔ echarts 模块或 pandas profiling
#  ✔ 分页显示岗位表格
#
# 若你的原项目在底部包含 main()、router、tab、button UI 等结构，
# 请确保继续写在下方（不必改动）。下面是兼容衔接区：
with tab3:
    model_path = "salary_predictor_rf.joblib"


    # --------------------------------------------------------
    # 核心模型加载 (使用 st.cache_resource)
    # --------------------------------------------------------
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


    # 检查并设置 session_state.model，优先使用缓存
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

    # --------------------------------------------------------
    # 模型训练逻辑
    # --------------------------------------------------------
    if st.checkbox("训练新模型（推荐）", key='train_model_checkbox'):
        with st.spinner('⏳ 正在训练模型，请稍候...'):

            # 1. 准备数据 (使用 df，它指向 session state 的数据)
            train_df = df.dropna(
                subset=["salary_k", "city", "industryField", "bigcategory", "positionName", "work_year_num",
                        "edu_level"])

            if len(train_df) < 10:
                st.error("❌ 有效训练数据不足，无法训练模型。")
            else:
                if len(train_df) < 100:
                    st.warning(f"⚠️ 有效训练数据不足（仅 {len(train_df)} 条），模型准确度可能较低。")

                X = train_df[["city", "industryField", "bigcategory", "positionName", "work_year_num", "edu_level"]]
                y = train_df["salary_k"]

                pipe = Pipeline([
                    ("pre", ColumnTransformer([
                        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                         ["work_year_num", "edu_level"]),
                        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="未知")),
                                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]),
                         ["city", "industryField", "bigcategory", "positionName"]),
                    ])),
                    ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
                ])

                # 2. 训练与评估
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=max(0.2, 10 / len(X)), random_state=42)
                pipe.fit(Xtr, ytr)
                pred = pipe.predict(Xte)

                # 3. 存储和缓存管理
                st.session_state.model = pipe  # 将训练好的模型存入 session_state
                joblib.dump(pipe, model_path)
                load_predictor_model.clear()  # 清除 st.cache_resource 缓存

                st.success(
                    f"✅ 模型训练与保存完成。**MAE**={mean_absolute_error(yte, pred):.3f} k/月，**R²**={r2_score(yte, pred):.3f}")

    # --------------------------------------------------------
    # 模型状态提示 (现在只在没有模型且未勾选训练时显示)
    # --------------------------------------------------------
    elif st.session_state.model is None and st.session_state.data_loaded_df is not None:
        st.subheader("模型状态")
        st.warning("⚠️ 尚未加载或训练模型，请勾选 **训练新模型** 选项进行训练。")

    # --------------------------------------------------------
    # 在线单条薪资预测
    # --------------------------------------------------------
    st.markdown("---")
    st.subheader("在线单条薪资预测")

    with st.form("pred"):
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
            city_in = st.selectbox("工作城市", cities, index=cities.index(default_city) if default_city else 0)
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

                p = st.session_state.model.predict(sample)[0]
                st.balloons()
                st.success(f"🎉 预测结果：**💰 {p:.2f} k/月** ≈ **{p * 1000:.0f} 元/月**")

# ==========================================================
# ============ 8. 页面尾部（版权 & 提示） ==================
# ==========================================================

st.markdown("---")
st.markdown("")
