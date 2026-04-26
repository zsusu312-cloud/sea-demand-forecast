"""
各国消费特征配置 + 节假日定义
覆盖 SG / MY / TH / PH / VN 五国
"""

import pandas as pd

# ── 国家基础配置 ──────────────────────────────────────────────
COUNTRY_PROFILE = {
    "SG": {
        "name": "Singapore",
        "currency": "SGD",
        "base_demand_multiplier": 1.6,   # 人均消费高
        "price_sensitivity": 0.4,         # 低价格敏感度（富裕市场）
        "promo_sensitivity": 1.2,         # 对促销响应适中
        "weekend_boost": 1.25,
        "top_categories": ["Electronics", "Beauty", "Fashion"],
        "language": "EN",
        "notes": "高收入，偏好品质/品牌，科技产品渗透率高",
    },
    "MY": {
        "name": "Malaysia",
        "currency": "MYR",
        "base_demand_multiplier": 1.2,
        "price_sensitivity": 0.7,
        "promo_sensitivity": 1.5,
        "weekend_boost": 1.3,
        "top_categories": ["Fashion", "Beauty", "Home"],
        "language": "MY/EN",
        "notes": "多民族市场(马来/华/印)，斋月效应显著，halal重要",
    },
    "TH": {
        "name": "Thailand",
        "currency": "THB",
        "base_demand_multiplier": 1.1,
        "price_sensitivity": 0.65,
        "promo_sensitivity": 1.6,
        "weekend_boost": 1.2,
        "top_categories": ["Beauty", "Fashion", "Toys"],
        "language": "TH",
        "notes": "美妆/时尚驱动，受K-pop/社媒影响大，宋干节爆发",
    },
    "PH": {
        "name": "Philippines",
        "currency": "PHP",
        "base_demand_multiplier": 0.9,
        "price_sensitivity": 0.8,
        "promo_sensitivity": 1.8,         # 对大促非常敏感
        "weekend_boost": 1.35,
        "top_categories": ["Fashion", "Toys", "Electronics"],
        "language": "EN/FIL",
        "notes": "圣诞季Q4占全年消费比例极高，OFW汇款驱动消费",
    },
    "VN": {
        "name": "Vietnam",
        "currency": "VND",
        "base_demand_multiplier": 0.85,
        "price_sensitivity": 0.9,         # 最高价格敏感度
        "promo_sensitivity": 2.0,
        "weekend_boost": 1.15,
        "top_categories": ["Fashion", "Home", "Beauty"],
        "language": "VN",
        "notes": "价格极度敏感，春节(Tet)前需求爆发，中产快速成长",
    },
}

# ── 品类配置 ──────────────────────────────────────────────────
CATEGORIES = {
    "Electronics": {
        "base_price_range": (50, 300),
        "avg_daily_base": 80,
        "seasonality_strength": 0.3,
        "skus": [
            "TWS_Earbuds_Pro", "SmartWatch_Basic", "PowerBank_20000",
            "USB_Hub_7Port", "Webcam_HD1080", "LED_DeskLamp",
            "Bluetooth_Speaker_Mini",
        ],
    },
    "Fashion": {
        "base_price_range": (10, 60),
        "avg_daily_base": 150,
        "seasonality_strength": 0.5,
        "skus": [
            "Sneakers_Classic_White", "Crossbody_Bag_Canvas",
            "Sunglasses_UV400", "Baseball_Cap_Logo",
            "Compression_Leggings", "Casual_Tshirt_Pack3",
        ],
    },
    "Beauty": {
        "base_price_range": (8, 80),
        "avg_daily_base": 200,
        "seasonality_strength": 0.35,
        "skus": [
            "Vitamin_C_Serum_30ml", "SPF50_Sunscreen_100ml",
            "Sheet_Mask_10pcs", "Lip_Tint_Korean",
            "Hair_Mask_Coconut", "Jade_Roller_Set",
            "Retinol_Night_Cream",
        ],
    },
    "Home": {
        "base_price_range": (15, 120),
        "avg_daily_base": 100,
        "seasonality_strength": 0.4,
        "skus": [
            "Bamboo_Cutting_Board", "Silicone_Storage_Bags",
            "Aromatherapy_Diffuser", "Magnetic_Spice_Rack",
            "Weighted_Blanket_5kg", "Cable_Management_Box",
        ],
    },
    "Sports": {
        "base_price_range": (20, 150),
        "avg_daily_base": 60,
        "seasonality_strength": 0.45,
        "skus": [
            "Resistance_Bands_Set", "Yoga_Mat_6mm",
            "Jump_Rope_Speed", "Foam_Roller_Grid",
            "Gym_Gloves_XL",
        ],
    },
    "Toys": {
        "base_price_range": (10, 80),
        "avg_daily_base": 90,
        "seasonality_strength": 0.6,   # 节假日效应最强
        "skus": [
            "Building_Blocks_500pcs", "RC_Car_Offroad",
            "Slime_Kit_DIY", "Fidget_Cube_Pack",
            "Puzzle_1000pcs_Animals", "Play_Dough_Set",
        ],
    },
}

# ── 节假日配置（含提前备货窗口 pre_days 和消退窗口 post_days）─────
# boost_by_category: 不同品类在该节日的需求倍数（相对基线）

HOLIDAYS = {
    "SG": [
        {
            "name": "Chinese_New_Year",
            "dates": ["2023-01-22", "2024-02-10", "2025-01-29"],
            "pre_days": 21, "post_days": 7,
            "peak_boost": 2.5,
            "boost_by_category": {
                "Electronics": 2.0, "Fashion": 2.8, "Beauty": 2.5,
                "Home": 3.0, "Sports": 1.2, "Toys": 2.2,
            },
        },
        {
            "name": "Hari_Raya_Puasa",
            "dates": ["2023-04-21", "2024-04-10", "2025-03-30"],
            "pre_days": 14, "post_days": 5,
            "peak_boost": 1.8,
            "boost_by_category": {
                "Electronics": 1.5, "Fashion": 2.5, "Beauty": 2.0,
                "Home": 1.8, "Sports": 1.1, "Toys": 1.6,
            },
        },
        {
            "name": "Deepavali",
            "dates": ["2023-11-12", "2024-11-01", "2025-10-20"],
            "pre_days": 14, "post_days": 3,
            "peak_boost": 1.6,
            "boost_by_category": {
                "Electronics": 1.4, "Fashion": 2.2, "Beauty": 2.0,
                "Home": 1.9, "Sports": 1.0, "Toys": 1.5,
            },
        },
        {
            "name": "National_Day",
            "dates": ["2023-08-09", "2024-08-09", "2025-08-09"],
            "pre_days": 7, "post_days": 2,
            "peak_boost": 1.4,
            "boost_by_category": {
                "Electronics": 1.6, "Fashion": 1.5, "Beauty": 1.2,
                "Home": 1.3, "Sports": 1.4, "Toys": 1.3,
            },
        },
        {
            "name": "Christmas",
            "dates": ["2023-12-25", "2024-12-25", "2025-12-25"],
            "pre_days": 21, "post_days": 5,
            "peak_boost": 2.0,
            "boost_by_category": {
                "Electronics": 2.2, "Fashion": 1.8, "Beauty": 1.8,
                "Home": 1.6, "Sports": 1.4, "Toys": 3.0,
            },
        },
    ],

    "MY": [
        {
            "name": "Chinese_New_Year",
            "dates": ["2023-01-22", "2024-02-10", "2025-01-29"],
            "pre_days": 21, "post_days": 7,
            "peak_boost": 2.3,
            "boost_by_category": {
                "Electronics": 1.8, "Fashion": 2.6, "Beauty": 2.3,
                "Home": 2.8, "Sports": 1.1, "Toys": 2.0,
            },
        },
        {
            "name": "Hari_Raya_Puasa",  # 最重要，马来族占60%+
            "dates": ["2023-04-21", "2024-04-10", "2025-03-30"],
            "pre_days": 28, "post_days": 7,
            "peak_boost": 3.0,
            "boost_by_category": {
                "Electronics": 2.0, "Fashion": 3.5, "Beauty": 2.8,
                "Home": 2.5, "Sports": 1.2, "Toys": 2.0,
            },
        },
        {
            "name": "Ramadan_Start",   # 斋月期间部分品类下降
            "dates": ["2023-03-23", "2024-03-11", "2025-03-01"],
            "pre_days": 0, "post_days": 28,
            "peak_boost": 0.8,         # 斋月期间部分品类需求下降
            "boost_by_category": {
                "Electronics": 0.9, "Fashion": 1.3, "Beauty": 0.8,
                "Home": 0.7, "Sports": 0.6, "Toys": 0.7,
            },
        },
        {
            "name": "Hari_Raya_Haji",
            "dates": ["2023-06-28", "2024-06-17", "2025-06-06"],
            "pre_days": 10, "post_days": 3,
            "peak_boost": 1.7,
            "boost_by_category": {
                "Electronics": 1.4, "Fashion": 2.0, "Beauty": 1.6,
                "Home": 1.8, "Sports": 1.1, "Toys": 1.4,
            },
        },
        {
            "name": "Malaysia_Day",
            "dates": ["2023-09-16", "2024-09-16", "2025-09-16"],
            "pre_days": 7, "post_days": 2,
            "peak_boost": 1.3,
            "boost_by_category": {
                "Electronics": 1.4, "Fashion": 1.4, "Beauty": 1.2,
                "Home": 1.3, "Sports": 1.3, "Toys": 1.2,
            },
        },
    ],

    "TH": [
        {
            "name": "Songkran",  # 泰国最重要节日，4月泼水节
            "dates": ["2023-04-13", "2024-04-13", "2025-04-13"],
            "pre_days": 14, "post_days": 5,
            "peak_boost": 3.2,
            "boost_by_category": {
                "Electronics": 2.0, "Fashion": 2.5, "Beauty": 3.5,
                "Home": 1.5, "Sports": 2.0, "Toys": 3.0,
            },
        },
        {
            "name": "Loy_Krathong",  # 水灯节11月
            "dates": ["2023-11-27", "2024-11-15", "2025-11-05"],
            "pre_days": 10, "post_days": 3,
            "peak_boost": 2.0,
            "boost_by_category": {
                "Electronics": 1.3, "Fashion": 2.2, "Beauty": 2.5,
                "Home": 2.0, "Sports": 1.1, "Toys": 1.8,
            },
        },
        {
            "name": "Thai_New_Year",
            "dates": ["2023-01-01", "2024-01-01", "2025-01-01"],
            "pre_days": 10, "post_days": 5,
            "peak_boost": 1.8,
            "boost_by_category": {
                "Electronics": 1.6, "Fashion": 2.0, "Beauty": 1.8,
                "Home": 1.5, "Sports": 1.3, "Toys": 1.6,
            },
        },
        {
            "name": "King_Birthday",
            "dates": ["2023-07-28", "2024-07-28", "2025-07-28"],
            "pre_days": 7, "post_days": 2,
            "peak_boost": 1.4,
            "boost_by_category": {
                "Electronics": 1.3, "Fashion": 1.5, "Beauty": 1.3,
                "Home": 1.4, "Sports": 1.2, "Toys": 1.3,
            },
        },
        {
            "name": "Christmas_TH",  # 泰国基督徒少，但商业化驱动有小高峰
            "dates": ["2023-12-25", "2024-12-25", "2025-12-25"],
            "pre_days": 14, "post_days": 3,
            "peak_boost": 1.5,
            "boost_by_category": {
                "Electronics": 1.6, "Fashion": 1.8, "Beauty": 1.7,
                "Home": 1.4, "Sports": 1.2, "Toys": 2.0,
            },
        },
    ],

    "PH": [
        {
            "name": "Christmas",  # 菲律宾最重要，Q4占全年40%+
            "dates": ["2023-12-25", "2024-12-25", "2025-12-25"],
            "pre_days": 90,  # 9月起即"Ber months"圣诞季开始
            "post_days": 7,
            "peak_boost": 4.0,
            "boost_by_category": {
                "Electronics": 3.5, "Fashion": 3.0, "Beauty": 3.0,
                "Home": 2.5, "Sports": 2.0, "Toys": 5.0,
            },
        },
        {
            "name": "Holy_Week",  # 圣周，部分品类下滑
            "dates": ["2023-04-07", "2024-03-29", "2025-04-18"],
            "pre_days": 0, "post_days": 7,
            "peak_boost": 0.6,
            "boost_by_category": {
                "Electronics": 0.5, "Fashion": 0.7, "Beauty": 0.8,
                "Home": 0.6, "Sports": 0.5, "Toys": 0.6,
            },
        },
        {
            "name": "Independence_Day",
            "dates": ["2023-06-12", "2024-06-12", "2025-06-12"],
            "pre_days": 7, "post_days": 2,
            "peak_boost": 1.5,
            "boost_by_category": {
                "Electronics": 1.4, "Fashion": 1.6, "Beauty": 1.4,
                "Home": 1.3, "Sports": 1.3, "Toys": 1.5,
            },
        },
        {
            "name": "Undas",  # 万圣节/诸圣节，菲律宾独特
            "dates": ["2023-11-01", "2024-11-01", "2025-11-01"],
            "pre_days": 5, "post_days": 2,
            "peak_boost": 0.7,
            "boost_by_category": {
                "Electronics": 0.8, "Fashion": 0.7, "Beauty": 0.8,
                "Home": 0.7, "Sports": 0.6, "Toys": 0.7,
            },
        },
        {
            "name": "PH_New_Year",
            "dates": ["2023-01-01", "2024-01-01", "2025-01-01"],
            "pre_days": 7, "post_days": 3,
            "peak_boost": 1.6,
            "boost_by_category": {
                "Electronics": 1.5, "Fashion": 1.8, "Beauty": 1.5,
                "Home": 1.4, "Sports": 1.3, "Toys": 1.7,
            },
        },
    ],

    "VN": [
        {
            "name": "Tet",  # 越南最重要节日，春节前后
            "dates": ["2023-01-22", "2024-02-10", "2025-01-29"],
            "pre_days": 28, "post_days": 10,
            "peak_boost": 4.5,
            "boost_by_category": {
                "Electronics": 3.0, "Fashion": 4.0, "Beauty": 3.5,
                "Home": 5.0, "Sports": 1.5, "Toys": 4.0,
            },
        },
        {
            "name": "Tet_Post_Slowdown",  # Tet后1-2周需求骤降
            "dates": ["2023-02-01", "2024-02-20", "2025-02-08"],
            "pre_days": 0, "post_days": 14,
            "peak_boost": 0.4,
            "boost_by_category": {
                "Electronics": 0.5, "Fashion": 0.4, "Beauty": 0.5,
                "Home": 0.3, "Sports": 0.5, "Toys": 0.4,
            },
        },
        {
            "name": "National_Day_VN",  # 9月2日国庆
            "dates": ["2023-09-02", "2024-09-02", "2025-09-02"],
            "pre_days": 7, "post_days": 3,
            "peak_boost": 1.8,
            "boost_by_category": {
                "Electronics": 1.6, "Fashion": 1.8, "Beauty": 1.5,
                "Home": 1.5, "Sports": 1.4, "Toys": 1.6,
            },
        },
        {
            "name": "Mid_Autumn_VN",  # 中秋，儿童节礼物消费
            "dates": ["2023-09-29", "2024-09-17", "2025-10-06"],
            "pre_days": 14, "post_days": 3,
            "peak_boost": 2.0,
            "boost_by_category": {
                "Electronics": 1.4, "Fashion": 1.6, "Beauty": 1.8,
                "Home": 1.5, "Sports": 1.2, "Toys": 3.5,
            },
        },
        {
            "name": "Christmas_VN",  # 年轻人商业化驱动
            "dates": ["2023-12-25", "2024-12-25", "2025-12-25"],
            "pre_days": 14, "post_days": 3,
            "peak_boost": 1.8,
            "boost_by_category": {
                "Electronics": 1.7, "Fashion": 2.0, "Beauty": 1.9,
                "Home": 1.4, "Sports": 1.2, "Toys": 2.2,
            },
        },
    ],
}

# ── 平台大促日历（全区通用，各国敏感度不同）─────────────────────
PLATFORM_PROMOS = [
    # (月, 日, 名称, 持续天数, 平均boost)
    (3, 28, "3.28_Sale", 2, 1.8),
    (4, 4,  "4.4_Sale",  1, 1.6),
    (5, 5,  "5.5_Sale",  2, 2.0),
    (6, 6,  "6.6_Sale",  2, 2.1),
    (7, 7,  "7.7_Sale",  2, 1.9),
    (8, 8,  "8.8_Sale",  2, 2.0),
    (9, 9,  "9.9_Sale",  3, 2.5),
    (10, 10,"10.10_Sale",3, 2.3),
    (11, 11,"11.11_Sale",3, 3.5),   # 最大
    (12, 12,"12.12_Sale",3, 3.0),
]
