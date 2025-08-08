"""
完整配置管理（包含Gemini API）- 保存为 config.py
"""

import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# ========== API密钥 ==========
COINDESK_API_KEY = os.getenv('COINDESK_API_KEY', '')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

# ========== 数据收集 ==========
START_DATE = os.getenv('START_DATE', '2021-08-01')
END_DATE = os.getenv('END_DATE', '2025-08-06')
TOP_N_COINS = int(os.getenv('TOP_N_COINS', '200'))

# ========== 数据库 ==========
DB_PATH = os.getenv('DB_PATH', 'data/cache/crypto_cache.db')

# ========== 应用 ==========
APP_PORT = int(os.getenv('APP_PORT', '8050'))
APP_DEBUG = os.getenv('APP_DEBUG', 'False').lower() == 'true'

# ========== Gemini AI设置 ==========
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-pro')
GEMINI_TEMPERATURE = float(os.getenv('GEMINI_TEMPERATURE', '0.7'))
GEMINI_MAX_TOKENS = int(os.getenv('GEMINI_MAX_TOKENS', '2048'))

def get_collection_config():
    """获取数据收集配置"""
    return {
        'api_key': COINDESK_API_KEY,
        'start_date': START_DATE,
        'end_date': END_DATE,
        'target_coins': TOP_N_COINS,
        'cache_path': DB_PATH,
        'output_dir': 'data',
        'batch_size': 20,
        'rate_limit_delay': 0.1,
        'max_concurrent': 25,
        'news_batch_size': 200,
        'news_parallel_days': 10
    }

def get_gemini_config():
    """获取Gemini AI配置"""
    return {
        'api_key': GEMINI_API_KEY,
        'model': GEMINI_MODEL,
        'temperature': GEMINI_TEMPERATURE,
        'max_output_tokens': GEMINI_MAX_TOKENS,
        'safety_settings': [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
    }

def check_config():
    """检查配置是否正确"""
    print("\n" + "="*60)
    print("配置检查")
    print("="*60)

    all_good = True

    # 检查CoinDesk API
    if COINDESK_API_KEY and COINDESK_API_KEY != 'your_api_key_here':
        masked = COINDESK_API_KEY[:10] + "..." + COINDESK_API_KEY[-4:]
        print(f"✅ CoinDesk API Key: {masked}")
    else:
        print("❌ CoinDesk API Key 未配置")
        print("   请在 .env 文件中设置: COINDESK_API_KEY=你的密钥")
        all_good = False

    # 检查Gemini API
    if GEMINI_API_KEY and GEMINI_API_KEY != 'your_gemini_api_key_here':
        masked = GEMINI_API_KEY[:10] + "..." + GEMINI_API_KEY[-4:]
        print(f"✅ Gemini API Key: {masked}")
        print(f"   模型: {GEMINI_MODEL}")
        print(f"   温度: {GEMINI_TEMPERATURE}")
    else:
        print("⚠️  Gemini API Key 未配置（AI顾问功能将不可用）")
        print("   获取密钥: https://makersuite.google.com/app/apikey")
        print("   然后在 .env 中设置: GEMINI_API_KEY=你的密钥")

    # 显示其他配置
    print(f"\n📊 数据收集配置:")
    print(f"   日期范围: {START_DATE} 到 {END_DATE}")
    print(f"   加密货币数量: {TOP_N_COINS}")
    print(f"   数据库路径: {DB_PATH}")

    print(f"\n🌐 应用配置:")
    print(f"   端口: {APP_PORT}")
    print(f"   调试模式: {APP_DEBUG}")

    print("="*60)

    if all_good:
        print("✅ 核心配置正确！可以开始使用")
    else:
        print("❌ 请先配置必需的API密钥")

    return all_good

def test_gemini_connection():
    """测试Gemini API连接"""
    if not GEMINI_API_KEY or GEMINI_API_KEY == 'your_gemini_api_key_here':
        print("❌ Gemini API Key 未配置")
        return False

    print("🔍 测试Gemini API连接...")

    try:
        import google.generativeai as genai

        # 配置API
        genai.configure(api_key=GEMINI_API_KEY)

        # 创建模型
        model = genai.GenerativeModel(GEMINI_MODEL)

        # 测试简单查询
        response = model.generate_content("Say 'API connection successful' if you can read this.")

        print(f"✅ Gemini API连接成功!")
        print(f"   响应: {response.text[:100]}...")
        return True

    except Exception as e:
        print(f"❌ Gemini API连接失败: {e}")
        return False

if __name__ == "__main__":
    # 检查配置
    check_config()

    # 如果配置了Gemini，测试连接
    if GEMINI_API_KEY and GEMINI_API_KEY != 'your_gemini_api_key_here':
        print("\n" + "="*60)
        test_gemini_connection()