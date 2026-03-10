"""
类人脑双系统全闭环AI架构 - Telegram Bot服务 (训练后版)
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TELEGRAM_BOT_TOKEN = os.environ.get(
    "TELEGRAM_BOT_TOKEN",
    "8728812786:AAHl8RFJdXSZyjdQTaiaG6vCA1aaDxXsNq0"
)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        from core.truly_integrated_engine import TrulyIntegratedEngine
        model_path = str(PROJECT_ROOT / "weights/Qwen3.5-0.8B-Base")
        _engine = TrulyIntegratedEngine(model_path)
    return _engine


async def run_bot():
    from telegram import Update
    from telegram.ext import (
        Application, CommandHandler, MessageHandler,
        filters, ContextTypes
    )
    from telegram.constants import ParseMode
    
    logger.info("Bot is starting (Engine will load lazily)...")
    
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "类人脑双系统AI (训练后版)\n\n"
            "已加载训练后的动态权重\n\n"
            "命令：\n"
            "/start - 开始\n"
            "/test - 测试推理能力",
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def test_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
        test_questions = [
            "房租1600元租了20天，日租金是多少？",
            "房租1600元租了20天，月租金是多少？",
        ]
        
        await update.message.reply_text("正在启动批量测试引擎...")
        engine = get_engine()
        if not engine._initialized:
            engine.initialize()
            
        results = []
        for q in test_questions:
            answer = engine.generate(q)
            results.append(f"Q: {q}\nA: {answer}")
        
        await update.message.reply_text("\n\n".join(results))
    
    async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_message = update.message.text
        print(f">>> Received: {user_message}")
        
        if not user_message: return
        
        engine = get_engine()
        message = await update.message.reply_text("已收到！类脑引擎正在准备中...")
        
        if not engine._initialized:
            await message.edit_text("类脑引擎（Qwen3.5-0.8B）初次运行，正在加载权重 (约需1-2分钟)，请耐心等待...")
            if not engine.initialize():
                await message.edit_text("❌ 引擎加载失败，请检查模型文件。")
                return
            await message.edit_text("✅ 引擎加载完成！现在开始思考您的指令...")
        
        await update.message.chat.send_action("typing")
        
        try:
            response_text = ""
            import time
            last_edit_time = time.time()
            edit_interval = 3.0  # 极度保守：3秒刷新一次，防止电报限制
            last_edit_len = 0
            
            message = await update.message.reply_text("思考中...")
            
            for token in engine.generate_stream(user_message):
                response_text += token
                
                # 仅在时间间隔超过 edit_interval 且内容有显著变化时更新 UI
                if time.time() - last_edit_time >= edit_interval and len(response_text) - last_edit_len >= 15:
                    try:
                        # 确保 <think> 块在流式传输中可见
                        display_text = response_text
                        if "<think>" in display_text and "</think>" not in display_text:
                           display_text = display_text + "\n\n(正在思考中...)"
                        
                        await message.edit_text(f"思考中...\n\n{display_text}")
                        last_edit_time = time.time()
                        last_edit_len = len(response_text)
                    except Exception:
                        pass
            
            # 最后发一次完整的，务必去掉 "思考中..." 提示
            if response_text:
                try:
                    await message.edit_text(response_text)
                except Exception:
                    await update.message.reply_text(response_text)
            else:
                await message.edit_text("抱歉，类脑引擎未产生有效输出。")
            
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            import traceback
            traceback.print_exc()
            await update.message.reply_text(f"❌ 处理失败: {str(e)}")
    
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("test", test_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info("类人脑AI Telegram Bot (训练后版) 启动中...")
    
    await application.initialize()
    await application.start()
    await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
    
    logger.info("Bot启动成功！等待消息...")
    
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("正在停止Bot...")
        await application.updater.stop()
        await application.stop()


def main():
    print("=" * 60)
    print("类人脑双系统AI架构 - Telegram Bot (训练后版)")
    print("=" * 60)
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
