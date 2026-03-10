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
    "8534413276:AAHzqgxVTOL2fapd8NV7UjppF4NXr1zSUek"
)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        from core.trained_engine import TrainedInferenceEngine
        _engine = TrainedInferenceEngine()
    return _engine


async def run_bot():
    from telegram import Update
    from telegram.ext import (
        Application, CommandHandler, MessageHandler,
        filters, ContextTypes
    )
    from telegram.constants import ParseMode
    
    engine = get_engine()
    
    logger.info("正在加载训练后的模型...")
    if not engine.initialize():
        logger.error("模型加载失败")
        return
    
    logger.info("模型加载成功！")
    
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🧠 *类人脑双系统AI (训练后版)*\n\n"
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
        
        results = []
        for q in test_questions:
            answer = engine.generate(q)
            results.append(f"Q: {q}\nA: {answer}")
        
        await update.message.reply_text("\n\n".join(results))
    
    async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_message = update.message.text
        
        if not user_message:
            return
        
        await update.message.chat.send_action("typing")
        
        try:
            response_text = ""
            chunk_size = 20
            last_sent_len = 0
            message = None
            
            for token in engine.generate_stream(user_message):
                response_text += token
                
                if len(response_text) - last_sent_len >= chunk_size:
                    try:
                        if message is None:
                            message = await update.message.reply_text(response_text)
                        else:
                            await message.edit_text(response_text)
                        last_sent_len = len(response_text)
                    except Exception:
                        pass
            
            if response_text:
                if message is None:
                    await update.message.reply_text(response_text)
                else:
                    try:
                        await message.edit_text(response_text)
                    except Exception:
                        await update.message.reply_text(response_text)
            else:
                await update.message.reply_text("抱歉，我无法生成回复。")
            
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            import traceback
            traceback.print_exc()
            await update.message.reply_text(f"❌ 处理失败: {str(e)}")
    
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("test", test_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info("🧠 类人脑AI Telegram Bot (训练后版) 启动中...")
    
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
    print("🧠 类人脑双系统AI架构 - Telegram Bot (训练后版)")
    print("=" * 60)
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
