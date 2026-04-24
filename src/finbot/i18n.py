from __future__ import annotations

from finbot.schemas import LanguageCode

EN = LanguageCode.EN.value
VI = LanguageCode.VI.value
ZH = LanguageCode.ZH.value

TASK_PROMPT: dict[str, str] = {
    EN: (
        "Hi! I'm your personal financial assistant. "
        "To get started, what would you like help with today?\n\n"
        "[1] Financial Planning\n"
        "[2] Investment Strategy\n"
        "[3] Trading Guidance"
    ),
    VI: (
        "Xin chào! Tôi là trợ lý tài chính cá nhân của bạn. "
        "Để bắt đầu, hôm nay bạn cần hỗ trợ về lĩnh vực nào?\n\n"
        "[1] Lập kế hoạch tài chính\n"
        "[2] Chiến lược đầu tư\n"
        "[3] Hướng dẫn giao dịch"
    ),
    ZH: (
        "你好！我是您的个人财务助手。"
        "请问今天您希望我在哪方面为您提供帮助？\n\n"
        "[1] 财务规划\n"
        "[2] 投资策略\n"
        "[3] 交易指导"
    ),
}

FIELD_QUESTIONS: dict[str, dict[str, str]] = {
    "GOAL": {
        EN: "Great choice! To tailor my advice, could you briefly describe your main financial goal? (e.g. buy a home, retire early, grow savings)",
        VI: "Tuyệt vời! Để tôi có thể tư vấn phù hợp, bạn có thể mô tả ngắn gọn mục tiêu tài chính chính của mình không? (ví dụ: mua nhà, nghỉ hưu sớm, tích lũy tiết kiệm)",
        ZH: "很好！为了给您提供有针对性的建议，能简单描述一下您的主要财务目标吗？（例如：买房、提前退休、积累储蓄）",
    },
    "INCOME_BAND": {
        EN: "Thanks! To better understand your situation, which range best describes your annual income?\n[1] Under $60K\n[2] $60K–$120K\n[3] $120K or above",
        VI: "Cảm ơn bạn! Để hiểu rõ hơn về tình hình của bạn, thu nhập hàng năm của bạn thuộc khoảng nào?\n[1] Dưới 60K\n[2] 60K–120K\n[3] Từ 120K trở lên",
        ZH: "谢谢！为了更好地了解您的情况，请问您的年收入大致属于哪个区间？\n[1] 6万以下\n[2] 6万至12万\n[3] 12万及以上",
    },
    "CAPITAL_RANGE": {
        EN: "Got it. How much are you looking to invest? Pick the closest range:\n[1] Under $10K\n[2] $10K–$50K\n[3] $50K–$250K\n[4] $250K or above",
        VI: "Hiểu rồi. Bạn đang muốn đầu tư khoảng bao nhiêu? Chọn khoảng gần nhất:\n[1] Dưới 10K\n[2] 10K–50K\n[3] 50K–250K\n[4] Từ 250K trở lên",
        ZH: "明白了。您计划投入多少资金？请选择最接近的范围：\n[1] 1万以下\n[2] 1万至5万\n[3] 5万至25万\n[4] 25万及以上",
    },
    "TIME_HORIZON": {
        EN: "How long are you planning to keep this investment?\n[1] Short-term — under 1 year\n[2] Medium-term — 1 to 5 years\n[3] Long-term — 5 years or more",
        VI: "Bạn dự định duy trì khoản đầu tư này trong bao lâu?\n[1] Ngắn hạn — dưới 1 năm\n[2] Trung hạn — 1 đến 5 năm\n[3] Dài hạn — trên 5 năm",
        ZH: "您计划持有这笔投资多长时间？\n[1] 短期 — 1年以内\n[2] 中期 — 1至5年\n[3] 长期 — 5年以上",
    },
    "RISK_TOLERANCE": {
        EN: "Last one — how comfortable are you with risk?\n[1] Low — I prefer safe, steady returns\n[2] Medium — I can handle some ups and downs\n[3] High — I'm okay with big swings for bigger gains",
        VI: "Câu cuối — bạn cảm thấy thế nào về rủi ro?\n[1] Thấp — tôi thích lợi nhuận ổn định, an toàn\n[2] Trung bình — tôi chấp nhận được biến động vừa phải\n[3] Cao — tôi chấp nhận rủi ro lớn để có lợi nhuận cao",
        ZH: "最后一个问题 — 您对风险的态度是？\n[1] 低 — 偏好稳健收益\n[2] 中 — 可以接受一定波动\n[3] 高 — 愿意承担较大风险以获取高回报",
    },
}

CLARIFY_SUFFIX: dict[str, str] = {
    EN: "\n\nNo worries if you're not sure — just type 'skip' and I'll move on.",
    VI: "\n\nKhông sao nếu bạn chưa chắc — chỉ cần gõ 'bỏ qua' và tôi sẽ tiếp tục.",
    ZH: "\n\n不确定也没关系，输入「跳过」我就继续下一步。",
}

TOO_MANY_UNKNOWN_PREFIX: dict[str, str] = {
    EN: "I'd love to give you a more accurate recommendation — could you help me fill in one more detail?\n\n",
    VI: "Để đưa ra khuyến nghị chính xác hơn, bạn có thể giúp tôi điền thêm một thông tin nữa không?\n\n",
    ZH: "为了给您更准确的建议，能再提供一个信息吗？\n\n",
}

READY_MESSAGE_TEMPLATE: dict[str, str] = {
    EN: "Perfect, I have everything I need{suffix}! Putting together your personalized recommendation now...",
    VI: "Tuyệt vời, tôi đã có đủ thông tin{suffix}! Đang chuẩn bị khuyến nghị cá nhân cho bạn...",
    ZH: "很好，我已获取所需信息{suffix}！正在为您生成个性化建议……",
}

COMPARE_MODEL_PROMPT: dict[str, str] = {
    EN: (
        "That completes your personalized recommendation! "
        "Feel free to change the language or model settings above and restart the session "
        "if you'd like to explore different perspectives or start a new consultation."
    ),
    VI: (
        "Hoàn thành khuyến nghị cá nhân của bạn! "
        "Bạn có thể thay đổi cài đặt ngôn ngữ hoặc mô hình ở trên và khởi động lại phiên "
        "nếu muốn khám phá các góc nhìn khác hoặc bắt đầu tư vấn mới."
    ),
    ZH: (
        "您的个性化建议已完成！"
        "如需探索不同观点或开始新的咨询，"
        "可随时更改上方的语言或模型设置并重启会话。"
    ),
}


RECOMMENDATION_FAILED: dict[str, str] = {
    EN: "Sorry, I was unable to generate a recommendation at this time. Please try again.",
    VI: "Xin lỗi, tôi không thể tạo khuyến nghị lúc này. Vui lòng thử lại.",
    ZH: "抱歉，目前无法生成建议，请稍后重试。",
}


def t(lookup: dict[str, str], lang: str) -> str:
    """Return text for lang, fall back to EN."""
    return lookup.get(lang) or lookup.get(EN, "")


def field_question(field: str, lang: str) -> str:
    return t(FIELD_QUESTIONS.get(field, {}), lang) or f"Please provide your {field.lower()}."


def ready_message(lang: str, unknown_fields: list[str]) -> str:
    suffix = f" (unknown: {', '.join(unknown_fields)})" if unknown_fields else ""
    intro = t(READY_MESSAGE_TEMPLATE, lang).format(suffix=suffix)
    compare = t(COMPARE_MODEL_PROMPT, lang)
    return f"{intro}\n\n{compare}"