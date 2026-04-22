from __future__ import annotations

from finbot.schemas import LanguageCode

EN = LanguageCode.EN.value
VI = LanguageCode.VI.value
ZH = LanguageCode.ZH.value

TASK_PROMPT: dict[str, str] = {
    EN: (
        "Please choose your goal mode:\n"
        "[1] Planning\n"
        "[2] Investment\n"
        "[3] Trading"
    ),
    VI: (
        "Vui lòng chọn chế độ mục tiêu của bạn:\n"
        "[1] Lập kế hoạch\n"
        "[2] Đầu tư\n"
        "[3] Giao dịch"
    ),
    ZH: (
        "请选择您的目标模式：\n"
        "[1] 规划\n"
        "[2] 投资\n"
        "[3] 交易"
    ),
}

FIELD_QUESTIONS: dict[str, dict[str, str]] = {
    "GOAL": {
        EN: "What is your primary financial goal? (e.g. buy a house, retire early, grow wealth)",
        VI: "Mục tiêu tài chính chính của bạn là gì? (ví dụ: mua nhà, nghỉ hưu sớm, tăng trưởng tài sản)",
        ZH: "您的主要财务目标是什么？（例如：购房、提前退休、积累财富）",
    },
    "INCOME_BAND": {
        EN: "What is your approximate annual income?\n[1] Under $60K\n[2] $60K–$120K\n[3] $120K or above",
        VI: "Thu nhập hàng năm của bạn khoảng bao nhiêu?\n[1] Dưới 60K\n[2] 60K–120K\n[3] Từ 120K trở lên",
        ZH: "您的年收入大约是多少？\n[1] 6万以下\n[2] 6万至12万\n[3] 12万及以上",
    },
    "CAPITAL_RANGE": {
        EN: "How much capital do you have available to invest?\n[1] Under $10K\n[2] $10K–$50K\n[3] $50K–$250K\n[4] $250K or above",
        VI: "Bạn có bao nhiêu vốn để đầu tư?\n[1] Dưới 10K\n[2] 10K–50K\n[3] 50K–250K\n[4] Từ 250K trở lên",
        ZH: "您有多少可用投资资金？\n[1] 1万以下\n[2] 1万至5万\n[3] 5万至25万\n[4] 25万及以上",
    },
    "TIME_HORIZON": {
        EN: "What is your investment time horizon?\n[1] Short-term (under 1 year)\n[2] Medium-term (1–5 years)\n[3] Long-term (5 years or more)",
        VI: "Kỳ hạn đầu tư của bạn là bao lâu?\n[1] Ngắn hạn (dưới 1 năm)\n[2] Trung hạn (1–5 năm)\n[3] Dài hạn (trên 5 năm)",
        ZH: "您的投资期限是多久？\n[1] 短期（1年以内）\n[2] 中期（1至5年）\n[3] 长期（5年以上）",
    },
    "RISK_TOLERANCE": {
        EN: "How would you describe your risk tolerance?\n[1] Low — prefer stability\n[2] Medium — accept some risk\n[3] High — tolerate significant loss for high growth",
        VI: "Mức độ chịu đựng rủi ro của bạn như thế nào?\n[1] Thấp — ưu tiên ổn định\n[2] Trung bình — chấp nhận rủi ro vừa phải\n[3] Cao — chấp nhận rủi ro cao",
        ZH: "您如何描述您的风险承受能力？\n[1] 低 — 稳定优先\n[2] 中 — 接受适度风险\n[3] 高 — 可承受较大损失",
    },
}

CLARIFY_SUFFIX: dict[str, str] = {
    EN: "\n\nIf you're not sure, type 'skip' and I will mark this as unknown and continue.",
    VI: "\n\nNếu bạn không chắc, gõ 'bỏ qua' để tôi đánh dấu là không rõ và tiếp tục.",
    ZH: "\n\n如果您不确定，请输入\"跳过\"，我将标记为未知并继续。",
}

TOO_MANY_UNKNOWN_PREFIX: dict[str, str] = {
    EN: "Too many fields are unknown. Let's try one more:\n\n",
    VI: "Còn quá nhiều thông tin chưa rõ. Hãy thử thêm:\n\n",
    ZH: "还有太多未知项，再回答一个：\n\n",
}

READY_MESSAGE_TEMPLATE: dict[str, str] = {
    EN: "I have collected your profile{suffix}. Generating your recommendation now...",
    VI: "Tôi đã thu thập đủ thông tin{suffix}. Đang tạo khuyến nghị cho bạn...",
    ZH: "我已收集您的信息{suffix}，正在为您生成建议……",
}


def t(lookup: dict[str, str], lang: str) -> str:
    """Return text for lang, fall back to EN."""
    return lookup.get(lang) or lookup.get(EN, "")


def field_question(field: str, lang: str) -> str:
    return t(FIELD_QUESTIONS.get(field, {}), lang) or f"Please provide your {field.lower()}."


def ready_message(lang: str, unknown_fields: list[str]) -> str:
    suffix = f" (unknown: {', '.join(unknown_fields)})" if unknown_fields else ""
    return t(READY_MESSAGE_TEMPLATE, lang).format(suffix=suffix)