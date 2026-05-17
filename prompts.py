"""
Prompt registry backed by config/prompts.yml.

The public repository keeps prompt text outside the Python runtime so users can
adapt extraction behavior without editing application code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from app_settings import DEFAULT_PROMPTS_PATH, get_prompts_path


def _load_prompts() -> dict[str, Any]:
    prompts_path: Path = get_prompts_path()
    if not prompts_path.exists():
        prompts_path = DEFAULT_PROMPTS_PATH

    with prompts_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return {}
    return data


PROMPTS = _load_prompts()


def _get_prompt(key: str, fallback: str) -> str:
    value = PROMPTS.get(key, fallback)
    return value.strip() if isinstance(value, str) else fallback


SIGNATURE_PROMPT = _get_prompt(
    "signature_prompt",
    '{"receiver_signature": "yes"}',
)

BILL_OF_LADING_PROMPT = _get_prompt(
    "bill_of_lading_prompt",
    '{"bill_no": "null", "date": "null"}',
)

CUSTOMER_ORDER_INFO_PROMPT = _get_prompt(
    "customer_order_info_prompt",
    '{"total_order_quantity": "null"}',
)

STAMP_LATEST_PROMPT = _get_prompt(
    "stamp_latest_prompt",
    '{"stamp_exist": "no"}',
)

DELIVERY_RECEIPT_PROMPT = _get_prompt(
    "delivery_receipt_prompt",
    '{"pod_date": "null"}',
)

RECEIPT_DATE_PROMPT = _get_prompt(
    "receipt_date_prompt",
    '{"pod_date": "null"}',
)

RECEIPT_SIGNATURE_PROMPT = _get_prompt(
    "receipt_signature_prompt",
    '{"pod_sign": "no"}',
)

RECEIPT_TOTAL_RECEIVED_PROMPT = _get_prompt(
    "receipt_total_received_prompt",
    '{"total_received": "null"}',
)

RECEIPT_DAMAGE_PROMPT = _get_prompt(
    "receipt_damage_prompt",
    '{"damage": "null"}',
)

RECEIPT_REFUSED_PROMPT = _get_prompt(
    "receipt_refused_prompt",
    '{"refused": "null"}',
)

RECEIPT_CUSTOMER_ORDER_NUMBER_PROMPT = _get_prompt(
    "receipt_customer_order_number_prompt",
    '{"customer_order_num": "null"}',
)
