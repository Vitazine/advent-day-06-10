"""Validate an untrusted model candidate. No writes, inference or financial actions.

This checks structure, not factual faithfulness to the user's utterance.
The caller must present a preview, resolve clarification and handle saving.
"""
import json
import re
from datetime import date
from decimal import Decimal

CATEGORIES={'groceries','cafes','transport','housing','utilities','health','shopping','entertainment','education','other'}
FIELDS={'status','amount','currency','category','date','payment_method','clarification_reason'}
REASONS={'missing_amount','ambiguous_amount','ambiguous_currency','ambiguous_date','multiple_expenses','missing_context'}

def _iso(value):
    if not isinstance(value,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',value):
        raise ValueError('Invalid date format')
    return date.fromisoformat(value)

def _pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ValueError('Duplicate JSON key')
        result[key]=value
    return result

def _constant(value):
    raise ValueError('Nonfinite JSON value')

def parse_model_response(raw: str, current_date: str) -> dict:
    """Return a validated candidate or raise ValueError; never auto-save it."""
    today=_iso(current_date)
    if not isinstance(raw,str) or len(raw)>16000: raise ValueError('Invalid response length or type')
    try: record=json.loads(raw,object_pairs_hook=_pairs,parse_constant=_constant)
    except (TypeError,RecursionError) as exc: raise ValueError('Invalid JSON') from exc
    if not isinstance(record,dict) or set(record)!=FIELDS: raise ValueError('Invalid fields')
    for value in record.values():
        if value is not None and not isinstance(value,str): raise ValueError('Fields must be strings or null')
    if record['status'] not in {'ok','ignored','needs_clarification'}: raise ValueError('Invalid status')
    if record['currency'] not in {'RUB','USD','EUR','PHP',None}: raise ValueError('Invalid currency')
    if record['category'] not in CATEGORIES|{None}: raise ValueError('Invalid category')
    if record['payment_method'] not in {'cash','card','transfer',None}: raise ValueError('Invalid payment method')
    amount=record['amount']
    if amount is not None and (not re.fullmatch(r'(?:0|[1-9]\d*)\.\d{2}',amount) or Decimal(amount)<=0):
        raise ValueError('Invalid amount')
    if record['date'] is not None and _iso(record['date'])>today: raise ValueError('Future expense')
    reason=record['clarification_reason']
    if record['status']=='ignored':
        if any(value is not None for key,value in record.items() if key!='status'): raise ValueError('Ignored must have null fields')
    elif record['status']=='ok':
        if reason is not None or any(record[k] is None for k in ('amount','currency','category','date')):
            raise ValueError('Incomplete expense')
    else:
        if reason not in REASONS: raise ValueError('Invalid clarification reason')
        if reason in {'multiple_expenses','missing_context'}:
            if any(record[k] is not None for k in ('amount','currency','category','date','payment_method')): raise ValueError('Cannot guess expense fields')
        else:
            missing={'missing_amount':'amount','ambiguous_amount':'amount','ambiguous_currency':'currency','ambiguous_date':'date'}[reason]
            if record[missing] is not None: raise ValueError('Ambiguous field must be null')
    return record
