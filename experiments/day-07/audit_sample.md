# Случайные 30 примеров v3


Seed=817. Результаты чтения — audit_review.md.


## train-252


{"text": "Оплатила интернет 900 рублей телефоном.", "current_date": "2026-11-01"}

```json
{
  "status": "ok",
  "amount": "900.00",
  "currency": "RUB",
  "category": "utilities",
  "date": "2026-11-01",
  "payment_method": null,
  "clarification_reason": null
}
```

## train-340


{"text": "Ботинки 5000 до скидки, фактически заплатил 3500.", "current_date": "2026-11-01"}

```json
{
  "status": "ok",
  "amount": "3500.00",
  "currency": "RUB",
  "category": "shopping",
  "date": "2026-11-01",
  "payment_method": null,
  "clarification_reason": null
}
```

## train-014


{"text": "Ушло 5205 руб. на куриную грудку, сегодня, картой.", "current_date": "2026-03-01"}

```json
{
  "status": "ok",
  "amount": "5205.00",
  "currency": "RUB",
  "category": "groceries",
  "date": "2026-03-01",
  "payment_method": "card",
  "clarification_reason": null
}
```

## train-248


{"text": "Купил продукты на 1200, скидка уже учтена.", "current_date": "2026-11-01"}

```json
{
  "status": "ok",
  "amount": "1200.00",
  "currency": "RUB",
  "category": "groceries",
  "date": "2026-11-01",
  "payment_method": null,
  "clarification_reason": null
}
```

## train-contrast-direction-05-2


{"text": "Собираюсь заплатить 2126 за приём невролога, ещё не платил.", "current_date": "2026-01-01"}

```json
{
  "status": "ignored",
  "amount": null,
  "currency": null,
  "category": null,
  "date": null,
  "payment_method": null,
  "clarification_reason": null
}
```

## train-contrast-precision-04-1


{"text": "Заплатил за абонентскую плату за связь 648, не по карте, а наличными.", "current_date": "2026-01-01"}

```json
{
  "status": "ok",
  "amount": "648.00",
  "currency": "RUB",
  "category": "utilities",
  "date": "2026-01-01",
  "payment_method": "cash",
  "clarification_reason": null
}
```

## train-319


{"text": "заправка 1700 по карте", "current_date": "2026-11-01"}

```json
{
  "status": "ok",
  "amount": "1700.00",
  "currency": "RUB",
  "category": "transport",
  "date": "2026-11-01",
  "payment_method": "card",
  "clarification_reason": null
}
```

## train-contrast-context-04-0


{"text": "Создай новую запись: за абонентскую плату за связь оплачено 611.", "current_date": "2026-11-01"}

```json
{
  "status": "ok",
  "amount": "611.00",
  "currency": "RUB",
  "category": "utilities",
  "date": "2026-11-01",
  "payment_method": null,
  "clarification_reason": null
}
```

## train-contrast-direction-02-0


{"text": "Я заплатил 186 за проезд на пароме.", "current_date": "2024-03-01"}

```json
{
  "status": "ok",
  "amount": "186.00",
  "currency": "RUB",
  "category": "transport",
  "date": "2024-03-01",
  "payment_method": null,
  "clarification_reason": null
}
```

## train-contrast-currency-08-1


{"text": "Оплатил урок немецкого: 29$.", "current_date": "2026-01-01"}

```json
{
  "status": "ok",
  "amount": "29.00",
  "currency": "USD",
  "category": "education",
  "date": "2026-01-01",
  "payment_method": null,
  "clarification_reason": null
}
```

## train-006


{"text": "Запиши расход на рис и яйца: 60 долларов США, позавчера, наличными.", "current_date": "2026-10-31"}

```json
{
  "status": "ok",
  "amount": "60.00",
  "currency": "USD",
  "category": "groceries",
  "date": "2026-10-29",
  "payment_method": "cash",
  "clarification_reason": null
}
```

## train-contrast-date-05-1


{"text": "Вчера потратил 2237 на приём невролога.", "current_date": "2026-11-01"}

```json
{
  "status": "ok",
  "amount": "2237.00",
  "currency": "RUB",
  "category": "health",
  "date": "2026-10-31",
  "payment_method": null,
  "clarification_reason": null
}
```

## train-contrast-context-08-0


{"text": "Создай новую запись: за урок немецкого оплачено 1274.", "current_date": "2026-11-01"}

```json
{
  "status": "ok",
  "amount": "1274.00",
  "currency": "RUB",
  "category": "education",
  "date": "2026-11-01",
  "payment_method": null,
  "clarification_reason": null
}
```

## train-contrast-amount-07-0


{"text": "За билет на выставку заплатил 707; на счёте осталось 5707.", "current_date": "2026-01-01"}

```json
{
  "status": "ok",
  "amount": "707.00",
  "currency": "RUB",
  "category": "entertainment",
  "date": "2026-01-01",
  "payment_method": null,
  "clarification_reason": null
}
```

## train-contrast-direction-09-3


{"text": "Я не заплатил 320 за хранение личной посылки: покупку отменил.", "current_date": "2026-01-01"}

```json
{
  "status": "ignored",
  "amount": null,
  "currency": null,
  "category": null,
  "date": null,
  "payment_method": null,
  "clarification_reason": null
}
```

## train-231


{"text": "Оплатила копирование документов, сумма 710 руб., позавчера, наличными.", "current_date": "2024-03-01"}

```json
{
  "status": "ok",
  "amount": "710.00",
  "currency": "RUB",
  "category": "other",
  "date": "2024-02-28",
  "payment_method": "cash",
  "clarification_reason": null
}
```

## train-contrast-context-07-3


{"text": "За билет на выставку отдал 781, а отдельно за другой расход 50.", "current_date": "2026-06-01"}

```json
{
  "status": "needs_clarification",
  "amount": null,
  "currency": null,
  "category": null,
  "date": null,
  "payment_method": null,
  "clarification_reason": "multiple_expenses"
}
```

## train-contrast-date-02-1


{"text": "Вчера потратил 297 на проезд на пароме.", "current_date": "2026-01-01"}

```json
{
  "status": "ok",
  "amount": "297.00",
  "currency": "RUB",
  "category": "transport",
  "date": "2025-12-31",
  "payment_method": null,
  "clarification_reason": null
}
```

## train-contrast-precision-04-0


{"text": "Заплатил за абонентскую плату за связь 648 по карте.", "current_date": "2026-01-01"}

```json
{
  "status": "ok",
  "amount": "648.00",
  "currency": "RUB",
  "category": "utilities",
  "date": "2026-01-01",
  "payment_method": "card",
  "clarification_reason": null
}
```

## train-150


{"text": "Потратил 83,99 евро на кроссовки, сегодня, банковским переводом продавцу.", "current_date": "2026-10-31"}

```json
{
  "status": "ok",
  "amount": "83.99",
  "currency": "EUR",
  "category": "shopping",
  "date": "2026-10-31",
  "payment_method": "transfer",
  "clarification_reason": null
}
```

## train-contrast-currency-01-2


{"text": "Оплатил сэндвич в киоске: 8 долларов.", "current_date": "2024-03-01"}

```json
{
  "status": "ok",
  "amount": "8.00",
  "currency": "USD",
  "category": "cafes",
  "date": "2024-03-01",
  "payment_method": null,
  "clarification_reason": null
}
```

## train-contrast-date-09-2


{"text": "Потратил 431 на хранение личной посылки 2026-10-31.", "current_date": "2026-11-01"}

```json
{
  "status": "ok",
  "amount": "431.00",
  "currency": "RUB",
  "category": "other",
  "date": "2026-10-31",
  "payment_method": null,
  "clarification_reason": null
}
```

## train-037


{"text": "Ушло 3902,99 руб. на ланч в столовой, сегодня, наличными.", "current_date": "2026-11-01"}

```json
{
  "status": "ok",
  "amount": "3902.99",
  "currency": "RUB",
  "category": "cafes",
  "date": "2026-11-01",
  "payment_method": "cash",
  "clarification_reason": null
}
```

## train-contrast-context-03-0


{"text": "Создай новую запись: за замену розетки дома оплачено 1448.", "current_date": "2026-06-01"}

```json
{
  "status": "ok",
  "amount": "1448.00",
  "currency": "RUB",
  "category": "housing",
  "date": "2026-06-01",
  "payment_method": null,
  "clarification_reason": null
}
```

## train-contrast-precision-08-0


{"text": "Заплатил за урок немецкого 1311 по карте.", "current_date": "2026-01-01"}

```json
{
  "status": "ok",
  "amount": "1311.00",
  "currency": "RUB",
  "category": "education",
  "date": "2026-01-01",
  "payment_method": "card",
  "clarification_reason": null
}
```

## train-contrast-amount-05-2


{"text": "За приём невролога заплатил не 2200, а 2270.", "current_date": "2026-06-01"}

```json
{
  "status": "ok",
  "amount": "2270.00",
  "currency": "RUB",
  "category": "health",
  "date": "2026-06-01",
  "payment_method": null,
  "clarification_reason": null
}
```

## train-contrast-context-00-0


{"text": "Создай новую запись: за гречку для дома оплачено 328.", "current_date": "2026-11-01"}

```json
{
  "status": "ok",
  "amount": "328.00",
  "currency": "RUB",
  "category": "groceries",
  "date": "2026-11-01",
  "payment_method": null,
  "clarification_reason": null
}
```

## train-292


{"text": "Куплю куртку за 6000, когда получу зарплату.", "current_date": "2026-11-01"}

```json
{
  "status": "ignored",
  "amount": null,
  "currency": null,
  "category": null,
  "date": null,
  "payment_method": null,
  "clarification_reason": null
}
```

## train-contrast-currency-08-0


{"text": "Оплатил урок немецкого: 29, без указания валюты.", "current_date": "2026-01-01"}

```json
{
  "status": "ok",
  "amount": "29.00",
  "currency": "RUB",
  "category": "education",
  "date": "2026-01-01",
  "payment_method": null,
  "clarification_reason": null
}
```

## train-contrast-context-08-1


{"text": "В существующей записи за урок немецкого исправь сумму на 1274.", "current_date": "2026-11-01"}

```json
{
  "status": "needs_clarification",
  "amount": null,
  "currency": null,
  "category": null,
  "date": null,
  "payment_method": null,
  "clarification_reason": "missing_context"
}
```
