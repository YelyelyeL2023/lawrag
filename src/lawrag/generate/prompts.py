"""Grounded prompts.

Three constraints are non-negotiable for a compliance tool whose users are not
lawyers and cannot spot a fabricated citation:

1. Nothing may be asserted that is not in the supplied sources.
2. Every assertion carries a pinpoint citation and a verbatim quote — the quote
   is what the verification layer checks mechanically afterwards.
3. When the sources do not answer the question, the model abstains. "I could not
   find a provision on this" is a correct answer; a plausible invented one is a
   fine or a prosecution.

The Kazakh prompt additionally forbids composing legal Kazakh. The model wraps
verbatim corpus text; the statutory wording is itself the legal fact, and a
paraphrase of it is not the law.
"""

from __future__ import annotations

from datetime import date

from ..models import Lang

_SCHEMA = """{
  "answer": "<краткий вывод простым языком, 2-5 предложений>",
  "citations": [
    {"source": <номер ИСТОЧНИКА>, "act": "<название акта>", "article": "<номер статьи>",
     "point": "<номер пункта или null>", "quote": "<дословная цитата из текста источника>"}
  ],
  "confidence": "high" | "medium" | "low",
  "abstain": false,
  "abstain_reason": "",
  "escalate": false
}"""

_RU = """Ты — справочная система по законодательству Республики Казахстан. Тобой пользуются сотрудники без юридического образования, которые будут действовать на основании твоего ответа.

ВОПРОС:
{question}

ДАТА, НА КОТОРУЮ НУЖНО ОТВЕТИТЬ: {as_of}

НИЖЕ — ЕДИНСТВЕННЫЕ ИСТОЧНИКИ, КОТОРЫМИ ТЫ РАСПОЛАГАЕШЬ:
{packet}

ПРАВИЛА:
1. Используй ТОЛЬКО текст источников выше. Не используй свои знания о законодательстве.
2. Если источники не содержат ответа — поставь "abstain": true, объясни в "abstain_reason", чего не хватает, и поставь "escalate": true. Не догадывайся.
3. Каждое утверждение подкрепи ссылкой на конкретный пункт статьи и ДОСЛОВНОЙ цитатой из текста источника. Цитата должна дословно встречаться в источнике — она проверяется автоматически.
4. Если источники противоречат друг другу, укажи это и назови акт с большей юридической силой.
5. Если источник помечен как утративший силу или прежняя редакция — прямо скажи об этом.
6. Не давай юридических советов и не оценивай перспективы дела. Излагай, что говорит норма.
7. Если вопрос требует оценки конкретной ситуации или толкования — поставь "escalate": true.

Ответь СТРОГО одним объектом JSON, без markdown и без пояснений:
{schema}"""

_KK = """Сен — Қазақстан Республикасының заңнамасы бойынша анықтамалық жүйесісің. Сені заң білімі жоқ қызметкерлер пайдаланады және сенің жауабың негізінде әрекет етеді.

СҰРАҚ:
{question}

ЖАУАП БЕРІЛЕТІН КҮН: {as_of}

ТӨМЕНДЕ — СЕНДЕ БАР ЖАЛҒЫЗ ДЕРЕККӨЗДЕР:
{packet}

ЕРЕЖЕЛЕР:
1. ТЕК жоғарыдағы дереккөздер мәтінін пайдалан. Заңнама туралы өз біліміңді пайдаланба.
2. Дереккөздерде жауап болмаса — "abstain": true қой, "abstain_reason" ішінде нені таба алмағаныңды жаз, "escalate": true қой. Болжама.
3. Әрбір тұжырымды нақты баптың тармағына сілтемемен және дереккөзден СӨЗБЕ-СӨЗ дәйексөзбен растап отыр. Дәйексөз автоматты түрде тексеріледі.
4. Заң нормасының қазақша мәтінін өзің құрастырма — оны дереккөзден сөзбе-сөз көшір. Түсіндірмені ғана өз сөзіңмен жаз.
5. Дереккөз күші жойылған немесе бұрынғы редакция деп белгіленсе — оны тікелей айт.
6. Заң кеңесін берме. Норма не дейтінін ғана баянда.
7. Сұрақ нақты жағдайды бағалауды талап етсе — "escalate": true қой.

ТЕК бір JSON объектісімен жауап бер, markdown-сыз:
{schema}"""


def build_prompt(question: str, packet: str, lang: Lang, as_of: date) -> str:
    template = _KK if lang is Lang.KK else _RU
    return template.format(question=question.strip(), packet=packet,
                           as_of=as_of.isoformat(), schema=_SCHEMA)


NO_SOURCES = {
    Lang.RU: ("В доступных нормативных актах не найдено положений, отвечающих на этот "
              "вопрос. Обратитесь к юристу."),
    Lang.KK: ("Қолжетімді нормативтік актілерде бұл сұраққа жауап беретін ережелер "
              "табылмады. Заңгерге жүгініңіз."),
}

NO_MODEL = {
    Lang.RU: ("Языковая модель не подключена, поэтому ответ не сформирован. "
              "Ниже перечислены нормы, найденные по запросу, — их можно прочитать "
              "самостоятельно."),
    Lang.KK: ("Тілдік модель қосылмаған, сондықтан жауап құрастырылмады. Төменде "
              "сұрау бойынша табылған нормалар келтірілген."),
}

DISCLAIMER = {
    Lang.RU: ("Это справочная информация из текстов нормативных актов, а не юридическая "
              "консультация. Проверьте актуальность нормы и при необходимости обратитесь "
              "к юристу."),
    Lang.KK: ("Бұл — нормативтік актілер мәтінінен алынған анықтамалық ақпарат, заң "
              "кеңесі емес. Норманың өзектілігін тексеріңіз, қажет болса заңгерге "
              "жүгініңіз."),
}


def no_text_on_date(lang: Lang, as_of, unavailable: list[dict]) -> str:
    """The named article exists, but no wording of it is stored for that date."""
    lines = []
    for u in unavailable:
        if lang is Lang.KK:
            lines.append(f"— {u['node_id']}: мәтін {u['earliest']} бастап сақталған")
        else:
            lines.append(f"— {u['node_id']}: текст хранится с {u['earliest']}")
    if lang is Lang.KK:
        head = (f"Сұралған бап корпуста бар, бірақ оның {as_of.isoformat()} күнгі редакциясы "
                "жүктелмеген. Бүгінгі мәтінді сол күнге қолданылған деп келтіруге болмайды. "
                "Заңгерге жүгініңіз немесе сол күнге дейінгі редакцияны жүктеңіз.")
    else:
        head = (f"Статья есть в корпусе, но её редакция на {as_of.isoformat()} не импортирована. "
                "Сегодняшний текст нельзя выдавать за действовавший в тот день. Обратитесь к "
                "юристу или загрузите редакцию на эту дату.")
    return head + "\n\n" + "\n".join(lines)
