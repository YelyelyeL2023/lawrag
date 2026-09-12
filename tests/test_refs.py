"""Cross-reference resolution — the citation graph's correctness."""

from __future__ import annotations

from lawrag.ingest.acts import find_act_mentions, resolve_act
from lawrag.ingest.refs import extract_definitions, extract_refs
from lawrag.models import Lang


def _targets(text, lang, self_act="koap"):
    return {r.dst_node_id for r in extract_refs("koap/st100", self_act, text, lang)}


def test_russian_reference_takes_the_act_named_after_it():
    text = ("в порядке, установленном пунктом 2 статьи 278 настоящего Кодекса, "
            "с учетом статьи 76 Налогового кодекса")
    targets = _targets(text, Lang.RU)
    assert "koap/st278" in targets
    # The regression this pins: article 76 inheriting "настоящего Кодекса" from
    # the previous citation in the same sentence.
    assert "nk/st76" in targets
    assert "koap/st76" not in targets


def test_genitive_plural_statej_is_matched():
    """'статей' has no soft sign; matching only 'стать' drops every list cite."""
    targets = _targets("статей 5, 6 и 7 Предпринимательского кодекса", Lang.RU)
    assert targets == {"pk/st5", "pk/st6", "pk/st7"}


def test_kazakh_consonant_alternation_bap_to_bab():
    """'76-бабына' is article 76; matching only '-бап' loses oblique cases."""
    text = "Салық кодексінің 76-бабына сәйкес, осы Кодекстің 278-бабы қолданылады"
    targets = _targets(text, Lang.KK)
    assert "nk/st76" in targets
    assert "koap/st278" in targets


def test_short_kazakh_act_aliases_need_the_word_kodeks():
    """'су' (water) must not match inside 'бизнесу'."""
    assert find_act_mentions("нужно ли малому бизнесу встать на учет") == []
    assert resolve_act("Водного кодекса") == "vk"


def test_self_references_are_dropped():
    targets = _targets("в соответствии со статьей 100 настоящего Кодекса", Lang.RU)
    assert "koap/st100" not in targets


def test_definition_extraction():
    text = ("1) субъект малого предпринимательства — индивидуальный предприниматель;\n"
            "2) уполномоченный орган — центральный исполнительный орган.")
    defs = dict(extract_definitions("pk/st24", "pk", text))
    assert "субъект малого предпринимательства" in defs
    assert defs["уполномоченный орган"].startswith("центральный")
