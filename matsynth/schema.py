"""Typed schema untuk record Kononova + assertion per record (langkah 1).

Skema mengikuti dataset_typing.py resmi dari CederGroupHub/text-mined-synthesis_public.
Field bertanda "New field!" di typing resmi (material_name, phase, is_acronym) baru ada
di rilis 2020; file 2019 yang kita pakai boleh tidak memilikinya, tapi kalau ada
tipenya tetap di-assert.
"""

from dataclasses import dataclass


class SchemaError(AssertionError):
    pass


def _req(cond: bool, msg: str, idx: int) -> None:
    if not cond:
        raise SchemaError(f"record[{idx}]: {msg}")


def _assert_str_list(x, name: str, idx: int) -> None:
    _req(isinstance(x, list) and all(isinstance(s, str) for s in x),
         f"{name} harus List[str], dapat {type(x).__name__}", idx)


def assert_material_schema(mat, name: str, idx: int) -> None:
    _req(isinstance(mat, dict), f"{name} harus dict, dapat {type(mat).__name__}", idx)
    _req(isinstance(mat.get("material_formula"), str), f"{name}.material_formula harus str", idx)
    _req(isinstance(mat.get("material_string"), str), f"{name}.material_string harus str", idx)

    comp = mat.get("composition")
    _req(isinstance(comp, list), f"{name}.composition harus list", idx)
    for j, c in enumerate(comp):
        _req(isinstance(c, dict), f"{name}.composition[{j}] harus dict", idx)
        _req(isinstance(c.get("formula"), str), f"{name}.composition[{j}].formula harus str", idx)
        _req(isinstance(c.get("amount"), str), f"{name}.composition[{j}].amount harus str", idx)
        els = c.get("elements")
        _req(isinstance(els, dict)
             and all(isinstance(k, str) and isinstance(v, str) for k, v in els.items()),
             f"{name}.composition[{j}].elements harus Dict[str, str]", idx)

    _req(isinstance(mat.get("amounts_vars"), dict), f"{name}.amounts_vars harus dict", idx)
    _req(isinstance(mat.get("elements_vars"), dict), f"{name}.elements_vars harus dict", idx)
    _req(isinstance(mat.get("additives"), list), f"{name}.additives harus list", idx)

    # Field khas rilis 2019 — mapping Materials Project bawaan dataset.
    if "mp_id" in mat:
        _req(mat["mp_id"] is None or isinstance(mat["mp_id"], str),
             f"{name}.mp_id harus Optional[str]", idx)

    # Field opsional (rilis 2020) — assert tipe hanya jika ada.
    if "material_name" in mat:
        _req(isinstance(mat["material_name"], str), f"{name}.material_name harus str", idx)
    if "is_acronym" in mat:
        _req(isinstance(mat["is_acronym"], bool), f"{name}.is_acronym harus bool", idx)
    if "phase" in mat:
        _req(mat["phase"] is None or isinstance(mat["phase"], str),
             f"{name}.phase harus Optional[str]", idx)
    if "oxygen_deficiency" in mat:
        v = mat["oxygen_deficiency"]
        _req(v is None or isinstance(v, (str, bool)),
             f"{name}.oxygen_deficiency harus Optional[str|bool]", idx)


def assert_record_schema(rec, idx: int) -> None:
    """Assert satu record mentah Kononova terhadap skema ReactionEntry."""
    _req(isinstance(rec, dict), f"record harus dict, dapat {type(rec).__name__}", idx)
    _req(isinstance(rec.get("doi"), str), "doi harus str", idx)
    # synthesis_type baru ada di rilis 2020; file 2019 seluruhnya solid-state.
    if "synthesis_type" in rec:
        _req(isinstance(rec["synthesis_type"], str), "synthesis_type harus str", idx)
    _req(isinstance(rec.get("reaction_string"), str), "reaction_string harus str", idx)
    _assert_str_list(rec.get("targets_string"), "targets_string", idx)

    reaction = rec.get("reaction")
    _req(isinstance(reaction, dict), "reaction harus dict", idx)
    for side in ("left_side", "right_side"):
        parts = reaction.get(side)
        _req(isinstance(parts, list), f"reaction.{side} harus list", idx)
        for j, p in enumerate(parts):
            _req(isinstance(p, dict)
                 and isinstance(p.get("amount"), str)
                 and isinstance(p.get("material"), str),
                 f"reaction.{side}[{j}] harus {{amount: str, material: str}}", idx)

    assert_material_schema(rec.get("target"), "target", idx)
    precursors = rec.get("precursors")
    _req(isinstance(precursors, list), "precursors harus list", idx)
    for j, p in enumerate(precursors):
        assert_material_schema(p, f"precursors[{j}]", idx)

    ops = rec.get("operations")
    _req(isinstance(ops, list), "operations harus list", idx)
    for j, op in enumerate(ops):
        _req(isinstance(op, dict), f"operations[{j}] harus dict", idx)
        _req(isinstance(op.get("type"), str), f"operations[{j}].type harus str", idx)
        # 2019 memakai key "string", 2020 memakai "token" — terima salah satu.
        _req(isinstance(op.get("token", op.get("string")), str),
             f"operations[{j}].token/string harus str", idx)
        # 2019: conditions bisa null untuk operasi tanpa kondisi terekstrak.
        _req(op.get("conditions") is None or isinstance(op["conditions"], dict),
             f"operations[{j}].conditions harus Optional[dict]", idx)


@dataclass(frozen=True)
class SynthesisRecord:
    """Record ternormalisasi: satu target tersintesis + precursor set-nya."""
    doi: str
    target_formula: str                 # pymatgen reduced formula, mis. "Li6PS5Cl"
    precursor_formulas: tuple[str, ...]  # reduced formula precursor, sorted
    synthesis_type: str
    target_mp_id: str | None = None     # mapping MP bawaan dataset (bila ada)

    def __post_init__(self):
        assert isinstance(self.doi, str)
        assert isinstance(self.target_formula, str) and self.target_formula
        assert isinstance(self.precursor_formulas, tuple)
        assert all(isinstance(p, str) for p in self.precursor_formulas)
        assert isinstance(self.synthesis_type, str)
        assert self.target_mp_id is None or isinstance(self.target_mp_id, str)
