"""Resolusi formula -> struktur pymatgen via OPTIMADE (tanpa API key).

Provider: Materials Project OPTIMADE lalu OQMD OPTIMADE. Hasil (hit maupun miss)
di-cache ke disk supaya query tidak diulang. Entry disordered / mengandung vacancy
dilewati karena MACE butuh struktur ordered.

OQMD juga dipakai untuk mining negatif skema CSLLM: entry dengan
_oqmd_icsd_id == null (tak pernah match struktur eksperimen ICSD) dan
_oqmd_stability kecil (dekat hull) = "hipotetis stabil-tapi-tak-tersintesis".
"""

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
from pymatgen.core import Composition, Lattice, Structure

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "structures"

MP_BASE = "https://optimade.materialsproject.org/v1"
OQMD_BASE = "https://oqmd.org/optimade"

STRUCT_FIELDS = ("lattice_vectors,cartesian_site_positions,species,species_at_sites,"
                 "chemical_formula_reduced,nsites")
OQMD_EXTRA_FIELDS = ",_oqmd_stability,_oqmd_delta_e,_oqmd_icsd_id,_oqmd_entry_id"

_REQUEST_GAP_S = 0.35
_last_request_t = 0.0


def optimade_reduced_formula(formula: str) -> str:
    """Format chemical_formula_reduced OPTIMADE: alfabetis, proporsi integer, '1' dihilangkan."""
    comp = Composition(formula)
    int_formula, _factor = comp.get_integer_formula_and_factor()
    icomp = Composition(int_formula)
    amounts = {el.symbol: int(round(amt)) for el, amt in icomp.items()}
    assert all(abs(icomp[el] - amounts[el.symbol]) < 1e-6 for el in icomp), \
        f"proporsi non-integer untuk {formula}"
    g = math.gcd(*amounts.values()) if len(amounts) > 1 else next(iter(amounts.values()))
    parts = []
    for el in sorted(amounts):
        n = amounts[el] // g
        parts.append(el if n == 1 else f"{el}{n}")
    return "".join(parts)


def _get_json(url: str, params: dict, tries: int = 5) -> dict:
    """GET dengan rate-limit sederhana + retry (backoff untuk DNS/HTTP transien)."""
    global _last_request_t
    for attempt in range(tries):
        wait = _REQUEST_GAP_S - (time.monotonic() - _last_request_t)
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(url, params=params, timeout=60)
            _last_request_t = time.monotonic()
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                time.sleep(min(3.0 * 2 ** attempt, 30.0))
                continue
            raise RuntimeError(f"HTTP {r.status_code} dari {url}: {r.text[:300]}")
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(min(3.0 * 2 ** attempt, 30.0))
    raise RuntimeError("unreachable")


def query_optimade(base: str, flt: str, response_fields: str, page_limit: int = 20) -> list[dict]:
    payload = _get_json(f"{base}/structures", {
        "filter": flt,
        "response_fields": response_fields,
        "page_limit": page_limit,
    })
    if payload.get("errors"):
        raise RuntimeError(f"OPTIMADE error dari {base}: {payload['errors']}")
    data = payload.get("data", [])
    assert isinstance(data, list)
    return data


def entry_to_structure(entry: dict) -> Structure | None:
    """Entry OPTIMADE -> Structure ordered, atau None kalau disordered/vacancy."""
    a = entry["attributes"]
    species_by_name = {s["name"]: s for s in a["species"]}
    symbols: list[str] = []
    for name in a["species_at_sites"]:
        sp = species_by_name[name]
        syms = sp["chemical_symbols"]
        concs = sp["concentration"]
        if len(syms) != 1 or syms[0] == "vacancy" or abs(concs[0] - 1.0) > 1e-6:
            return None
        symbols.append(syms[0])
    lattice = a["lattice_vectors"]
    coords = a["cartesian_site_positions"]
    assert len(symbols) == len(coords) == a["nsites"], "nsites tidak konsisten"
    assert len(lattice) == 3 and all(len(row) == 3 for row in lattice)
    s = Structure(Lattice(lattice), symbols, coords, coords_are_cartesian=True)
    assert len(s) == a["nsites"]
    return s


@dataclass
class ResolvedStructure:
    formula: str          # reduced formula pymatgen dari struktur
    provider: str         # "mp" | "oqmd"
    source_id: str
    structure: Structure
    meta: dict = field(default_factory=dict)

    @property
    def material_id(self) -> str:
        return f"{self.provider}:{self.source_id}"


def _cache_path(key: str, cache_dir: Path) -> Path:
    return cache_dir / f"{key}.json"


def resolve_by_mp_id(mp_id: str, cache_dir: Path = CACHE_DIR) -> ResolvedStructure | None:
    """Ambil struktur langsung dari MP OPTIMADE berdasarkan id (mis. "mp-19538")."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = _cache_path(f"mpid-{mp_id}", cache_dir)
    if cpath.exists():
        cached = json.loads(cpath.read_text(encoding="utf-8"))
        if cached["status"] == "miss":
            return None
        return ResolvedStructure(
            formula=cached["formula"], provider="mp", source_id=cached["source_id"],
            structure=Structure.from_dict(cached["structure"]),
            meta=cached.get("meta", {}),
        )
    result = None
    try:
        entries = query_optimade(MP_BASE, f'id="{mp_id}"', STRUCT_FIELDS, page_limit=1)
    except RuntimeError as e:
        print(f"  [resolve] mp id {mp_id} gagal: {e}")
        entries = []
    for e in entries:
        s = entry_to_structure(e)
        if s is not None:
            result = ResolvedStructure(formula=s.composition.reduced_formula,
                                       provider="mp", source_id=str(e["id"]),
                                       structure=s, meta={})
            break
    if result is None:
        cpath.write_text(json.dumps({"status": "miss", "key": mp_id}), encoding="utf-8")
        return None
    cpath.write_text(json.dumps({
        "status": "hit", "key": mp_id, "formula": result.formula,
        "provider": "mp", "source_id": result.source_id, "meta": result.meta,
        "structure": result.structure.as_dict(),
    }), encoding="utf-8")
    return result


def resolve_structure(formula: str, cache_dir: Path = CACHE_DIR) -> ResolvedStructure | None:
    """Formula -> struktur ordered dari MP lalu OQMD; deterministik; None kalau gagal."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = optimade_reduced_formula(formula)
    cpath = _cache_path(key, cache_dir)
    if cpath.exists():
        cached = json.loads(cpath.read_text(encoding="utf-8"))
        if cached["status"] == "miss":
            return None
        return ResolvedStructure(
            formula=cached["formula"], provider=cached["provider"],
            source_id=cached["source_id"],
            structure=Structure.from_dict(cached["structure"]),
            meta=cached.get("meta", {}),
        )

    result: ResolvedStructure | None = None
    for provider, base, fields in (
        ("mp", MP_BASE, STRUCT_FIELDS),
        ("oqmd", OQMD_BASE, STRUCT_FIELDS + OQMD_EXTRA_FIELDS),
    ):
        try:
            entries = query_optimade(base, f'chemical_formula_reduced="{key}"', fields)
        except RuntimeError as e:
            print(f"  [resolve] {provider} gagal untuk {key}: {e}")
            continue
        # Deterministik: situs paling sedikit dulu (dekat primitive), lalu id.
        entries.sort(key=lambda e: (e["attributes"]["nsites"], str(e["id"])))
        for e in entries:
            s = entry_to_structure(e)
            if s is None:
                continue
            assert e["attributes"]["chemical_formula_reduced"] == key, \
                f"formula mismatch: minta {key}, dapat {e['attributes']['chemical_formula_reduced']}"
            meta = {k: v for k, v in e["attributes"].items()
                    if k.startswith("_oqmd_")}
            result = ResolvedStructure(
                formula=s.composition.reduced_formula, provider=provider,
                source_id=str(e["id"]), structure=s, meta=meta,
            )
            break
        if result is not None:
            break

    if result is None:
        cpath.write_text(json.dumps({"status": "miss", "key": key}), encoding="utf-8")
        return None
    cpath.write_text(json.dumps({
        "status": "hit", "key": key, "formula": result.formula,
        "provider": result.provider, "source_id": result.source_id,
        "meta": result.meta, "structure": result.structure.as_dict(),
    }), encoding="utf-8")
    return result


def fetch_oqmd_hypothetical(candidate_formulas: list[str],
                            exclude_formulas: set[str],
                            stability_max: float = 0.1,
                            batch: int = 12,
                            cache_dir: Path = CACHE_DIR) -> list[ResolvedStructure]:
    """Mining negatif CSLLM dari OQMD: entry tanpa ICSD id, dekat hull.

    candidate_formulas: formula kimia (str bebas, dinormalisasi di sini).
    exclude_formulas: reduced formula (pymatgen) yang diketahui TERSINTESIS.
    Per formula diambil satu entry ber-stability terendah (deterministik).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    exclude_keys = {optimade_reduced_formula(f) for f in exclude_formulas}
    keys = []
    for f in candidate_formulas:
        k = optimade_reduced_formula(f)
        if k not in exclude_keys and k not in keys:
            keys.append(k)

    ccache = cache_dir / "oqmd_hypothetical_batches.json"
    entries_by_key: dict[str, list[dict]] = {}
    cached_batches: dict[str, list] = {}
    if ccache.exists():
        cached_batches = json.loads(ccache.read_text(encoding="utf-8"))

    fields = STRUCT_FIELDS + OQMD_EXTRA_FIELDS
    for i in range(0, len(keys), batch):
        chunk = keys[i:i + batch]
        chunk_id = ",".join(chunk)
        if chunk_id in cached_batches:
            data = cached_batches[chunk_id]
        else:
            flt = " OR ".join(f'chemical_formula_reduced="{k}"' for k in chunk)
            data = query_optimade(OQMD_BASE, flt, fields, page_limit=200)
            cached_batches[chunk_id] = data
            ccache.write_text(json.dumps(cached_batches), encoding="utf-8")
        for e in data:
            entries_by_key.setdefault(e["attributes"]["chemical_formula_reduced"], []).append(e)

    out: list[ResolvedStructure] = []
    for k in keys:
        cands = []
        for e in entries_by_key.get(k, []):
            a = e["attributes"]
            stab = a.get("_oqmd_stability")
            if a.get("_oqmd_icsd_id") is not None:
                continue  # pernah match ICSD -> bukan hipotetis
            if stab is None or stab > stability_max:
                continue
            cands.append(e)
        cands.sort(key=lambda e: (e["attributes"]["_oqmd_stability"], str(e["id"])))
        for e in cands:
            s = entry_to_structure(e)
            if s is None:
                continue
            out.append(ResolvedStructure(
                formula=s.composition.reduced_formula, provider="oqmd",
                source_id=str(e["id"]), structure=s,
                meta={kk: v for kk, v in e["attributes"].items() if kk.startswith("_oqmd_")},
            ))
            break
    return out
