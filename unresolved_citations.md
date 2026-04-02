# Unresolved Citations Report

**Date:** 2026-04-02  
**Task:** Documentation-only citation migration validation

## Summary

All citations used throughout the GeoPrior-v3 documentation have been validated against `docs/source/references.bib`.

**Result:** ✅ **ALL CITATIONS RESOLVED**

## Details

### Python Docstrings
- **File:** `geoprior/utils/calibrate.py`
- **Citation count:** 2 unique keys across 10 references
- **Keys validated:**
  - ✅ `Limetal2021` — Found in references.bib (article)
  - ✅ `Kouadio2025XTFT` — Found in references.bib (misc)

### RST Documentation Files
- **Total files with citations:** 10
- **Total citation occurrences:** 36
- **Unique citation keys:** 21
- **All keys validated:** ✅ Yes

**Resolved citation keys:**
- ✅ `CandelaKoster2022`
- ✅ `Liuetal2024`
- ✅ `Limetal2021` (also used in Python docstrings)
- ✅ `kouadio_geopriorsubsnet_nature_2025`
- ✅ `Shirzaeietal2021`
- ✅ `Nichollsetal2021`
- ✅ `Fangetal2022`
- ✅ `GallowayBurbey2011`
- ✅ `Ellisetal2023`
- ✅ `Donnelly2023`
- ✅ `Roy2024`
- ✅ `Sarmaetal2024IPINNs`
- ✅ `Hoffmannetal2000`
- ✅ `Bennethumetal1997`
- ✅ `ZapataNorbertoetal2025`
- ✅ `Royetal2024HeteroPoroelasticPINNs`
- ✅ `Hasanetal2023`
- ✅ `Bagherietal2021`
- ✅ `Kouadio2025XTFT` (also used in Python docstrings)

### Placeholder Example
- **File:** `docs/source/references.rst`
- **Note:** Contains example placeholders `some_key` used in code-block documentation (lines 27-28)
- **Status:** Documentation example only—not a real citation

## Citation Format Validation

All citations follow proper Sphinx BibTeX syntax:
- ✅ Parenthetical citations: `:cite:p:`key``
- ✅ Textual citations: `:cite:t:`key``
- ✅ Multiple citations: `:cite:p:`key1,key2``

## Conclusion

The citation migration validation is complete. All citations in Python docstrings and RST documentation files are:
1. **Properly formatted** as Sphinx BibTeX roles
2. **Correctly resolved** to entries in `docs/source/references.bib`
3. **Documented and traceable** for future maintenance

No migration action required.
