"""
Known Persons & Self Accounts Exclusion

KNOWN LIMITATIONS & INVARIANTS:
- Merchant Suppression Overlap: _MERCHANT_INDICATOR_TOKENS and _MERCHANT_SUFFIXES
  overlap by design and must never be deduplicated. Indicators suppress context; 
  suffixes block concat recovery.
- Double Scoring: Overlap between exact partial match and concat partial match gives +2 score
  on the same word component. This is algorithmically safe but conceptually a double-dip.
  Left as-is to avoid complex branching.
- Suggestion Key Bias: Very short tokens (len=2) like "ok" or "re" may be selected as keys 
  due to `min(key=len)`. Acceptable as suggestions are advisory only and not used for classification.
- Concat Name Recovery: Bounding by business suffixes ("global", "services", etc.) is 
  heuristic. Edge cases may still misclassify if a personal partial alias is prefixed to 
  a non-standard business suffix.
"""

import re
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from logger_factory import get_logger
from schema import Col
import config

logger = get_logger(__name__)

_NAME_NOISE_TOKENS = frozenset({"to", "by", "cr", "dr", "ref", "via", "for", "of"})

_MERCHANT_INDICATOR_TOKENS = frozenset({
    "store", "shop", "mart", "supermarket", "retail", "wholesale", 
    "traders", "trading", "enterprise", "enterprises", "solutions", 
    "technologies", "services", "infotech", "systems", "medical", 
    "pharmacy", "clinic", "hospital", "motors", "auto", "garage"
})

_MERCHANT_SUFFIXES = frozenset({
    "enterprise", "enterprises", "global", "solution", "solutions",
    "services", "technologies", "infotech", "systems"
})

_TRANSFER_CONTEXT_TOKENS = frozenset({
    "upi", "neft", "imps", "rtgs", "fund", "funds", "transfer", "tpt", "mob"
})

# NOTE: @ is included. Without it, "rahul@ybl" survives as a single token 
# causing partial match "rahul" to fail. UPI IDs are extracted BEFORE this 
# separator normalization so the @ in UPI extraction still works correctly.
_SEPARATOR_PATTERN = re.compile(r'[-_/\\|.@]+')

@dataclass
class SignalBundle:
    raw_original: str
    raw_normalized: str
    name_tokens: List[str]
    upi_ids: List[str]
    has_transfer_context: bool
    has_merchant_indicator: bool

def _extract_signals(remark: str) -> SignalBundle:
    raw = str(remark).lower()
    
    # 1. Exact UPI ID extraction BEFORE replacing @
    # e.g., rahul@oksbi123 should be captured
    upi_ids = re.findall(r'[a-z0-9._]+@[a-z0-9]+', raw)
    
    # 2. Transfer context detection BEFORE dropping digits
    has_transfer = any(ctx in raw for ctx in _TRANSFER_CONTEXT_TOKENS)
    
    # 3. Strip digits and normalize separators
    no_digits = ''.join(c if not c.isdigit() else ' ' for c in raw)
    spaced = _SEPARATOR_PATTERN.sub(' ', no_digits)
    
    # 4. Tokenize and filter noise
    tokens = spaced.split()
    name_tokens = [
        t for t in tokens
        if t not in _TRANSFER_CONTEXT_TOKENS and t not in _NAME_NOISE_TOKENS
    ]
    
    # 5. Merchant context
    has_merchant = any(tok in _MERCHANT_INDICATOR_TOKENS for tok in tokens)
    
    return SignalBundle(
        raw_original=raw,
        raw_normalized=' '.join(tokens),
        name_tokens=name_tokens,
        upi_ids=upi_ids,
        has_transfer_context=has_transfer,
        has_merchant_indicator=has_merchant
    )

def _is_contiguous_subsequence(full_name_tokens: List[str], target_tokens: List[str]) -> bool:
    """
    Check if target_tokens appear exactly in that order within full_name_tokens.
    e.g., full=["sujata", "devi", "sharma"], target=["sujata", "devi"] -> True
    e.g., full=["sujata", "sharma"], target=["sujata", "devi"] -> False
    """
    n, m = len(full_name_tokens), len(target_tokens)
    if m == 0 or m > n:
        return False
    
    for i in range(n - m + 1):
        if full_name_tokens[i:i+m] == target_tokens:
            return True
            
    return False

def _find_concat_partial_match(bundle: SignalBundle, partials: List[str]) -> bool:
    """
    Concatenated partial recovery heuristic. Returns True if any name token 
    is a valid concatenation of a known partial. 
    Guards: token len >= 8, partial len >= 4, no merchant suffix.
    """
    long_tokens = [t for t in bundle.name_tokens if len(t) >= config.CONCAT_MIN_LENGTH]
    if not long_tokens:
        return False
        
    for partial in partials:
        if len(partial) < config.CONCAT_PARTIAL_MIN_LENGTH:
            continue
            
        for token in long_tokens:
            if token.startswith(partial) and token != partial:
                if not any(token.endswith(suffix) for suffix in _MERCHANT_SUFFIXES):
                    return True
    return False

def _find_account_fragment_match(bundle: SignalBundle, fragments: List[str]) -> bool:
    """
    Check if any account fragment is present in the raw string.
    """
    return any(frag in bundle.raw_original for frag in fragments)

def _compile_matchers(config_dict: Dict) -> Dict:
    matchers = {}
    for alias, identifiers in config_dict.items():
        # Pre-tokenize full names and partials
        full_names_tokenized = [
            name.lower().split() 
            for name in identifiers.get("names", [])
            if " " in name
        ]
        
        partials = [
            name.lower() 
            for name in identifiers.get("names", [])
            if " " not in name
        ]
        
        matchers[alias] = {
            "full_names": full_names_tokenized,
            "partials": partials,
            "upi_ids": [u.lower() for u in identifiers.get("upi_ids", [])],
            "account_fragments": [str(f).lower() for f in identifiers.get("account_fragments", [])]
        }
    return matchers

def _score_remark(bundle: SignalBundle, matcher: Dict) -> Tuple[int, List[str]]:
    score = 0
    signals = []
    
    # 1. Exact UPI ID
    if set(bundle.upi_ids) & set(matcher["upi_ids"]):
        score += 2
        signals.append("upi_id_match")
        
    # 2. Account Fragment (applies mostly to SELF_ACCOUNTS)
    if _find_account_fragment_match(bundle, matcher.get("account_fragments", [])):
        score += 2
        signals.append("account_fragment_match")
        
    # 3. Full / Multi-token Name
    has_full_name = False
    for fn_tokens in matcher["full_names"]:
        if _is_contiguous_subsequence(bundle.name_tokens, fn_tokens):
            score += 2
            signals.append(f"full_name_match: {' '.join(fn_tokens)}")
            has_full_name = True
            break
            
    # 4. Partial Name (standalone exact token)
    has_partial = False
    if not has_full_name:
        for partial in matcher["partials"]:
            if partial in bundle.name_tokens:
                score += 1
                signals.append(f"partial_name: {partial}")
                has_partial = True
                break
                
    # 5. Concatenated Partial Name
    if not has_partial and not has_full_name:
        if _find_concat_partial_match(bundle, matcher["partials"]):
            score += 1
            signals.append("concat_partial_match")
            
    # 6. Transfer Context
    if bundle.has_transfer_context:
        if bundle.has_merchant_indicator:
            # Merchant overrides context. But if score >= 2 from UPI, it stays classified.
            signals.append("transfer_context_suppressed_by_merchant")
        else:
            score += 1
            signals.append("transfer_context")
            
    return score, signals

def tag_known_persons(
    df: pd.DataFrame,
    known_persons: dict = None,
    self_accounts: dict = None,
) -> pd.DataFrame:
    df = df.copy()
    
    if Col.IS_KNOWN_PERSON not in df.columns:
        df[Col.IS_KNOWN_PERSON] = False
        df[Col.KNOWN_PERSON_ALIAS] = pd.NA
        df[Col.TRANSFER_CLASS] = "transfer_external"
        df[Col.MATCH_SCORE] = pd.NA
        df[Col.MATCH_SIGNALS] = pd.NA
        
    kp = known_persons if known_persons is not None else config.KNOWN_PERSONS
    sa = self_accounts if self_accounts is not None else config.SELF_ACCOUNTS
    
    # If both configs are empty, return early to save processing time
    if not kp and not sa:
        return df
        
    kp_matchers = _compile_matchers(kp)
    sa_matchers = _compile_matchers(sa)
    
    for idx, row in df.iterrows():
        # Do not override existing known personal classifications in case of multiple passes
        if pd.notna(row.get(Col.IS_KNOWN_PERSON)) and row.get(Col.IS_KNOWN_PERSON) is True:
            continue
            
        remark = row.get(Col.REMARKS, "")
        bundle = _extract_signals(remark)
        
        best_score = -1
        best_alias = pd.NA
        best_signals = []
        best_class = "transfer_external"
        
        # Check self accounts first
        for alias, matcher in sa_matchers.items():
            score, signals = _score_remark(bundle, matcher)
            if score > best_score:
                best_score = score
                best_alias = f"Self:{alias}"
                best_signals = signals
                best_class = "transfer_self"
                
        # Check known persons
        for alias, matcher in kp_matchers.items():
            score, signals = _score_remark(bundle, matcher)
            if score > best_score:
                best_score = score
                best_alias = alias
                best_signals = signals
                best_class = "transfer_known"
                
        if best_score >= config.KNOWN_PERSON_MATCH_THRESHOLD:
            df.at[idx, Col.IS_KNOWN_PERSON] = True
            df.at[idx, Col.KNOWN_PERSON_ALIAS] = best_alias
            df.at[idx, Col.TRANSFER_CLASS] = best_class
            df.at[idx, Col.MATCH_SCORE] = best_score
            df.at[idx, Col.MATCH_SIGNALS] = str(best_signals)
            
    return df

def _enforce_known_person_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee: is_known_person == True -> ML columns are pd.NA. Checks before copying."""
    if Col.IS_KNOWN_PERSON not in df.columns:
        return df
        
    personal_mask = df[Col.IS_KNOWN_PERSON].fillna(False)
    if not personal_mask.any():
        return df
        
    required_na_cols = [c for c in Col.ml_output_columns() if c in df.columns]
    needs_fix = False
    
    for col in required_na_cols:
        if df.loc[personal_mask, col].notna().any():
            needs_fix = True
            break
            
    if not needs_fix:
        return df
        
    logger.warning("Personal rows found with populated ML columns. Forcing NaN.",
                   extra={"event_type": "enforce_known_person_schema_fix"})
                   
    df = df.copy()
    for col in required_na_cols:
        df.loc[personal_mask, col] = pd.NA
        
    return df

def _suggestion_key(bundle: SignalBundle) -> str:
    """
    Derive a grouping key from structured signals for unmatched
    transfer suggestions.
    
    KNOWN LIMITATION: Still imperfect. Very short tokens (len=2)
    like "ok", "re" may be selected as keys. Acceptable because
    suggestions are advisory-only and never auto-classify.
    """
    if bundle.upi_ids:
        return bundle.upi_ids[0]
    
    if bundle.name_tokens:
        return min(bundle.name_tokens, key=len)
    
    return bundle.raw_normalized

def log_unmatched_recurring_transfers(df: pd.DataFrame) -> None:
    if Col.TRANSFER_CLASS not in df.columns:
        return
        
    external_mask = df[Col.TRANSFER_CLASS] == "transfer_external"
    external = df.loc[external_mask]
    
    if external.empty:
        return
        
    keys = external[Col.REMARKS].apply(
        lambda r: _suggestion_key(_extract_signals(str(r)))
    )
    
    counts = keys.value_counts()
    frequent = counts[counts >= 3]
    
    for key, count in frequent.items():
        logger.info(
            f"Frequent unmatched transfer: '{key}' ({count} occurrences). "
            "Consider adding to KNOWN_PERSONS if this is a known person.",
            extra={
                "event_type": "unmatched_transfer_suggestion",
                "metrics": {"suggestion_key": key, "count": int(count)}
            }
        )

def _analyze_person_group(group: pd.DataFrame, alias: str, cfg: dict) -> dict:
    group = group.sort_values(Col.DATE)
    dates = pd.to_datetime(group[Col.DATE])
    amounts = group[Col.AMOUNT].abs()
    
    if len(group) < cfg["global"]["min_occurrences"]:
        return {"pattern": "ad_hoc", "alias": alias, "count": len(group)}
        
    gaps = dates.diff().dt.days.dropna()
    mean_gap = gaps.mean()
    mean_amount = amounts.mean()
    
    # Analyze Monthly
    m_cfg = cfg["monthly"]
    is_monthly = m_cfg["min_gap"] <= mean_gap <= m_cfg["max_gap"]
    if is_monthly and gaps.std() <= m_cfg["var"]:
        amt_std_pct = amounts.std() / mean_amount if mean_amount > 0 else 0
        if amt_std_pct <= cfg["global"]["amount_tolerance"]:
            return {
                "pattern": "monthly_support", 
                "alias": alias, 
                "count": len(group),
                "frequency_days": float(mean_gap),
                "avg_amount": float(mean_amount)
            }
            
    # Analyze Weekly
    w_cfg = cfg["weekly"]
    is_weekly = w_cfg["min_gap"] <= mean_gap <= w_cfg["max_gap"]
    if is_weekly and gaps.std() <= w_cfg["var"]:
        amt_std_pct = amounts.std() / mean_amount if mean_amount > 0 else 0
        if amt_std_pct <= cfg["global"]["amount_tolerance"]:
            return {
                "pattern": "weekly_allowance", 
                "alias": alias, 
                "count": len(group),
                "frequency_days": float(mean_gap),
                "avg_amount": float(mean_amount)
            }
            
    return {"pattern": "irregular_support", "alias": alias, "count": len(group)}

def detect_personal_patterns(
    personal_df: pd.DataFrame,
    cfg: dict = None,
) -> tuple[List[str], dict]:
    c = cfg if cfg is not None else config.PERSONAL_RECURRING_CONFIG
    
    if personal_df.empty or Col.KNOWN_PERSON_ALIAS not in personal_df.columns:
        return [], {}
        
    insights = []
    summary_dict = {}
    
    for alias, group in personal_df.groupby(Col.KNOWN_PERSON_ALIAS):
        # We only look at alias groups from KNOWN_PERSONS, skip 'Self:'
        if pd.isna(alias) or str(alias).startswith("Self:"):
            continue
            
        analysis = _analyze_person_group(group, str(alias), c)
        
        pattern = analysis["pattern"]
        if pattern == "monthly_support":
            insights.append(f"Monthly support detected for '{alias}' averaging ~₹{analysis['avg_amount']:.2f}")
        elif pattern == "weekly_allowance":
            insights.append(f"Weekly allowance detected for '{alias}' averaging ~₹{analysis['avg_amount']:.2f}")
        elif pattern == "irregular_support":
            insights.append(f"Irregular transfers to '{alias}' ({analysis['count']} occurrences)")
            
        summary_dict[str(alias)] = analysis
            
    return insights, summary_dict
