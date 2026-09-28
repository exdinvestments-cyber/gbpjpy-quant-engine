"""Phase 1F: Account Risk Engine (PROPOSED TRADE -> RISK-APPROVED TRADE).

Sizes Phase 1E proposals from account state and a conservative, centralised
risk policy.  A risk-approved trade is NOT an order: nothing here connects to
a platform or places, modifies or closes anything.  Quality scores and profit
objectives are not risk inputs.
"""

from .account import AccountState, AccountStateSource, BalanceAdjustment, ClosedResultSink, ClosedTradeResult, OpenPosition, StopChange
from .config import RiskPolicy, load_risk_policy, risk_policy_from_dict
from .engine import RESET_CONFIRMATION, AccountRiskEngine, ResetAuthorisation, geometry_fingerprint
from .fx import ConversionRate, ConversionUnavailable, RateProvider, StaticRates, factor, rates_from_bars
from .instruments import GBPJPY_CONTRACT, ContractSpec, ContractSpecSource
from .margin import FixedMarginPerVolume, MarginEstimate, MarginModel, NotionalLeverageMarginModel, UnknownMargin
from .periods import period_keys
from .policies import ExternalContext, ExternalPolicyResult, ExternalRiskPolicy, GenericRuleset
from .report import explain_risk_decision
from .research import ResearchAccount, ResearchRunner, RiskOfRuinInputs, compare_policies, monte_carlo_records, risk_of_ruin
from .state import CorruptedRiskState, InMemoryRiskStateStore, JsonFileRiskStateStore, RiskState

__all__ = ["AccountRiskEngine", "AccountState", "AccountStateSource", "ClosedResultSink", "BalanceAdjustment", "ClosedTradeResult", "ContractSpec", "ContractSpecSource",
           "ConversionRate", "ConversionUnavailable", "CorruptedRiskState", "ExternalContext", "ExternalPolicyResult",
           "ExternalRiskPolicy", "FixedMarginPerVolume", "GBPJPY_CONTRACT", "GenericRuleset", "InMemoryRiskStateStore",
           "JsonFileRiskStateStore", "MarginEstimate", "MarginModel", "NotionalLeverageMarginModel", "OpenPosition",
           "RESET_CONFIRMATION", "RateProvider", "ResearchAccount", "ResearchRunner", "ResetAuthorisation", "RiskOfRuinInputs",
           "RiskPolicy", "RiskState", "StaticRates", "StopChange", "UnknownMargin", "compare_policies", "explain_risk_decision",
           "factor", "geometry_fingerprint", "load_risk_policy", "monte_carlo_records", "period_keys", "rates_from_bars",
           "risk_of_ruin", "risk_policy_from_dict"]
