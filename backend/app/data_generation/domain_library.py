"""Domain and scenario library for high-diversity prompt generation."""

from __future__ import annotations

from dataclasses import dataclass


CATEGORIES: tuple[str, ...] = (
    "coding",
    "planning",
    "reasoning",
    "translation",
    "summarization",
    "creative_writing",
    "general",
    "mathematics",
)

DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")

DOMAINS: tuple[str, ...] = (
    "healthcare",
    "finance",
    "retail",
    "cloud",
    "cybersecurity",
    "education",
    "machine_learning",
    "databases",
    "distributed_systems",
    "robotics",
    "legal",
    "research",
    "business",
    "government",
    "science",
)


@dataclass(frozen=True)
class DomainProfile:
    """Domain-specific vocabulary and realistic task settings."""

    name: str
    actors: tuple[str, ...]
    artifacts: tuple[str, ...]
    risks: tuple[str, ...]
    metrics: tuple[str, ...]
    entities: tuple[str, ...]


DOMAIN_PROFILES: dict[str, DomainProfile] = {
    "healthcare": DomainProfile(
        "healthcare",
        ("clinical operations team", "hospital data analyst", "patient safety committee", "telehealth product manager"),
        ("triage protocol", "claims feed", "clinical note", "consent form", "appointment workflow"),
        ("privacy exposure", "incorrect dosage guidance", "care delay", "audit failure"),
        ("readmission rate", "wait time", "patient satisfaction", "false negative rate"),
        ("patient cohort", "nurse queue", "specialist referral", "EHR record"),
    ),
    "finance": DomainProfile(
        "finance",
        ("risk analyst", "payments engineer", "portfolio manager", "compliance officer"),
        ("ledger export", "credit model", "settlement report", "cash-flow forecast", "fraud alert"),
        ("model drift", "liquidity shortfall", "regulatory breach", "chargeback spike"),
        ("basis points", "default rate", "gross margin", "transaction volume"),
        ("merchant account", "loan applicant", "invoice batch", "market scenario"),
    ),
    "retail": DomainProfile(
        "retail",
        ("store operations lead", "merchandising analyst", "ecommerce manager", "supply planner"),
        ("inventory snapshot", "promotion calendar", "returns log", "product catalog", "customer review feed"),
        ("stockout", "margin erosion", "late delivery", "pricing mismatch"),
        ("conversion rate", "sell-through", "basket size", "return rate"),
        ("SKU", "warehouse zone", "loyalty segment", "supplier"),
    ),
    "cloud": DomainProfile(
        "cloud",
        ("platform engineer", "site reliability engineer", "cloud architect", "FinOps analyst"),
        ("Kubernetes cluster", "Terraform module", "service mesh", "load balancer", "object storage policy"),
        ("regional outage", "cost overrun", "credential leakage", "deployment rollback"),
        ("p95 latency", "error budget", "egress cost", "CPU utilization"),
        ("tenant namespace", "VPC", "autoscaling group", "container image"),
    ),
    "cybersecurity": DomainProfile(
        "cybersecurity",
        ("SOC analyst", "incident commander", "security engineer", "threat researcher"),
        ("SIEM alert", "phishing report", "malware sample", "access log", "vulnerability advisory"),
        ("privilege escalation", "data exfiltration", "false positive", "credential compromise"),
        ("MTTR", "severity score", "attack dwell time", "patch coverage"),
        ("endpoint", "identity provider", "firewall rule", "threat actor"),
    ),
    "education": DomainProfile(
        "education",
        ("curriculum designer", "school administrator", "teacher coach", "learning analyst"),
        ("lesson plan", "assessment rubric", "student survey", "attendance dataset", "course outline"),
        ("accessibility gap", "grading inconsistency", "low engagement", "privacy concern"),
        ("completion rate", "reading level", "attendance", "assessment score"),
        ("student cohort", "classroom", "online module", "parent group"),
    ),
    "machine_learning": DomainProfile(
        "machine_learning",
        ("ML engineer", "data scientist", "model risk reviewer", "MLOps lead"),
        ("feature store", "training run", "model card", "evaluation notebook", "labeling guideline"),
        ("data leakage", "bias amplification", "overfitting", "serving skew"),
        ("ROC AUC", "F1 score", "calibration error", "inference latency"),
        ("embedding model", "validation split", "feature pipeline", "drift monitor"),
    ),
    "databases": DomainProfile(
        "databases",
        ("database administrator", "backend engineer", "data architect", "analytics engineer"),
        ("query plan", "migration script", "replication stream", "schema design", "index report"),
        ("deadlock", "slow query", "data loss", "schema drift"),
        ("query latency", "lock wait time", "replica lag", "storage growth"),
        ("table partition", "foreign key", "materialized view", "transaction"),
    ),
    "distributed_systems": DomainProfile(
        "distributed_systems",
        ("distributed systems engineer", "infrastructure architect", "reliability lead", "protocol designer"),
        ("consensus protocol", "message queue", "sharded service", "cache layer", "replication log"),
        ("split-brain", "message loss", "clock skew", "cascading failure"),
        ("availability", "throughput", "quorum size", "tail latency"),
        ("node", "partition", "replica set", "event stream"),
    ),
    "robotics": DomainProfile(
        "robotics",
        ("robotics engineer", "warehouse automation lead", "controls researcher", "field technician"),
        ("sensor log", "motion planner", "robot arm routine", "navigation map", "safety envelope"),
        ("collision risk", "sensor drift", "actuator failure", "unsafe trajectory"),
        ("position error", "cycle time", "battery level", "path length"),
        ("mobile robot", "gripper", "LIDAR scan", "control loop"),
    ),
    "legal": DomainProfile(
        "legal",
        ("legal operations manager", "privacy counsel", "contract analyst", "policy reviewer"),
        ("data processing agreement", "employment clause", "case summary", "compliance memo", "discovery request"),
        ("ambiguous liability", "jurisdiction conflict", "confidentiality breach", "missed obligation"),
        ("review time", "risk rating", "clause count", "retention period"),
        ("counterparty", "regulated entity", "document custodian", "contract template"),
    ),
    "research": DomainProfile(
        "research",
        ("principal investigator", "research coordinator", "peer reviewer", "lab manager"),
        ("study protocol", "literature review", "survey instrument", "experimental log", "grant summary"),
        ("sampling bias", "low power", "reproducibility issue", "ethics review delay"),
        ("p-value", "effect size", "sample size", "confidence interval"),
        ("participant group", "control condition", "dataset", "measurement instrument"),
    ),
    "business": DomainProfile(
        "business",
        ("operations director", "product manager", "sales lead", "strategy analyst"),
        ("quarterly plan", "customer interview", "pricing model", "market analysis", "support queue"),
        ("churn increase", "scope creep", "budget gap", "misaligned incentives"),
        ("ARR", "conversion rate", "NPS", "payback period"),
        ("customer segment", "sales pipeline", "feature request", "vendor"),
    ),
    "government": DomainProfile(
        "government",
        ("policy analyst", "public services manager", "procurement officer", "program evaluator"),
        ("public records request", "benefits workflow", "budget proposal", "citizen survey", "agency memo"),
        ("equity gap", "procurement delay", "compliance finding", "service backlog"),
        ("case volume", "processing time", "approval rate", "program cost"),
        ("resident", "agency", "district", "public dataset"),
    ),
    "science": DomainProfile(
        "science",
        ("lab scientist", "climate analyst", "field researcher", "science communicator"),
        ("sensor reading", "lab protocol", "simulation result", "field observation", "journal abstract"),
        ("measurement error", "contamination", "model uncertainty", "instrument drift"),
        ("temperature anomaly", "concentration", "confidence interval", "sample purity"),
        ("specimen", "weather station", "chemical compound", "simulation grid"),
    ),
}


def domain_profile(name: str) -> DomainProfile:
    """Return a domain profile by name."""
    return DOMAIN_PROFILES[name]

