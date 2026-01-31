"""
FSKU (Financial Security Knowledge Understanding) Benchmark
Category and Domain Structure Definition

This file defines the hierarchical structure of the FSKU benchmark,
mapping domains to their sub-domains and providing metadata for evaluation.
"""

# =============================================================================
# Domain -> Sub_domain Mapping
# =============================================================================

DOMAIN_SUBDOMAIN_MAP = {
    "금융보안IT": [
        "금융IT보안 심화",
        "금융IT보안 일반",
    ],
    "금융보안동향": [
        "금융보안 최신동향",
    ],
    "금융보안법률": [
        "개인정보보호",
        "신용정보",
        "전자금융",
        "전자서명",
        "정보통신",
    ],
    "금융보안일반": [
        "개인정보보호",
        "금융 비즈니스",
        "금융보안관리체계",
        "전자금융보안",
    ],
}

# =============================================================================
# English Translation Mapping (for international use)
# =============================================================================

DOMAIN_TRANSLATION = {
    "금융보안IT": "Financial Security IT",
    "금융보안동향": "Financial Security Trends",
    "금융보안법률": "Financial Security Laws",
    "금융보안일반": "Financial Security Fundamentals",
}

SUBDOMAIN_TRANSLATION = {
    # 금융보안IT
    "금융IT보안 심화": "Advanced Financial IT Security",
    "금융IT보안 일반": "General Financial IT Security",
    # 금융보안동향
    "금융보안 최신동향": "Latest Financial Security Trends",
    # 금융보안법률
    "개인정보보호": "Personal Information Protection",
    "신용정보": "Credit Information",
    "전자금융": "Electronic Finance",
    "전자서명": "Digital Signature",
    "정보통신": "Information and Communications",
    # 금융보안일반
    "금융 비즈니스": "Financial Business",
    "금융보안관리체계": "Financial Security Management System",
    "전자금융보안": "Electronic Financial Security",
}

# =============================================================================
# High-level Category Grouping (similar to MMLU's Humanities, STEM, etc.)
# =============================================================================

CATEGORY_GROUPS = {
    "Technical": [
        "금융보안IT",
    ],
    "Legal": [
        "금융보안법률",
    ],
    "General": [
        "금융보안일반",
    ],
    "Trends": [
        "금융보안동향",
    ],
}

# Reverse mapping: domain -> category
DOMAIN_TO_CATEGORY = {}
for category, domains in CATEGORY_GROUPS.items():
    for domain in domains:
        DOMAIN_TO_CATEGORY[domain] = category

# =============================================================================
# Dataset Statistics
# =============================================================================

DATASET_STATS = {
    "total_questions": 1000,
    "num_domains": 4,
    "num_subdomains": 12,
    "num_choices_per_question": 4,
    "question_format": "multiple_choice",
    "language": "Korean",
}

DOMAIN_QUESTION_COUNTS = {
    "금융보안IT": {
        "금융IT보안 심화": 120,
        "금융IT보안 일반": 80,
        "total": 200,
    },
    "금융보안동향": {
        "금융보안 최신동향": 50,
        "total": 50,
    },
    "금융보안법률": {
        "개인정보보호": 120,
        "신용정보": 120,
        "전자금융": 120,
        "전자서명": 70,
        "정보통신": 120,
        "total": 550,
    },
    "금융보안일반": {
        "개인정보보호": 50,
        "금융 비즈니스": 50,
        "금융보안관리체계": 50,
        "전자금융보안": 50,
        "total": 200,
    },
}

CATEGORY_QUESTION_COUNTS = {
    "Technical": 200,   # 금융보안IT
    "Legal": 550,       # 금융보안법률
    "General": 200,     # 금융보안일반
    "Trends": 50,       # 금융보안동향
}

# =============================================================================
# Utility Functions
# =============================================================================

def get_all_domains():
    """Return list of all domains."""
    return list(DOMAIN_SUBDOMAIN_MAP.keys())


def get_all_subdomains():
    """Return list of all sub-domains."""
    subdomains = []
    for subs in DOMAIN_SUBDOMAIN_MAP.values():
        subdomains.extend(subs)
    return subdomains


def get_subdomains_by_domain(domain):
    """Return list of sub-domains for a given domain."""
    return DOMAIN_SUBDOMAIN_MAP.get(domain, [])


def get_domain_by_subdomain(subdomain):
    """Return the domain that contains the given sub-domain."""
    for domain, subs in DOMAIN_SUBDOMAIN_MAP.items():
        if subdomain in subs:
            return domain
    return None


def get_category_by_domain(domain):
    """Return the high-level category for a given domain."""
    return DOMAIN_TO_CATEGORY.get(domain, None)


def get_question_count(domain=None, subdomain=None):
    """
    Return the number of questions for a given domain or sub-domain.
    If no arguments provided, returns total question count.
    """
    if subdomain:
        domain = get_domain_by_subdomain(subdomain)
        if domain:
            return DOMAIN_QUESTION_COUNTS.get(domain, {}).get(subdomain, 0)
        return 0
    elif domain:
        return DOMAIN_QUESTION_COUNTS.get(domain, {}).get("total", 0)
    else:
        return DATASET_STATS["total_questions"]


def print_structure():
    """Print the complete benchmark structure."""
    print("=" * 60)
    print("FSKU Benchmark Structure")
    print("=" * 60)
    print(f"Total Questions: {DATASET_STATS['total_questions']}")
    print(f"Domains: {DATASET_STATS['num_domains']}")
    print(f"Sub-domains: {DATASET_STATS['num_subdomains']}")
    print(f"Choices per Question: {DATASET_STATS['num_choices_per_question']}")
    print("=" * 60)
    
    for domain, subdomains in DOMAIN_SUBDOMAIN_MAP.items():
        domain_total = DOMAIN_QUESTION_COUNTS[domain]["total"]
        category = DOMAIN_TO_CATEGORY[domain]
        print(f"\n[{domain}] ({DOMAIN_TRANSLATION[domain]})")
        print(f"  Category: {category} | Total: {domain_total} questions")
        print(f"  Sub-domains:")
        for sub in subdomains:
            count = DOMAIN_QUESTION_COUNTS[domain].get(sub, 0)
            print(f"    - {sub} ({SUBDOMAIN_TRANSLATION[sub]}): {count}")


if __name__ == "__main__":
    print_structure()
