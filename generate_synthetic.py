"""
Synthetic schema pairs generator for schema matching benchmarking.
Generates 5 standard test cases covering:
1. Exact name match, identical values (sanity check)
2. Renamed columns, identical value distributions (tests value signal)
3. Similar names, conflicting data types (tests type-mismatch penalty)
4. Mixed: exact, renamed, and unmatchable columns
5. Real-world Customer vs Client schema (8 columns)

Seeded, deterministic, and fully reproducible.
Saves pairs to data/synthetic/pair_1.json .. pair_5.json alongside ground truth files.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

from person_a.normalize_names import normalize_column_name
from shared.contracts import ColumnProfile, validate_column_profile
from value_pipeline.profiler import build_value_profile
from value_pipeline.sampler import sample_column

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "synthetic"


def make_column(
    column_id: str,
    raw_name: str,
    tokens: List[str],
    sample_values: List[Any],
    total_count: int = 100
) -> Dict[str, Any]:
    """
    Construct a full ColumnProfile object complying with shared contracts.
    """
    sampled = sample_column(sample_values, n=500, random_state=42)
    profile = build_value_profile(sampled, total_count=max(total_count, len(sample_values)))

    # Use Person A's normalizer if tokens not provided
    if not tokens:
        tokens = normalize_column_name(raw_name)

    schema_name = column_id.split(".")[0] if "." in column_id else "default"

    col_profile = ColumnProfile(
        column_id=column_id,
        schema_name=schema_name,
        raw_name=raw_name,
        normalized_tokens=tokens,
        sampled_values=sampled,
        name_embedding=[],
        value_profile=profile,
        data_type=profile["dtype"]
    )

    # Validate against JSON schema contract
    validate_column_profile(col_profile)
    return col_profile.to_dict()


def generate_all_pairs() -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []

    # =========================================================================
    # Pair 1: Exact Name Match, Same Values (Sanity check, expect F1 ≈ 1.0)
    # =========================================================================
    ids = ["C101", "C102", "C103", "C104", "C105"]
    names = ["Alice Smith", "Bob Jones", "Charlie Brown", "Diana Prince", "Evan Wright"]
    emails = ["alice@example.com", "bob@example.com", "charlie@corp.org", "diana@mail.com", "evan@test.net"]
    balances = [150.50, 2300.00, 45.10, 890.25, 12.00]

    pair_1_a = [
        make_column("DB1.Users.id", "id", ["id"], ids),
        make_column("DB1.Users.name", "name", ["name"], names),
        make_column("DB1.Users.email", "email", ["email"], emails),
        make_column("DB1.Users.balance", "balance", ["balance"], balances),
    ]
    pair_1_b = [
        make_column("DB2.Users.id", "id", ["id"], ids),
        make_column("DB2.Users.name", "name", ["name"], names),
        make_column("DB2.Users.email", "email", ["email"], emails),
        make_column("DB2.Users.balance", "balance", ["balance"], balances),
    ]
    gt_1 = [
        ("DB1.Users.id", "DB2.Users.id"),
        ("DB1.Users.name", "DB2.Users.name"),
        ("DB1.Users.email", "DB2.Users.email"),
        ("DB1.Users.balance", "DB2.Users.balance"),
    ]
    pairs.append({
        "pair_id": "pair_1",
        "description": "Exact name match, same values (sanity check)",
        "profiles_a": pair_1_a,
        "profiles_b": pair_1_b,
        "ground_truth": gt_1
    })

    # =========================================================================
    # Pair 2: Renamed Columns, Same Value Distribution (Tests Value Signal)
    # =========================================================================
    cust_codes = ["TXN-101", "TXN-102", "TXN-103", "TXN-104", "TXN-105"]
    phones = ["+1-555-0192", "+1-555-0193", "+1-555-0194", "+1-555-0195", "+1-555-0196"]
    dates = ["2023-01-15", "2023-03-22", "2023-06-01", "2023-09-18", "2023-11-30"]
    amounts = [49.99, 129.50, 9.99, 899.00, 24.50]

    pair_2_a = [
        make_column("Source.Orders.ref_num", "ref_num", ["ref", "num"], cust_codes),
        make_column("Source.Orders.contact_phone", "contact_phone", ["contact", "phone"], phones),
        make_column("Source.Orders.placed_on", "placed_on", ["placed", "on"], dates),
        make_column("Source.Orders.total_due", "total_due", ["total", "due"], amounts),
    ]
    pair_2_b = [
        make_column("Target.Invoices.identifier", "identifier", ["identifier"], cust_codes),
        make_column("Target.Invoices.telephone", "telephone", ["telephone"], phones),
        make_column("Target.Invoices.invoice_date", "invoice_date", ["invoice", "date"], dates),
        make_column("Target.Invoices.amount_charged", "amount_charged", ["amount", "charged"], amounts),
    ]
    gt_2 = [
        ("Source.Orders.ref_num", "Target.Invoices.identifier"),
        ("Source.Orders.contact_phone", "Target.Invoices.telephone"),
        ("Source.Orders.placed_on", "Target.Invoices.invoice_date"),
        ("Source.Orders.total_due", "Target.Invoices.amount_charged"),
    ]
    pairs.append({
        "pair_id": "pair_2",
        "description": "Renamed columns, same value distribution (tests value signal)",
        "profiles_a": pair_2_a,
        "profiles_b": pair_2_b,
        "ground_truth": gt_2
    })

    # =========================================================================
    # Pair 3: Similar Names, Different Value Types (Tests Dtype mismatch penalty)
    # =========================================================================
    pair_3_a = [
        make_column("DB1.Profiles.age", "age", ["age"], [25.0, 34.0, 42.0, 19.0, 58.0]),
        make_column("DB1.Profiles.status", "status", ["status"], ["ACTIVE", "ACTIVE", "PENDING", "ACTIVE", "INACTIVE"] * 10),
        make_column("DB1.Profiles.score", "score", ["score"], [88.5, 92.0, 79.0, 95.5, 84.0]),
    ]
    pair_3_b = [
        make_column("DB2.Records.age", "age", ["age"], ["1999-01-15", "1990-08-22", "1982-12-05", "2005-03-10", "1966-07-19"]),
        make_column("DB2.Records.status", "status", ["status"], [100.0, 100.0, 200.0, 100.0, 300.0]),
        make_column("DB2.Records.score", "score", ["score"], [88.5, 92.0, 79.0, 95.5, 84.0]),
    ]
    gt_3 = [
        ("DB1.Profiles.score", "DB2.Records.score")
    ]
    pairs.append({
        "pair_id": "pair_3",
        "description": "Similar names, different value types (tests dtype penalty)",
        "profiles_a": pair_3_a,
        "profiles_b": pair_3_b,
        "ground_truth": gt_3
    })

    # =========================================================================
    # Pair 4: Mixed: Exact matches, renamed, and one unmatchable column
    # =========================================================================
    pair_4_a = [
        make_column("App1.Items.sku", "sku", ["sku"], ["SKU-001", "SKU-002", "SKU-003", "SKU-004"]),
        make_column("App1.Items.name", "name", ["name"], ["Widget A", "Gadget B", "Tool C", "Device D"]),
        make_column("App1.Items.price", "price", ["price"], [19.99, 49.50, 5.00, 120.00]),
        make_column("App1.Items.internal_warehouse_bin", "internal_bin", ["internal", "bin"], ["BIN-NORTH-01", "BIN-SOUTH-02", "BIN-EAST-03", "BIN-WEST-04"]),
    ]
    pair_4_b = [
        make_column("App2.Catalog.product_code", "product_code", ["product", "code"], ["SKU-001", "SKU-002", "SKU-003", "SKU-004"]),
        make_column("App2.Catalog.name", "name", ["name"], ["Widget A", "Gadget B", "Tool C", "Device D"]),
        make_column("App2.Catalog.cost", "cost", ["cost"], [19.99, 49.50, 5.00, 120.00]),
        make_column("App2.Catalog.supplier_notes", "notes", ["notes"], ["Local vendor", "Bulk supplier", "Direct order", "Overseas"]),
    ]
    gt_4 = [
        ("App1.Items.sku", "App2.Catalog.product_code"),
        ("App1.Items.name", "App2.Catalog.name"),
        ("App1.Items.price", "App2.Catalog.cost"),
    ]
    pairs.append({
        "pair_id": "pair_4",
        "description": "Mixed: exact matches, renamed columns, and unmatchable column",
        "profiles_a": pair_4_a,
        "profiles_b": pair_4_b,
        "ground_truth": gt_4
    })

    # =========================================================================
    # Pair 5: Real-world Customer vs Client Tables (8 columns each)
    # =========================================================================
    c_ids = ["CUST-001", "CUST-002", "CUST-003", "CUST-004", "CUST-005", "CUST-006"]
    first_names = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia"]
    cust_emails = ["james.s@gmail.com", "mary.j@yahoo.com", "rob.w@corp.com", "pat.b@outlook.com", "john.j@mail.com", "jen.g@web.de"]
    cust_phones = ["(555) 234-5678", "(555) 345-6789", "(555) 456-7890", "(555) 567-8901", "(555) 678-9012", "(555) 789-0123"]
    reg_dates = ["2021-04-12", "2021-08-19", "2022-01-05", "2022-06-23", "2023-02-14", "2023-10-09"]
    credit_scores = [720.0, 685.0, 750.0, 610.0, 805.0, 690.0]
    is_active = ["ACTIVE", "ACTIVE", "INACTIVE", "ACTIVE", "ACTIVE", "INACTIVE"] * 10

    pair_5_a = [
        make_column("CRM.Customer.cust_id", "cust_id", ["customer", "id"], c_ids),
        make_column("CRM.Customer.fname", "fname", ["first", "name"], first_names),
        make_column("CRM.Customer.lname", "lname", ["last", "name"], last_names),
        make_column("CRM.Customer.email_addr", "email_addr", ["email", "address"], cust_emails),
        make_column("CRM.Customer.phone_num", "phone_num", ["phone", "number"], cust_phones),
        make_column("CRM.Customer.created_at", "created_at", ["created", "at"], reg_dates),
        make_column("CRM.Customer.credit_rating", "credit_rating", ["credit", "rating"], credit_scores),
        make_column("CRM.Customer.active_flag", "active_flag", ["active", "flag"], is_active),
    ]
    pair_5_b = [
        make_column("Billing.Client.client_code", "client_code", ["client", "code"], c_ids),
        make_column("Billing.Client.first_name", "first_name", ["first", "name"], first_names),
        make_column("Billing.Client.last_name", "last_name", ["last", "name"], last_names),
        make_column("Billing.Client.contact_email", "contact_email", ["contact", "email"], cust_emails),
        make_column("Billing.Client.telephone", "telephone", ["telephone"], cust_phones),
        make_column("Billing.Client.signup_date", "signup_date", ["signup", "date"], reg_dates),
        make_column("Billing.Client.credit_score", "credit_score", ["credit", "score"], credit_scores),
        make_column("Billing.Client.is_active", "is_active", ["is", "active"], is_active),
    ]
    gt_5 = [
        ("CRM.Customer.cust_id", "Billing.Client.client_code"),
        ("CRM.Customer.fname", "Billing.Client.first_name"),
        ("CRM.Customer.lname", "Billing.Client.last_name"),
        ("CRM.Customer.email_addr", "Billing.Client.contact_email"),
        ("CRM.Customer.phone_num", "Billing.Client.telephone"),
        ("CRM.Customer.created_at", "Billing.Client.signup_date"),
        ("CRM.Customer.credit_rating", "Billing.Client.credit_score"),
        ("CRM.Customer.active_flag", "Billing.Client.is_active"),
    ]
    pairs.append({
        "pair_id": "pair_5",
        "description": "Real-world flavored: Customer vs Client tables (8 columns each)",
        "profiles_a": pair_5_a,
        "profiles_b": pair_5_b,
        "ground_truth": gt_5
    })

    return pairs


def save_pairs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = generate_all_pairs()
    for i, pair in enumerate(pairs, start=1):
        # Save full pair json
        file_path = OUTPUT_DIR / f"pair_{i}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(pair, f, indent=2)

        # Save standalone ground-truth file alongside pair
        gt_path = OUTPUT_DIR / f"pair_{i}_ground_truth.json"
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(pair["ground_truth"], f, indent=2)

        print(f"[OK] Saved {file_path.name} and {gt_path.name} with {len(pair['profiles_a'])}x{len(pair['profiles_b'])} columns")


if __name__ == "__main__":
    save_pairs()
