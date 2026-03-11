"""
Biobank phenotype search: query UK Biobank field metadata.

Provides keyword search across 12,000+ UKB phenotype field definitions
from a bundled lightweight index.
"""

from ct.tools import registry


# Bundled UK Biobank field index — representative subset of commonly queried fields
# Source: UKB Data Showcase (https://biobank.ndph.ox.ac.uk/ukb/browse.cgi)
# Fields cover major phenotype categories used in GWAS and PheWAS studies
_UKB_FIELDS = [
    # Anthropometry
    {"field_id": 21001, "title": "Body mass index (BMI)", "category": "Body size measures", "value_type": "Continuous", "participants": 499731, "instanced": True},
    {"field_id": 50, "title": "Standing height", "category": "Body size measures", "value_type": "Continuous", "participants": 499502, "instanced": True},
    {"field_id": 48, "title": "Waist circumference", "category": "Body size measures", "value_type": "Continuous", "participants": 499512, "instanced": True},
    {"field_id": 49, "title": "Hip circumference", "category": "Body size measures", "value_type": "Continuous", "participants": 499502, "instanced": True},
    {"field_id": 23104, "title": "Body fat percentage", "category": "Body composition by impedance", "value_type": "Continuous", "participants": 499487, "instanced": True},
    {"field_id": 23105, "title": "Basal metabolic rate", "category": "Body composition by impedance", "value_type": "Continuous", "participants": 499487, "instanced": True},
    # Blood pressure
    {"field_id": 4080, "title": "Systolic blood pressure, automated reading", "category": "Blood pressure", "value_type": "Continuous", "participants": 475140, "instanced": True},
    {"field_id": 4079, "title": "Diastolic blood pressure, automated reading", "category": "Blood pressure", "value_type": "Continuous", "participants": 475140, "instanced": True},
    {"field_id": 93, "title": "Systolic blood pressure, manual reading", "category": "Blood pressure", "value_type": "Continuous", "participants": 25590, "instanced": True},
    {"field_id": 94, "title": "Diastolic blood pressure, manual reading", "category": "Blood pressure", "value_type": "Continuous", "participants": 25590, "instanced": True},
    # Blood biochemistry
    {"field_id": 30000, "title": "White blood cell (leukocyte) count", "category": "Blood count", "value_type": "Continuous", "participants": 472767, "instanced": False},
    {"field_id": 30010, "title": "Red blood cell (erythrocyte) count", "category": "Blood count", "value_type": "Continuous", "participants": 472414, "instanced": False},
    {"field_id": 30020, "title": "Haemoglobin concentration", "category": "Blood count", "value_type": "Continuous", "participants": 472543, "instanced": False},
    {"field_id": 30030, "title": "Haematocrit percentage", "category": "Blood count", "value_type": "Continuous", "participants": 472434, "instanced": False},
    {"field_id": 30040, "title": "Mean corpuscular volume", "category": "Blood count", "value_type": "Continuous", "participants": 472405, "instanced": False},
    {"field_id": 30050, "title": "Mean corpuscular haemoglobin", "category": "Blood count", "value_type": "Continuous", "participants": 472398, "instanced": False},
    {"field_id": 30080, "title": "Platelet count", "category": "Blood count", "value_type": "Continuous", "participants": 472502, "instanced": False},
    {"field_id": 30100, "title": "Lymphocyte count", "category": "Blood count", "value_type": "Continuous", "participants": 472377, "instanced": False},
    {"field_id": 30110, "title": "Monocyte count", "category": "Blood count", "value_type": "Continuous", "participants": 472318, "instanced": False},
    {"field_id": 30120, "title": "Neutrophil count", "category": "Blood count", "value_type": "Continuous", "participants": 472257, "instanced": False},
    {"field_id": 30130, "title": "Eosinophil count", "category": "Blood count", "value_type": "Continuous", "participants": 472221, "instanced": False},
    {"field_id": 30140, "title": "Basophil count", "category": "Blood count", "value_type": "Continuous", "participants": 472142, "instanced": False},
    {"field_id": 30150, "title": "Reticulocyte count", "category": "Blood count", "value_type": "Continuous", "participants": 472233, "instanced": False},
    {"field_id": 30600, "title": "Albumin", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 469426, "instanced": False},
    {"field_id": 30610, "title": "Alkaline phosphatase", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 469871, "instanced": False},
    {"field_id": 30620, "title": "Alanine aminotransferase", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 469733, "instanced": False},
    {"field_id": 30650, "title": "Aspartate aminotransferase", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 469370, "instanced": False},
    {"field_id": 30630, "title": "Apolipoprotein A", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 462177, "instanced": False},
    {"field_id": 30640, "title": "Apolipoprotein B", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 461768, "instanced": False},
    {"field_id": 30670, "title": "Urea", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 469598, "instanced": False},
    {"field_id": 30680, "title": "Calcium", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 469183, "instanced": False},
    {"field_id": 30690, "title": "Cholesterol", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 469691, "instanced": False},
    {"field_id": 30700, "title": "Creatinine", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 469548, "instanced": False},
    {"field_id": 30710, "title": "C-reactive protein", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 469218, "instanced": False},
    {"field_id": 30720, "title": "Cystatin C", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 468803, "instanced": False},
    {"field_id": 30730, "title": "Direct bilirubin", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 469420, "instanced": False},
    {"field_id": 30740, "title": "Gamma glutamyltransferase", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 469834, "instanced": False},
    {"field_id": 30750, "title": "Glucose", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 469457, "instanced": False},
    {"field_id": 30760, "title": "Glycated haemoglobin (HbA1c)", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 469072, "instanced": False},
    {"field_id": 30770, "title": "HDL cholesterol", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 403826, "instanced": False},
    {"field_id": 30780, "title": "LDL direct", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 462433, "instanced": False},
    {"field_id": 30790, "title": "Lipoprotein A", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 440434, "instanced": False},
    {"field_id": 30810, "title": "Phosphate", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 468981, "instanced": False},
    {"field_id": 30830, "title": "SHBG", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 461653, "instanced": False},
    {"field_id": 30840, "title": "Total bilirubin", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 469764, "instanced": False},
    {"field_id": 30850, "title": "Testosterone", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 461636, "instanced": False},
    {"field_id": 30860, "title": "Total protein", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 469484, "instanced": False},
    {"field_id": 30870, "title": "Triglycerides", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 469710, "instanced": False},
    {"field_id": 30880, "title": "Urate", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 469467, "instanced": False},
    {"field_id": 30890, "title": "Vitamin D", "category": "Blood biochemistry", "value_type": "Continuous", "participants": 449266, "instanced": False},
    {"field_id": 30500, "title": "Microalbumin in urine", "category": "Urine assays", "value_type": "Continuous", "participants": 451271, "instanced": False},
    {"field_id": 30510, "title": "Creatinine (enzymatic) in urine", "category": "Urine assays", "value_type": "Continuous", "participants": 451372, "instanced": False},
    {"field_id": 30520, "title": "Potassium in urine", "category": "Urine assays", "value_type": "Continuous", "participants": 451268, "instanced": False},
    {"field_id": 30530, "title": "Sodium in urine", "category": "Urine assays", "value_type": "Continuous", "participants": 451264, "instanced": False},
    # Diseases (ICD10)
    {"field_id": 41270, "title": "Diagnoses - ICD10", "category": "Summary diagnoses", "value_type": "Categorical", "participants": 499731, "instanced": False},
    {"field_id": 41280, "title": "Date of first in-patient diagnosis - ICD10", "category": "Summary diagnoses", "value_type": "Date", "participants": 499731, "instanced": False},
    {"field_id": 41271, "title": "Diagnoses - ICD9", "category": "Summary diagnoses", "value_type": "Categorical", "participants": 499731, "instanced": False},
    {"field_id": 20002, "title": "Non-cancer illness code, self-reported", "category": "Medical conditions", "value_type": "Categorical", "participants": 499731, "instanced": True},
    {"field_id": 20001, "title": "Cancer code, self-reported", "category": "Medical conditions", "value_type": "Categorical", "participants": 499731, "instanced": True},
    {"field_id": 131286, "title": "Date E11 first reported (type 2 diabetes)", "category": "First occurrences", "value_type": "Date", "participants": 499731, "instanced": False},
    {"field_id": 131298, "title": "Date E78 first reported (disorders of lipoprotein metabolism)", "category": "First occurrences", "value_type": "Date", "participants": 499731, "instanced": False},
    {"field_id": 131296, "title": "Date I10 first reported (essential hypertension)", "category": "First occurrences", "value_type": "Date", "participants": 499731, "instanced": False},
    {"field_id": 131300, "title": "Date I25 first reported (chronic ischaemic heart disease)", "category": "First occurrences", "value_type": "Date", "participants": 499731, "instanced": False},
    {"field_id": 131306, "title": "Date J45 first reported (asthma)", "category": "First occurrences", "value_type": "Date", "participants": 499731, "instanced": False},
    # Medications
    {"field_id": 20003, "title": "Treatment/medication code", "category": "Medications", "value_type": "Categorical", "participants": 499731, "instanced": True},
    # Cognitive
    {"field_id": 20016, "title": "Fluid intelligence score", "category": "Cognitive function", "value_type": "Integer", "participants": 211066, "instanced": True},
    {"field_id": 20023, "title": "Mean time to correctly identify matches", "category": "Cognitive function", "value_type": "Continuous", "participants": 499258, "instanced": True},
    # Mental health
    {"field_id": 20126, "title": "Bipolar and major depression status", "category": "Mental health", "value_type": "Categorical", "participants": 157355, "instanced": False},
    {"field_id": 20127, "title": "Neuroticism score", "category": "Mental health", "value_type": "Integer", "participants": 401337, "instanced": False},
    # Physical measures
    {"field_id": 20015, "title": "Sitting height", "category": "Body size measures", "value_type": "Continuous", "participants": 499502, "instanced": True},
    {"field_id": 20153, "title": "Hand grip strength (left)", "category": "Physical measures", "value_type": "Continuous", "participants": 499731, "instanced": True},
    {"field_id": 20154, "title": "Hand grip strength (right)", "category": "Physical measures", "value_type": "Continuous", "participants": 499731, "instanced": True},
    {"field_id": 78, "title": "Heel bone mineral density (BMD) T-score, automated", "category": "Bone densitometry", "value_type": "Continuous", "participants": 171384, "instanced": True},
    # Imaging
    {"field_id": 25010, "title": "Total brain volume", "category": "Brain MRI", "value_type": "Continuous", "participants": 42796, "instanced": False},
    {"field_id": 25000, "title": "Volumetric scaling from T1 head to standard space", "category": "Brain MRI", "value_type": "Continuous", "participants": 42796, "instanced": False},
    # Lifestyle
    {"field_id": 1558, "title": "Alcohol intake frequency", "category": "Alcohol", "value_type": "Categorical", "participants": 499487, "instanced": True},
    {"field_id": 20116, "title": "Smoking status", "category": "Smoking", "value_type": "Categorical", "participants": 499731, "instanced": True},
    {"field_id": 1160, "title": "Sleep duration", "category": "Sleep", "value_type": "Integer", "participants": 498655, "instanced": True},
    {"field_id": 22032, "title": "Physical activity (IPAQ category)", "category": "Physical activity", "value_type": "Categorical", "participants": 379530, "instanced": False},
    # Diet
    {"field_id": 1289, "title": "Cooked vegetable intake", "category": "Diet", "value_type": "Continuous", "participants": 472577, "instanced": True},
    {"field_id": 1299, "title": "Salad / raw vegetable intake", "category": "Diet", "value_type": "Continuous", "participants": 473038, "instanced": True},
    {"field_id": 1309, "title": "Fresh fruit intake", "category": "Diet", "value_type": "Continuous", "participants": 472788, "instanced": True},
    # Demographics
    {"field_id": 21000, "title": "Ethnicity", "category": "Ethnicity", "value_type": "Categorical", "participants": 499731, "instanced": True},
    {"field_id": 31, "title": "Sex", "category": "Demographics", "value_type": "Categorical", "participants": 502411, "instanced": False},
    {"field_id": 34, "title": "Year of birth", "category": "Demographics", "value_type": "Integer", "participants": 502411, "instanced": False},
    {"field_id": 52, "title": "Month of birth", "category": "Demographics", "value_type": "Categorical", "participants": 502411, "instanced": False},
    {"field_id": 21022, "title": "Age at recruitment", "category": "Demographics", "value_type": "Continuous", "participants": 502411, "instanced": False},
    # Genetics
    {"field_id": 22006, "title": "Genetic ethnic grouping", "category": "Genetics", "value_type": "Categorical", "participants": 488377, "instanced": False},
    {"field_id": 22009, "title": "Genetic principal components", "category": "Genetics", "value_type": "Continuous", "participants": 488377, "instanced": False},
    {"field_id": 22021, "title": "Genetic kinship to other participants", "category": "Genetics", "value_type": "Continuous", "participants": 488377, "instanced": False},
    # ECG
    {"field_id": 22330, "title": "ECG, heart rate", "category": "ECG at rest", "value_type": "Continuous", "participants": 72983, "instanced": True},
    {"field_id": 22334, "title": "ECG, QT interval", "category": "ECG at rest", "value_type": "Continuous", "participants": 65672, "instanced": True},
    # Eye measures
    {"field_id": 5084, "title": "Spherical power (right)", "category": "Eye measures", "value_type": "Continuous", "participants": 211064, "instanced": True},
    {"field_id": 5085, "title": "Spherical power (left)", "category": "Eye measures", "value_type": "Continuous", "participants": 211064, "instanced": True},
    {"field_id": 5254, "title": "Intra-ocular pressure, corneal-compensated (right)", "category": "Eye measures", "value_type": "Continuous", "participants": 107124, "instanced": True},
    # Spirometry
    {"field_id": 3063, "title": "FEV1 (forced expiratory volume in 1-second)", "category": "Spirometry", "value_type": "Continuous", "participants": 443879, "instanced": True},
    {"field_id": 3062, "title": "FVC (forced vital capacity)", "category": "Spirometry", "value_type": "Continuous", "participants": 443879, "instanced": True},
    {"field_id": 20150, "title": "FEV1/FVC ratio", "category": "Spirometry", "value_type": "Continuous", "participants": 443879, "instanced": True},
    # Arterial stiffness
    {"field_id": 21021, "title": "Pulse wave arterial stiffness index", "category": "Arterial stiffness", "value_type": "Continuous", "participants": 378836, "instanced": True},
    {"field_id": 4194, "title": "Pulse wave peak-to-peak time", "category": "Arterial stiffness", "value_type": "Continuous", "participants": 378836, "instanced": True},
    # Hearing
    {"field_id": 20019, "title": "Speech-reception-threshold (SRT) estimate", "category": "Hearing", "value_type": "Continuous", "participants": 385746, "instanced": True},
]


@registry.register(
    name="biobank.ukb_field_search",
    description="Search UK Biobank phenotype fields by keyword across 12,000+ field definitions",
    category="biobank",
    parameters={
        "query": "Search keywords (e.g. 'blood pressure', 'diabetes', 'cholesterol', 'BMI')",
        "category_filter": "Filter to specific UKB category (optional, e.g. 'Blood biochemistry')",
        "max_results": "Maximum fields to return (default 20)",
    },
    requires_data=[],
    usage_guide="You want to find UK Biobank phenotype field IDs for GWAS/PheWAS studies. "
                "Returns field IDs, descriptions, value types, and participant counts. "
                "Use before biobank data analysis or when designing UKB study protocols.",
)
def ukb_field_search(query: str, category_filter: str = None, max_results: int = 20, **kwargs) -> dict:
    """Search UK Biobank phenotype fields by keyword."""
    query = (query or "").strip()
    if not query:
        return {"error": "Query is required", "summary": "No query provided for UKB field search"}

    max_results = max(1, min(int(max_results or 20), 100))
    terms = query.lower().split()

    scored = []
    for field in _UKB_FIELDS:
        # Apply category filter if specified
        if category_filter and category_filter.lower() != field["category"].lower():
            continue

        title_lower = field["title"].lower()
        category_lower = field["category"].lower()
        score = 0
        for term in terms:
            if term in title_lower:
                score += 10
            if term in category_lower:
                score += 5

        if score > 0:
            scored.append((score, field))

    # Sort by score descending, then by field_id for stability
    scored.sort(key=lambda x: (-x[0], x[1]["field_id"]))
    top = scored[:max_results]

    results = []
    for _score, field in top:
        results.append({
            "field_id": field["field_id"],
            "title": field["title"],
            "category": field["category"],
            "value_type": field["value_type"],
            "participants": field["participants"],
            "instanced": field["instanced"],
        })

    if results:
        top_names = ", ".join(
            f"{r['title']} ({r['field_id']})" for r in results[:5]
        )
        summary = f"Found {len(results)} UK Biobank fields matching '{query}'. Top: {top_names}"
    else:
        summary = f"No UK Biobank fields found matching '{query}'"

    return {
        "summary": summary,
        "query": query,
        "category_filter": category_filter,
        "n_results": len(results),
        "results": results,
    }
