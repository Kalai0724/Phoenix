import os
import phoenix as px
from phoenix.evals import (
    QAEvaluator,
    HallucinationEvaluator,
    RelevanceEvaluator,
    OpenAIModel,
    run_evals,
)

# ================= CONFIG =================
PHOENIX_ENDPOINT = "http://localhost:6006"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ================= CONNECT =================
px_client = px.Client(endpoint=PHOENIX_ENDPOINT)
print(f"✅ Connected to Phoenix at {PHOENIX_ENDPOINT}")

# ================= FETCH SPANS =================
spans_df = px_client.get_spans_dataframe()
spans_df = spans_df[spans_df["span_kind"] == "LLM"]
print(f"📥 Loaded {len(spans_df)} LLM spans")

# 🔍 DEBUG — see real columns (keep this for now)
print("🧾 Available columns:")
for c in spans_df.columns:
    print(" -", c)

# ================= RESOLVE COLUMNS SAFELY =================
def pick_column(candidates):
    for c in candidates:
        if c in spans_df.columns:
            return c
    return None

input_col = pick_column([
    "attributes.metadata.question",
    "attributes.input.value",
    "attributes.input",
])

output_col = pick_column([
    "attributes.output.value",
    "attributes.response",
])

reference_col = pick_column([
    "attributes.metadata.reference_answer",
    "attributes.reference_answer",
])

print("\n✅ Using columns:")
print("input     ->", input_col)
print("output    ->", output_col)
print("reference ->", reference_col)

# ================= BUILD EVAL DF =================
eval_df = spans_df.rename(columns={
    input_col: "input",
    output_col: "output",
    reference_col: "reference",
})

# Drop invalid rows (only if column exists)
required = [c for c in ["input", "output"] if c in eval_df.columns]
eval_df = eval_df.dropna(subset=required)

print("\n🔍 Eval preview:")
print(eval_df[["input", "output", "reference"]].head())
print(f"🧪 Eval-ready rows: {len(eval_df)}")

# ================= EVAL MODEL =================
eval_model = OpenAIModel(
    model="gpt-4o-mini",
    temperature=0,
)

# ================= RUN EVALS =================
if len(eval_df) > 0:
    run_evals(
        dataframe=eval_df,
        evaluators=[
            QAEvaluator(model=eval_model),
            HallucinationEvaluator(model=eval_model),
            RelevanceEvaluator(model=eval_model),
        ],
    )
    print("✅ Phoenix LLM evaluations completed")
else:
    print("⚠️ No eval rows found — check logged columns above")