// evals-integration.ts - Import this in your existing code
import { 
  allEvaluators, 
  runAllEvaluations, 
  logEvaluationsToPhoenix 
} from "./phoenix-evaluations.js";

/**
 * Evaluate any RAG response with all 15 metrics
 */
export async function evaluateWithAllMetrics(data: {
  query: string;
  response: string;
  context: string;
  spanId: string;
  projectName?: string;
}) {
  const { query, response, context, spanId, projectName = "default" } = data;
  
  console.log(`\n🔍 Evaluating with 15 metrics...\n`);
  
  const evaluations = await runAllEvaluations({
    query,
    response,
    context,
    expected: context,
    keywords: extractKeywords(context),
    pattern: "Phoenix|Arize|observability",
    predicted: tokenize(response),
    actual: tokenize(context),
    available_tools: ["search", "retrieve"],
    selected_tool: "retrieve",
    min: 10,
    max: 1000
  });
  
  await logEvaluationsToPhoenix(spanId, evaluations, projectName);
  
  return evaluations;
}

// Helper functions
function extractKeywords(text: string): string[] {
  const important = ["Phoenix", "Arize", "observability", "platform", "open-source", "AI"];
  return important.filter(kw => text.toLowerCase().includes(kw.toLowerCase()));
}

function tokenize(text: string): string[] {
  return text.toLowerCase()
    .replace(/[^\w\s]/g, '')
    .split(/\s+/)
    .filter(w => w.length > 3);
}

// Re-export for convenience
export { allEvaluators, runAllEvaluations, logEvaluationsToPhoenix };