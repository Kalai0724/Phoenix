// evaluators.ts - Complete evaluation suite (FIXED for latest SDK)
import { 
  createFaithfulnessEvaluator,
  createCorrectnessEvaluator,
  createDocumentRelevanceEvaluator,
  createToolSelectionEvaluator,
  createToolInvocationEvaluator,
  createClassificationEvaluator,
  createEvaluator,
  bindEvaluator
} from "@arizeai/phoenix-evals";
import { openai } from "@ai-sdk/openai";
import { PhoenixClient } from "@arizeai/phoenix-client";

// ==========================================
// CONFIGURATION - Connect to your local Phoenix
// ==========================================
const PHOENIX_BASE_URL = "http://localhost:6006"; // Your local Phoenix

// Initialize Phoenix client (FIXED: use PhoenixClient not Client)
export const phoenixClient = new PhoenixClient({
  baseUrl: PHOENIX_BASE_URL,
});

// Initialize LLM for evaluations
const evalLLM = openai("gpt-4o-mini"); // Use gpt-4o-mini for cost savings

// ==========================================
// LLM EVALUATORS (Judge-based)
// ==========================================

export const llmEvaluators = {
  // Modern Evaluators (FIXED: wrap model properly)
  faithfulness: createFaithfulnessEvaluator({ 
    model: evalLLM as any // Type cast to avoid version mismatch
  }),
  correctness: createCorrectnessEvaluator({ 
    model: evalLLM as any 
  }),
  documentRelevance: createDocumentRelevanceEvaluator({ 
    model: evalLLM as any 
  }),
  toolSelection: createToolSelectionEvaluator({ 
    model: evalLLM as any 
  }),
  toolInvocation: createToolInvocationEvaluator({ 
    model: evalLLM as any 
  }),
};

// ==========================================
// CODE EVALUATORS (Deterministic)
// ==========================================

export const codeEvaluators = {
  // Exact Match
  exactMatch: createEvaluator<{ output: string; expected: string }>(
    ({ output, expected }) => output.trim() === expected.trim() ? 1 : 0,
    { name: "exact_match", kind: "CODE", optimizationDirection: "MAXIMIZE" }
  ),

  // Regex Match
  regexMatch: createEvaluator<{ output: string; pattern: string }>(
    ({ output, pattern }) => new RegExp(pattern).test(output) ? 1 : 0,
    { name: "regex_match", kind: "CODE", optimizationDirection: "MAXIMIZE" }
  ),

  // JSON Valid
  jsonValid: createEvaluator<{ output: string }>(
    ({ output }) => {
      try {
        JSON.parse(output);
        return 1;
      } catch {
        return 0;
      }
    },
    { name: "json_valid", kind: "CODE", optimizationDirection: "MAXIMIZE" }
  ),

  // Precision
  precision: createEvaluator<{ predicted: string[]; actual: string[] }>(
    ({ predicted, actual }) => {
      if (predicted.length === 0) return 0;
      const tp = predicted.filter(p => actual.includes(p)).length;
      return tp / predicted.length;
    },
    { name: "precision", kind: "CODE", optimizationDirection: "MAXIMIZE" }
  ),

  // Recall
  recall: createEvaluator<{ predicted: string[]; actual: string[] }>(
    ({ predicted, actual }) => {
      if (actual.length === 0) return 0;
      const tp = predicted.filter(p => actual.includes(p)).length;
      return tp / actual.length;
    },
    { name: "recall", kind: "CODE", optimizationDirection: "MAXIMIZE" }
  ),

  // F1 Score (FIXED: use async evaluate)
  f1Score: createEvaluator<{ predicted: string[]; actual: string[] }>(
    async ({ predicted, actual }) => {
      const pResult = await codeEvaluators.precision.evaluate({ predicted, actual });
      const rResult = await codeEvaluators.recall.evaluate({ predicted, actual });
      const p = pResult.score;
      const r = rResult.score;
      if (p + r === 0) return 0;
      return (2 * p * r) / (p + r);
    },
    { name: "f1_score", kind: "CODE", optimizationDirection: "MAXIMIZE" }
  ),

  // Length Check
  lengthCheck: createEvaluator<{ output: string; min?: number; max?: number }>(
    ({ output, min = 10, max = 1000 }) => {
      const len = output.length;
      return len >= min && len <= max ? 1 : 0;
    },
    { name: "length_check", kind: "CODE", optimizationDirection: "MAXIMIZE" }
  ),

  // Contains Keywords
  containsKeywords: createEvaluator<{ output: string; keywords: string[] }>(
    ({ output, keywords }) => {
      const missing = keywords.filter(kw => !output.toLowerCase().includes(kw.toLowerCase()));
      return 1 - (missing.length / keywords.length);
    },
    { name: "contains_keywords", kind: "CODE", optimizationDirection: "MAXIMIZE" }
  ),
};

// ==========================================
// CUSTOM EVALUATORS (FIXED: choices format)
// ==========================================

// Sentiment Analysis Evaluator
export const sentimentEvaluator = createClassificationEvaluator({
  model: evalLLM as any,
  name: "sentiment",
  promptTemplate: `Analyze the sentiment of: {text}
Respond with EXACTLY ONE of these labels: positive, neutral, negative`,
  choices: {
    positive: 1.0,
    neutral: 0.5,
    negative: 0.0
  },
  optimizationDirection: "MAXIMIZE"
});

// Quality Rating Evaluator (1-5 scale)
export const qualityEvaluator = createClassificationEvaluator({
  model: evalLLM as any,
  name: "quality",
  promptTemplate: `Rate the quality (1-5) of this response to: {query}
Response: {response}
Respond with EXACTLY ONE number: 1, 2, 3, 4, or 5`,
  choices: {
    "1": 0.2,
    "2": 0.4,
    "3": 0.6,
    "4": 0.8,
    "5": 1.0
  },
  optimizationDirection: "MAXIMIZE"
});

// ==========================================
// ALL EVALUATORS COMBINED
// ==========================================

export const allEvaluators = {
  ...llmEvaluators,
  ...codeEvaluators,
  sentiment: sentimentEvaluator,
  quality: qualityEvaluator,
};

// ==========================================
// PHOENIX INTEGRATION FUNCTIONS
// ==========================================

/**
 * Run evaluations on a single example
 */
export async function evaluateAndLog(
  spanId: string,
  input: any,
  output: any,
  context?: string,
  projectName: string = "default"
) {
  const results: Record<string, any> = {};

  // Run LLM evaluators
  if (context) {
    const faithResult = await llmEvaluators.faithfulness.evaluate({
      input: input,
      output: output,
      context: context
    });
    results.faithfulness = faithResult;
  }

  const correctResult = await llmEvaluators.correctness.evaluate({
    input: input,
    output: output
  });
  results.correctness = correctResult;

  // Log to Phoenix as span annotations (using REST API directly)
  await logEvaluationsToPhoenix(spanId, results, projectName);

  return results;
}

/**
 * Log evaluation results to Phoenix traces (FIXED: use REST API)
 */
export async function logEvaluationsToPhoenix(
  spanId: string,
  evaluations: Record<string, any>,
  projectName: string = "default"
) {
  for (const [name, result] of Object.entries(evaluations)) {
    try {
      // Use Phoenix REST API directly
      const response = await fetch(`${PHOENIX_BASE_URL}/v1/span_annotations`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          data: [{
            span_id: spanId,
            name: name,
            annotator_kind: "LLM",
            score: result.score,
            label: result.label,
            explanation: result.explanation,
          }]
        })
      });
      
      if (!response.ok) {
        console.warn(`Failed to log ${name}: ${response.statusText}`);
      }
    } catch (e) {
      console.warn(`Failed to log annotation ${name}:`, e);
    }
  }
}

/**
 * Run all evaluators on a dataset example
 */
export async function runAllEvaluations(data: {
  input: string;
  output: string;
  context?: string;
  expected?: string;
}) {
  const results: Record<string, any> = {};

  // LLM Evaluations
  if (data.context) {
    results.faithfulness = await llmEvaluators.faithfulness.evaluate({
      input: data.input,
      output: data.output,
      context: data.context
    });
  }

  results.correctness = await llmEvaluators.correctness.evaluate({
    input: data.input,
    output: data.output
  });

  // Code Evaluations
  if (data.expected) {
    results.exactMatch = codeEvaluators.exactMatch.evaluate({
      output: data.output,
      expected: data.expected
    });
  }

  results.jsonValid = codeEvaluators.jsonValid.evaluate({
    output: data.output
  });

  results.lengthCheck = codeEvaluators.lengthCheck.evaluate({
    output: data.output
  });

  return results;
}