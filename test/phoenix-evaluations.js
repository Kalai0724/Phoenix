// @ts-check
// phoenix-evaluations.js - Complete evaluation suite for your Phoenix
// Integrates with your existing datasets and experiments

import { GoogleGenerativeAI } from "@google/generative-ai";

// ==========================================
// CONFIGURATION
// ==========================================
const PHOENIX_BASE_URL = "http://localhost:6006";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

if (!GEMINI_API_KEY) {
  console.error("❌ GEMINI_API_KEY environment variable is required!");
  process.exit(1);
}

// Initialize Gemini
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

// Helper to call Gemini
async function callGemini(prompt) {
  try {
    const result = await model.generateContent(prompt);
    return result.response.text().trim();
  } catch (error) {
    console.error("Gemini API error:", error);
    throw error;
  }
}

// ==========================================
// ALL 15 EVALUATORS
// ==========================================

export const allEvaluators = {
  // ========== LLM EVALUATORS (5) ==========
  
  faithfulness: {
    name: "faithfulness",
    description: "Checks if response is grounded in context (no hallucinations)",
    type: "llm",
    async evaluate({ input, output, context }) {
      const prompt = `Evaluate if the response is faithful to the context.
Respond with ONLY: "faithful" or "unfaithful"
Then explain in one sentence.

Context: ${context}
Question: ${input}
Response: ${output}`;

      const response = await callGemini(prompt);
      const lines = response.split('\n').filter(l => l.trim());
      const label = lines[0].toLowerCase().includes("unfaithful") ? "unfaithful" : "faithful";
      
      return {
        score: label === "faithful" ? 1 : 0,
        label: label,
        explanation: lines[1] || "No explanation",
        evaluator: "faithfulness"
      };
    }
  },

  correctness: {
    name: "correctness",
    description: "Checks factual accuracy of response",
    type: "llm",
    async evaluate({ input, output }) {
      const prompt = `Evaluate if the response is factually correct.
Respond with ONLY: "correct" or "incorrect"
Then explain in one sentence.

Question: ${input}
Response: ${output}`;

      const response = await callGemini(prompt);
      const lines = response.split('\n').filter(l => l.trim());
      const label = lines[0].toLowerCase().includes("incorrect") ? "incorrect" : "correct";
      
      return {
        score: label === "correct" ? 1 : 0,
        label: label,
        explanation: lines[1] || "No explanation",
        evaluator: "correctness"
      };
    }
  },

  documentRelevance: {
    name: "document_relevance",
    description: "Checks if retrieved documents are relevant to query",
    type: "llm",
    async evaluate({ input, document }) {
      const prompt = `Evaluate if the document is relevant to the query.
Respond with ONLY: "relevant" or "unrelated"
Then explain in one sentence.

Query: ${input}
Document: ${document}`;

      const response = await callGemini(prompt);
      const lines = response.split('\n').filter(l => l.trim());
      const label = lines[0].toLowerCase().includes("unrelated") ? "unrelated" : "relevant";
      
      return {
        score: label === "relevant" ? 1 : 0,
        label: label,
        explanation: lines[1] || "No explanation",
        evaluator: "document_relevance"
      };
    }
  },

  toolSelection: {
    name: "tool_selection",
    description: "Checks if correct tool was chosen",
    type: "llm",
    async evaluate({ input, available_tools, selected_tool }) {
      const prompt = `Evaluate if the correct tool was selected.
Respond with ONLY: "correct" or "incorrect"
Then explain in one sentence.

Available tools: ${JSON.stringify(available_tools)}
Task: ${input}
Selected tool: ${selected_tool}`;

      const response = await callGemini(prompt);
      const lines = response.split('\n').filter(l => l.trim());
      const label = lines[0].toLowerCase().includes("incorrect") ? "incorrect" : "correct";
      
      return {
        score: label === "correct" ? 1 : 0,
        label: label,
        explanation: lines[1] || "No explanation",
        evaluator: "tool_selection"
      };
    }
  },

  toolInvocation: {
    name: "tool_invocation",
    description: "Checks if tool was called with correct parameters",
    type: "llm",
    async evaluate({ tool_name, tool_input, expected_schema }) {
      const prompt = `Evaluate if the tool was invoked correctly.
Respond with ONLY: "correct" or "incorrect"
Then explain in one sentence.

Tool: ${tool_name}
Expected schema: ${JSON.stringify(expected_schema)}
Actual input: ${JSON.stringify(tool_input)}`;

      const response = await callGemini(prompt);
      const lines = response.split('\n').filter(l => l.trim());
      const label = lines[0].toLowerCase().includes("incorrect") ? "incorrect" : "correct";
      
      return {
        score: label === "correct" ? 1 : 0,
        label: label,
        explanation: lines[1] || "No explanation",
        evaluator: "tool_invocation"
      };
    }
  },

  // ========== CODE EVALUATORS (8) ==========

  exactMatch: {
    name: "exact_match",
    description: "Checks exact string match",
    type: "code",
    evaluate({ output, expected }) {
      const score = output.trim() === expected.trim() ? 1 : 0;
      return {
        score,
        label: score === 1 ? "match" : "mismatch",
        explanation: `Output ${score === 1 ? "matches" : "does not match"} expected value`,
        evaluator: "exact_match"
      };
    }
  },

  regexMatch: {
    name: "regex_match",
    description: "Checks if output matches regex pattern",
    type: "code",
    evaluate({ output, pattern }) {
      const regex = new RegExp(pattern);
      const score = regex.test(output) ? 1 : 0;
      return {
        score,
        label: score === 1 ? "matches" : "no_match",
        explanation: `Pattern ${pattern} ${score === 1 ? "found" : "not found"} in output`,
        evaluator: "regex_match"
      };
    }
  },

  jsonValid: {
    name: "json_valid",
    description: "Checks if output is valid JSON",
    type: "code",
    evaluate({ output }) {
      try {
        JSON.parse(output);
        return {
          score: 1,
          label: "valid",
          explanation: "Valid JSON format",
          evaluator: "json_valid"
        };
      } catch {
        return {
          score: 0,
          label: "invalid",
          explanation: "Invalid JSON format",
          evaluator: "json_valid"
        };
      }
    }
  },

  precision: {
    name: "precision",
    description: "Precision = TP / (TP + FP)",
    type: "code",
    evaluate({ predicted, actual }) {
      if (predicted.length === 0) {
        return { score: 0, label: "0.00", explanation: "No predictions", evaluator: "precision" };
      }
      const truePositives = predicted.filter(p => actual.includes(p)).length;
      const score = truePositives / predicted.length;
      return {
        score,
        label: score.toFixed(2),
        explanation: `${truePositives} true positives out of ${predicted.length} predictions`,
        evaluator: "precision"
      };
    }
  },

  recall: {
    name: "recall",
    description: "Recall = TP / (TP + FN)",
    type: "code",
    evaluate({ predicted, actual }) {
      if (actual.length === 0) {
        return { score: 0, label: "0.00", explanation: "No actual values", evaluator: "recall" };
      }
      const truePositives = predicted.filter(p => actual.includes(p)).length;
      const score = truePositives / actual.length;
      return {
        score,
        label: score.toFixed(2),
        explanation: `${truePositives} true positives out of ${actual.length} actual values`,
        evaluator: "recall"
      };
    }
  },

  f1Score: {
    name: "f1_score",
    description: "F1 = 2 * (precision * recall) / (precision + recall)",
    type: "code",
    evaluate({ predicted, actual }) {
      const precisionResult = allEvaluators.precision.evaluate({ predicted, actual });
      const recallResult = allEvaluators.recall.evaluate({ predicted, actual });
      const p = precisionResult.score;
      const r = recallResult.score;
      
      if (p + r === 0) {
        return { score: 0, label: "0.00", explanation: "No true positives", evaluator: "f1_score" };
      }
      
      const score = (2 * p * r) / (p + r);
      return {
        score,
        label: score.toFixed(2),
        explanation: `F1 score based on P=${p.toFixed(2)}, R=${r.toFixed(2)}`,
        evaluator: "f1_score"
      };
    }
  },

  lengthCheck: {
    name: "length_check",
    description: "Checks if output length is within bounds",
    type: "code",
    evaluate({ output, min = 10, max = 1000 }) {
      const len = output.length;
      const score = (len >= min && len <= max) ? 1 : 0;
      return {
        score,
        label: score === 1 ? "valid" : "invalid",
        explanation: `Length: ${len} characters (required: ${min}-${max})`,
        evaluator: "length_check"
      };
    }
  },

  containsKeywords: {
    name: "contains_keywords",
    description: "Checks if output contains required keywords",
    type: "code",
    evaluate({ output, keywords }) {
      const missing = keywords.filter(kw => !output.toLowerCase().includes(kw.toLowerCase()));
      const score = keywords.length > 0 ? 1 - (missing.length / keywords.length) : 1;
      return {
        score,
        label: `${Math.round(score * 100)}%`,
        explanation: missing.length > 0 ? `Missing: ${missing.join(", ")}` : "All keywords found",
        evaluator: "contains_keywords"
      };
    }
  },

  // ========== CUSTOM EVALUATORS (2) ==========

  sentiment: {
    name: "sentiment",
    description: "Analyzes sentiment of text",
    type: "llm",
    async evaluate({ text }) {
      const prompt = `Analyze the sentiment.
Respond with ONLY: "positive", "neutral", or "negative"
Then explain in one sentence.

Text: "${text}"`;

      const response = await callGemini(prompt);
      const lines = response.split('\n').filter(l => l.trim());
      const result = lines[0].toLowerCase();
      
      let label = "neutral";
      let score = 0.5;
      
      if (result.includes("positive")) {
        label = "positive";
        score = 1.0;
      } else if (result.includes("negative")) {
        label = "negative";
        score = 0.0;
      }
      
      return {
        score,
        label,
        explanation: lines[1] || "No explanation",
        evaluator: "sentiment"
      };
    }
  },

  quality: {
    name: "quality",
    description: "Rates quality on scale 1-5",
    type: "llm",
    async evaluate({ query, response }) {
      const prompt = `Rate the quality of this response (1-5).
Respond with ONLY a number: 1, 2, 3, 4, or 5
Then explain in one sentence.

Query: ${query}
Response: ${response}`;

      const geminiResponse = await callGemini(prompt);
      const lines = geminiResponse.split('\n').filter(l => l.trim());
      const match = lines[0].match(/[1-5]/);
      const rating = match ? parseInt(match[0]) : 3;
      
      const scoreMap = { 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0 };
      
      return {
        score: scoreMap[rating],
        label: `${rating}/5`,
        explanation: lines[1] || "No explanation",
        evaluator: "quality"
      };
    }
  }
};

// ==========================================
// PHOENIX INTEGRATION FUNCTIONS
// ==========================================

/**
 * Log evaluations to Phoenix via REST API
 */
export async function logEvaluationsToPhoenix(spanId, evaluations, projectName = "default") {
  const results = [];
  
  for (const evalResult of evaluations) {
    try {
      const evaluatorObj = allEvaluators[evalResult.evaluator];
      const description = evaluatorObj && evaluatorObj.description ? evaluatorObj.description : "No description available";
      const response = await fetch(`${PHOENIX_BASE_URL}/v1/span_annotations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: [{
            span_id: spanId,
            name: evalResult.evaluator,
            annotator_kind: evalResult.type === 'llm' ? "LLM" : "CODE",
            score: evalResult.score,
            label: evalResult.label,
            explanation: evalResult.explanation,
            metadata: {
              description
            }
          }]
        })
      });
      
      if (!response.ok) {
        console.warn(`⚠️ Failed to log ${evalResult.evaluator}: ${response.statusText}`);
      } else {
        results.push(evalResult);
        console.log(`✅ Logged: ${evalResult.evaluator} = ${evalResult.score.toFixed(2)}`);
      }
    } catch (e) {
      console.warn(`⚠️ Error logging ${evalResult.evaluator}:`, e.message);
    }
  }
  
  return results;
}

/**
 * Run all 15 evaluators on test data
 */
export async function runAllEvaluations(testData) {
  console.log("\n🚀 Running all 15 evaluators...\n");
  
  const results = [];
  const { query, response, context, expected, available_tools, selected_tool, tool_input, expected_schema, text, keywords, pattern, predicted, actual, output, min, max } = testData;

  // 1. Faithfulness
  if (query && response && context) {
    console.log("1️⃣  Faithfulness...");
    results.push(await allEvaluators.faithfulness.evaluate({ input: query, output: response, context }));
  }

  // 2. Correctness
  if (query && response) {
    console.log("2️⃣  Correctness...");
    results.push(await allEvaluators.correctness.evaluate({ input: query, output: response }));
  }

  // 3. Document Relevance
  if (query && context) {
    console.log("3️⃣  Document Relevance...");
    results.push(await allEvaluators.documentRelevance.evaluate({ input: query, document: context }));
  }

  // 4. Tool Selection
  if (query && available_tools && selected_tool) {
    console.log("4️⃣  Tool Selection...");
    results.push(await allEvaluators.toolSelection.evaluate({ input: query, available_tools, selected_tool }));
  }

  // 5. Tool Invocation
  if (tool_input && expected_schema) {
    console.log("5️⃣  Tool Invocation...");
    results.push(await allEvaluators.toolInvocation.evaluate({ tool_name: "test_tool", tool_input, expected_schema }));
  }

  // 6. Exact Match
  if (response && expected) {
    console.log("6️⃣  Exact Match...");
    results.push(allEvaluators.exactMatch.evaluate({ output: response, expected }));
  }

  // 7. Regex Match
  if (response && pattern) {
    console.log("7️⃣  Regex Match...");
    results.push(allEvaluators.regexMatch.evaluate({ output: response, pattern }));
  }

  // 8. JSON Valid
  if (output || response) {
    console.log("8️⃣  JSON Valid...");
    results.push(allEvaluators.jsonValid.evaluate({ output: output || response }));
  }

  // 9. Precision
  if (predicted && actual) {
    console.log("9️⃣  Precision...");
    results.push(allEvaluators.precision.evaluate({ predicted, actual }));
  }

  // 10. Recall
  if (predicted && actual) {
    console.log("🔟 Recall...");
    results.push(allEvaluators.recall.evaluate({ predicted, actual }));
  }

  // 11. F1 Score
  if (predicted && actual) {
    console.log("1️⃣1️⃣ F1 Score...");
    results.push(allEvaluators.f1Score.evaluate({ predicted, actual }));
  }

  // 12. Length Check
  if (response) {
    console.log("1️⃣2️⃣ Length Check...");
    results.push(allEvaluators.lengthCheck.evaluate({ output: response, min: min || 10, max: max || 1000 }));
  }

  // 13. Contains Keywords
  if (response && keywords) {
    console.log("1️⃣3️⃣ Contains Keywords...");
    results.push(allEvaluators.containsKeywords.evaluate({ output: response, keywords }));
  }

  // 14. Sentiment
  if (text || response) {
    console.log("1️⃣4️⃣ Sentiment...");
    results.push(await allEvaluators.sentiment.evaluate({ text: text || response }));
  }

  // 15. Quality
  if (query && response) {
    console.log("1️⃣5️⃣ Quality...");
    results.push(await allEvaluators.quality.evaluate({ query, response }));
  }

  console.log(`\n✅ Completed ${results.length} evaluations\n`);
  return results;
}

/**
 * Create dataset in Phoenix with evaluations
 */
export async function createDatasetWithEvaluations(name, examples, projectName = "default") {
  console.log(`\n📊 Creating dataset: ${name}`);
  
  try {
    const response = await fetch(`${PHOENIX_BASE_URL}/v1/datasets`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: name,
        description: "Dataset with all 15 evaluation metrics",
        project_name: projectName,
        examples: examples.map((ex, idx) => ({
          input: ex.input,
          output: ex.output,
          expected: ex.expected,
          metadata: {
            example_id: `ex-${idx}`,
            ...ex.metadata
          }
        }))
      })
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    
    const data = await response.json();
    console.log(`✅ Dataset created: ${data.data.id}`);
    return data.data;
  } catch (e) {
    console.error("❌ Failed to create dataset:", e.message);
    throw e;
  }
}

/**
 * Run experiment with all evaluators
 */
export async function runExperimentWithEvaluations(datasetId, taskFn, projectName = "default") {
  console.log(`\n🔬 Running experiment on dataset: ${datasetId}`);
  
  // This would integrate with your existing experiment runner
  // For now, we'll just show how to use the evaluators
  
  console.log("✅ Experiment setup complete");
  console.log("Use the evaluators in your task function like this:");
  console.log(`
    const result = await taskFn(example);
    const evals = await runAllEvaluations({
      query: example.input,
      response: result,
      context: example.context,
      // ... other fields
    });
    await logEvaluationsToPhoenix(spanId, evals, projectName);
  `);
}

// ==========================================
// MAIN TEST FUNCTION
// ==========================================

// ==========================================
// MAIN TEST FUNCTION (FIXED)
// ==========================================

export async function testAllEvaluators() {
  console.log("╔════════════════════════════════════════════════════════╗");
  console.log("║     PHOENIX COMPLETE EVALUATION SUITE (15 Metrics)     ║");
  console.log("╚════════════════════════════════════════════════════════╝\n");

  const testData = {
    query: "What is Phoenix?",
    response: "Phoenix is an open-source AI observability platform by Arize AI.",
    context: "Phoenix is an open-source AI observability platform by Arize AI.",
    expected: "Phoenix is an open-source AI observability platform by Arize AI.",
    available_tools: ["search", "calculator", "weather"],
    selected_tool: "search",
    tool_input: { query: "Phoenix AI" },
    expected_schema: { query: "string" },
    keywords: ["Phoenix", "Arize", "observability"],
    pattern: "Phoenix.*Arize",
    predicted: ["Phoenix", "AI", "platform"],
    actual: ["Phoenix", "platform", "observability"],
    min: 10,
    max: 200
  };

  // Run all evaluations
  const results = await runAllEvaluations(testData);

  // Display summary (FIXED)
  console.log("══════════════════════════════════════════════════════════");
  console.log("📊 EVALUATION SUMMARY");
  console.log("══════════════════════════════════════════════════════════");
  
  // FIX: Check if evaluator exists before accessing type
  const llmEvals = results.filter(r => {
    const evaluator = allEvaluators[r.evaluator];
    return evaluator && evaluator.type === 'llm';
  });
  
  const codeEvals = results.filter(r => {
    const evaluator = allEvaluators[r.evaluator];
    return evaluator && evaluator.type === 'code';
  });
  
  console.log(`\n🤖 LLM Evaluations: ${llmEvals.length}`);
  llmEvals.forEach(r => {
    const icon = r.score >= 0.8 ? "✅" : r.score >= 0.5 ? "⚠️" : "❌";
    console.log(`   ${icon} ${r.evaluator}: ${r.score.toFixed(2)} (${r.label})`);
  });
  
  console.log(`\n💻 Code Evaluations: ${codeEvals.length}`);
  codeEvals.forEach(r => {
    const icon = r.score >= 0.8 ? "✅" : r.score >= 0.5 ? "⚠️" : "❌";
    console.log(`   ${icon} ${r.evaluator}: ${r.score.toFixed(2)} (${r.label})`);
  });
  
  const avgScore = results.reduce((a, b) => a + b.score, 0) / results.length;
  console.log(`\n📈 Average Score: ${avgScore.toFixed(2)}`);
  console.log(`🎯 Passed (≥0.8): ${results.filter(r => r.score >= 0.8).length}/${results.length}`);

  // Try to log to Phoenix
  console.log("\n══════════════════════════════════════════════════════════");
  console.log("📝 Logging to Phoenix...");
  console.log("══════════════════════════════════════════════════════════");
  
  const spanId = `eval-test-${Date.now()}`;
  const logged = await logEvaluationsToPhoenix(spanId, results, "default");
  
  console.log(`\n✅ Logged ${logged.length}/${results.length} evaluations`);
  console.log(`🌐 View at: http://localhost:6006/projects/default`);
  console.log(`   (Look for span: ${spanId})`);

  return results;
}