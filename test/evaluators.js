// @ts-check
// evaluators.js - Complete evaluation suite using GEMINI + Local Dashboard

import { 
  createEvaluator,
} from "@arizeai/phoenix-evals";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { join } from 'path';

// ==========================================
// CONFIGURATION - Using GEMINI directly
// ==========================================
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

if (!GEMINI_API_KEY) {
  console.error("❌ GEMINI_API_KEY environment variable is required!");
  console.error("Set it with: $env:GEMINI_API_KEY='your-key' (PowerShell)");
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
// LLM EVALUATORS (Direct Gemini)
// ==========================================

export const llmEvaluators = {
  faithfulness: {
    name: "faithfulness",
    async evaluate({ input, output, context }) {
      const prompt = `You are evaluating if a response is faithful to the provided context.
A response is "faithful" if it only contains information present in the context.
A response is "unfaithful" if it contains information not in the context (hallucination).

Context: ${context}

Question: ${input}

Response: ${output}

Respond with EXACTLY ONE word: "faithful" or "unfaithful"
Then on a new line, briefly explain your reasoning in 1 sentence.`;

      const response = await callGemini(prompt);
      const lines = response.split('\n').filter(l => l.trim());
      const label = lines[0].toLowerCase().includes("unfaithful") ? "unfaithful" : "faithful";
      const explanation = lines[1] || "No explanation provided";
      
      return [{
        score: label === "faithful" ? 1 : 0,
        label: label,
        explanation: explanation,
        name: "faithfulness"
      }];
    }
  },

  correctness: {
    name: "correctness",
    async evaluate({ input, output }) {
      const prompt = `You are evaluating if a response is factually correct and complete.
A response is "correct" if it is factually accurate and addresses the question.
A response is "incorrect" if it contains factual errors or is incomplete.

Question: ${input}

Response: ${output}

Respond with EXACTLY ONE word: "correct" or "incorrect"
Then on a new line, briefly explain your reasoning in 1 sentence.`;

      const response = await callGemini(prompt);
      const lines = response.split('\n').filter(l => l.trim());
      const label = lines[0].toLowerCase().includes("incorrect") ? "incorrect" : "correct";
      const explanation = lines[1] || "No explanation provided";
      
      return [{
        score: label === "correct" ? 1 : 0,
        label: label,
        explanation: explanation,
        name: "correctness"
      }];
    }
  },

  documentRelevance: {
    name: "document_relevance",
    async evaluate({ input, document_text }) {
      const prompt = `You are evaluating if a document is relevant to a query.
A document is "relevant" if it contains information that helps answer the query.
A document is "unrelated" if it does not help answer the query.

Query: ${input}

Document: ${document_text}

Respond with EXACTLY ONE word: "relevant" or "unrelated"
Then on a new line, briefly explain your reasoning in 1 sentence.`;

      const response = await callGemini(prompt);
      const lines = response.split('\n').filter(l => l.trim());
      const label = lines[0].toLowerCase().includes("unrelated") ? "unrelated" : "relevant";
      const explanation = lines[1] || "No explanation provided";
      
      return [{
        score: label === "relevant" ? 1 : 0,
        label: label,
        explanation: explanation,
        name: "document_relevance"
      }];
    }
  },

  toolSelection: {
    name: "tool_selection",
    async evaluate({ input, available_tools, selected_tool }) {
      const prompt = `You are evaluating if the correct tool was selected for a task.
Available tools: ${available_tools}

Task: ${input}

Selected tool: ${selected_tool}

Respond with EXACTLY ONE word: "correct" or "incorrect"
Then on a new line, briefly explain your reasoning in 1 sentence.`;

      const response = await callGemini(prompt);
      const lines = response.split('\n').filter(l => l.trim());
      const label = lines[0].toLowerCase().includes("incorrect") ? "incorrect" : "correct";
      const explanation = lines[1] || "No explanation provided";
      
      return [{
        score: label === "correct" ? 1 : 0,
        label: label,
        explanation: explanation,
        name: "tool_selection"
      }];
    }
  },

  toolInvocation: {
    name: "tool_invocation",
    async evaluate({ tool_name, tool_input, expected_schema }) {
      const prompt = `You are evaluating if a tool was invoked correctly.
Tool: ${tool_name}
Expected schema: ${expected_schema}
Actual input: ${tool_input}

Respond with EXACTLY ONE word: "correct" or "incorrect"
Then on a new line, briefly explain your reasoning in 1 sentence.`;

      const response = await callGemini(prompt);
      const lines = response.split('\n').filter(l => l.trim());
      const label = lines[0].toLowerCase().includes("incorrect") ? "incorrect" : "correct";
      const explanation = lines[1] || "No explanation provided";
      
      return [{
        score: label === "correct" ? 1 : 0,
        label: label,
        explanation: explanation,
        name: "tool_invocation"
      }];
    }
  },
};

// ==========================================
// CODE EVALUATORS (Phoenix evals)
// ==========================================

export const codeEvaluators = {
  exactMatch: createEvaluator(
    ({ output, expected }) => output.trim() === expected.trim() ? 1 : 0,
    { name: "exact_match", kind: "CODE", optimizationDirection: "MAXIMIZE" }
  ),

  regexMatch: createEvaluator(
    ({ output, pattern }) => new RegExp(pattern).test(output) ? 1 : 0,
    { name: "regex_match", kind: "CODE", optimizationDirection: "MAXIMIZE" }
  ),

  jsonValid: createEvaluator(
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

  precision: createEvaluator(
    ({ predicted, actual }) => {
      if (predicted.length === 0) return 0;
      const tp = predicted.filter(p => actual.includes(p)).length;
      return tp / predicted.length;
    },
    { name: "precision", kind: "CODE", optimizationDirection: "MAXIMIZE" }
  ),

  recall: createEvaluator(
    ({ predicted, actual }) => {
      if (actual.length === 0) return 0;
      const tp = predicted.filter(p => actual.includes(p)).length;
      return tp / actual.length;
    },
    { name: "recall", kind: "CODE", optimizationDirection: "MAXIMIZE" }
  ),

  f1Score: createEvaluator(
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

  lengthCheck: createEvaluator(
    ({ output, min = 10, max = 1000 }) => {
      const len = output.length;
      return len >= min && len <= max ? 1 : 0;
    },
    { name: "length_check", kind: "CODE", optimizationDirection: "MAXIMIZE" }
  ),

  containsKeywords: createEvaluator(
    ({ output, keywords }) => {
      const missing = keywords.filter(kw => !output.toLowerCase().includes(kw.toLowerCase()));
      return 1 - (missing.length / keywords.length);
    },
    { name: "contains_keywords", kind: "CODE", optimizationDirection: "MAXIMIZE" }
  ),
};

// ==========================================
// CUSTOM EVALUATORS
// ==========================================

export const sentimentEvaluator = {
  name: "sentiment",
  async evaluate({ text }) {
    const prompt = `Analyze the sentiment of the following text.
Text: "${text}"

Respond with EXACTLY ONE word: "positive", "neutral", or "negative"
Then on a new line, briefly explain why.`;

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
    
    return [{
      score: score,
      label: label,
      explanation: lines[1] || "No explanation",
      name: "sentiment"
    }];
  }
};

export const qualityEvaluator = {
  name: "quality",
  async evaluate({ query, response }) {
    const prompt = `Rate the quality of this customer service response on a scale of 1-5.
Query: ${query}
Response: ${response}

Respond with EXACTLY ONE number: 1, 2, 3, 4, or 5
Then on a new line, briefly explain your rating.`;

    const geminiResponse = await callGemini(prompt);
    const lines = geminiResponse.split('\n').filter(l => l.trim());
    const rating = parseInt(lines[0].match(/\d+/)?.[0] || "3");
    
    const scoreMap = { 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0 };
    
    return [{
      score: scoreMap[rating] || 0.6,
      label: String(rating),
      explanation: lines[1] || "No explanation",
      name: "quality"
    }];
  }
};

// ==========================================
// SAVE RESULTS TO LOCAL HTML DASHBOARD
// ==========================================

function generateDashboard(results, timestamp) {
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phoenix Evaluation Results</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .subtitle {
            color: rgba(255,255,255,0.9);
            text-align: center;
            margin-bottom: 40px;
            font-size: 1.1em;
        }
        .timestamp {
            color: rgba(255,255,255,0.7);
            text-align: center;
            margin-bottom: 30px;
            font-size: 0.9em;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }
        .card {
            background: white;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .card-title {
            font-size: 1.2em;
            font-weight: 600;
            color: #333;
        }
        .badge {
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            text-transform: uppercase;
        }
        .badge-llm {
            background: #e3f2fd;
            color: #1976d2;
        }
        .badge-code {
            background: #e8f5e9;
            color: #388e3c;
        }
        .score {
            font-size: 2.5em;
            font-weight: 700;
            margin: 15px 0;
        }
        .score-good {
            color: #4caf50;
        }
        .score-bad {
            color: #f44336;
        }
        .score-mid {
            color: #ff9800;
        }
        .label {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: 600;
            margin-bottom: 10px;
        }
        .label-success {
            background: #e8f5e9;
            color: #2e7d32;
        }
        .label-fail {
            background: #ffebee;
            color: #c62828;
        }
        .label-warn {
            background: #fff3e0;
            color: #ef6c00;
        }
        .explanation {
            color: #666;
            font-size: 0.95em;
            line-height: 1.5;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #eee;
        }
        .summary {
            background: white;
            border-radius: 16px;
            padding: 30px;
            margin-top: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }
        .summary h2 {
            color: #333;
            margin-bottom: 20px;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        .summary-item {
            text-align: center;
            padding: 20px;
            background: #f5f5f5;
            border-radius: 12px;
        }
        .summary-value {
            font-size: 2em;
            font-weight: 700;
            color: #667eea;
        }
        .summary-label {
            color: #666;
            margin-top: 5px;
        }
        .test-data {
            background: white;
            border-radius: 16px;
            padding: 24px;
            margin-top: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }
        .test-data h3 {
            color: #333;
            margin-bottom: 15px;
        }
        .test-data pre {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Phoenix Evaluation Results</h1>
        <p class="subtitle">Complete evaluation suite powered by Gemini</p>
        <p class="timestamp">Generated: ${timestamp}</p>
        
        <div class="grid">
            ${results.map(r => {
                const scoreClass = r.score >= 0.8 ? 'score-good' : r.score >= 0.5 ? 'score-mid' : 'score-bad';
                const labelClass = r.score >= 0.8 ? 'label-success' : r.score >= 0.5 ? 'label-warn' : 'label-fail';
                const badgeClass = r.type === 'llm' ? 'badge-llm' : 'badge-code';
                const typeLabel = r.type === 'llm' ? 'LLM Eval' : 'Code Eval';
                
                return `
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">${r.name}</span>
                        <span class="badge ${badgeClass}">${typeLabel}</span>
                    </div>
                    <div class="score ${scoreClass}">${r.score.toFixed(2)}</div>
                    <span class="label ${labelClass}">${r.label}</span>
                    ${r.explanation ? `<div class="explanation">${r.explanation}</div>` : ''}
                </div>
                `;
            }).join('')}
        </div>

        <div class="summary">
            <h2>📊 Summary Statistics</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-value">${results.length}</div>
                    <div class="summary-label">Total Evaluations</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value">${results.filter(r => r.score >= 0.8).length}</div>
                    <div class="summary-label">Passed (≥0.8)</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value">${results.filter(r => r.type === 'llm').length}</div>
                    <div class="summary-label">LLM Evaluations</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value">${results.filter(r => r.type === 'code').length}</div>
                    <div class="summary-label">Code Evaluations</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value">${(results.reduce((a, b) => a + b.score, 0) / results.length).toFixed(2)}</div>
                    <div class="summary-label">Average Score</div>
                </div>
            </div>
        </div>

        <div class="test-data">
            <h3>📝 Test Data</h3>
            <pre>${JSON.stringify({
                query: "What is Phoenix?",
                response: "Phoenix is an open-source AI observability platform by Arize AI.",
                context: "Phoenix is an open-source AI observability platform by Arize AI."
            }, null, 2)}</pre>
        </div>
    </div>
</body>
</html>`;

  return html;
}

// ==========================================
// TEST FUNCTION
// ==========================================

export async function testAllEvaluators() {
  console.log("🚀 Testing all Phoenix evaluators with GEMINI...\n");

  const testData = {
    query: "What is Phoenix?",
    response: "Phoenix is an open-source AI observability platform by Arize AI.",
    context: "Phoenix is an open-source AI observability platform by Arize AI."
  };

  const allResults = [];
  const timestamp = new Date().toLocaleString();

  console.log("1️⃣  FAITHFULNESS (Gemini-powered):");
  const f = await llmEvaluators.faithfulness.evaluate({
    input: testData.query,
    output: testData.response,
    context: testData.context
  });
  allResults.push({ ...f[0], type: 'llm' });
  console.log(`   Score: ${f[0].score}, Label: ${f[0].label}`);

  console.log("2️⃣  CORRECTNESS (Gemini-powered):");
  const c = await llmEvaluators.correctness.evaluate({
    input: testData.query,
    output: testData.response
  });
  allResults.push({ ...c[0], type: 'llm' });
  console.log(`   Score: ${c[0].score}, Label: ${c[0].label}`);

  console.log("3️⃣  DOCUMENT RELEVANCE (Gemini-powered):");
  const dr = await llmEvaluators.documentRelevance.evaluate({
    input: testData.query,
    document_text: testData.context
  });
  allResults.push({ ...dr[0], type: 'llm' });
  console.log(`   Score: ${dr[0].score}, Label: ${dr[0].label}`);

  console.log("4️⃣  EXACT MATCH (Code):");
  const em = codeEvaluators.exactMatch.evaluate({
    output: testData.response,
    expected: testData.context
  });
  allResults.push({ name: 'Exact Match', score: em.score, label: em.score === 1 ? 'match' : 'mismatch', explanation: 'String comparison', type: 'code' });
  console.log(`   Score: ${em.score}`);

  console.log("5️⃣  JSON VALID (Code):");
  const jv = codeEvaluators.jsonValid.evaluate({
    output: '{"valid": true}'
  });
  allResults.push({ name: 'JSON Valid', score: jv.score, label: jv.score === 1 ? 'valid' : 'invalid', explanation: 'Syntax check', type: 'code' });
  console.log(`   Score: ${jv.score}`);

  console.log("6️⃣  LENGTH CHECK (Code):");
  const lc = codeEvaluators.lengthCheck.evaluate({
    output: testData.response,
    min: 10,
    max: 200
  });
  allResults.push({ name: 'Length Check', score: lc.score, label: lc.score === 1 ? 'valid' : 'invalid', explanation: 'Character count check', type: 'code' });
  console.log(`   Score: ${lc.score}`);

  console.log("7️⃣  CONTAINS KEYWORDS (Code):");
  const kw = codeEvaluators.containsKeywords.evaluate({
    output: testData.response,
    keywords: ["Phoenix", "Arize", "observability"]
  });
  allResults.push({ name: 'Contains Keywords', score: kw.score, label: `${Math.round(kw.score * 100)}% match`, explanation: 'Keyword matching', type: 'code' });
  console.log(`   Score: ${kw.score}`);

  console.log("8️⃣  SENTIMENT (Gemini-powered):");
  const s = await sentimentEvaluator.evaluate({
    text: testData.response
  });
  allResults.push({ ...s[0], type: 'llm' });
  console.log(`   Score: ${s[0].score}, Label: ${s[0].label}`);

  console.log("9️⃣  QUALITY (Gemini-powered):");
  const q = await qualityEvaluator.evaluate({
    query: testData.query,
    response: testData.response
  });
  allResults.push({ ...q[0], type: 'llm' });
  console.log(`   Score: ${q[0].score}, Label: ${q[0].label}`);

  // Generate and save dashboard
  console.log("\n📊 Generating HTML dashboard...");
  const dashboardHtml = generateDashboard(allResults, timestamp);
  
  // Create output directory if not exists
  if (!existsSync('output')) {
    mkdirSync('output');
  }
  
  const outputPath = join(process.cwd(), 'output', 'evaluation-results.html');
  writeFileSync(outputPath, dashboardHtml);
  
  console.log(`\n✅ All evaluations completed!`);
  console.log(`\n📁 Results saved to: ${outputPath}`);
  console.log(`🌐 Open this file in your browser to view the dashboard\n`);
  
  // Also save JSON for programmatic access
  const jsonPath = join(process.cwd(), 'output', 'evaluation-results.json');
  writeFileSync(jsonPath, JSON.stringify({
    timestamp,
    testData,
    results: allResults,
    summary: {
      total: allResults.length,
      passed: allResults.filter(r => r.score >= 0.8).length,
      llmEvals: allResults.filter(r => r.type === 'llm').length,
      codeEvals: allResults.filter(r => r.type === 'code').length,
      averageScore: allResults.reduce((a, b) => a + b.score, 0) / allResults.length
    }
  }, null, 2));
  
  console.log(`📄 JSON data saved to: ${jsonPath}`);
  
  // Try to open browser
  try {
    const { exec } = await import('child_process');
    const url = `file://${outputPath}`;
    
    console.log(`\n🚀 Opening browser...`);
    
    if (process.platform === 'win32') {
      exec(`start "" "${outputPath}"`);
    } else if (process.platform === 'darwin') {
      exec(`open "${outputPath}"`);
    } else {
      exec(`xdg-open "${outputPath}"`);
    }
  } catch (e) {
    console.log(`   (Could not open browser automatically)`);
  }
  
  return allResults;
}