// evals-test.ts - Your existing file with ALL evaluations added

import { trace } from "@opentelemetry/api";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { 
  allEvaluators, 
  runAllEvaluations, 
  logEvaluationsToPhoenix 
} from "./phoenix-evaluations.js";

// ==========================================
// YOUR EXISTING CONFIGURATION
// ==========================================
const CONFIG = {
  model: "gemini-2.5-flash",
  datasetName: `rag-evaluation-${Date.now()}`,
  experimentName: `rag-experiment-${Date.now()}`,
};

// Initialize Gemini (your existing code)
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);
const model = genAI.getGenerativeModel({ model: CONFIG.model });

// Get tracer (your existing code)
const tracer = trace.getTracer("rag-application");

// Your existing knowledge base
const KNOWLEDGE_BASE: Record<string, string> = {
  "What is Phoenix?": "Phoenix is an open-source AI observability platform by Arize AI.",
  "How to install Phoenix?": "You can install Phoenix using pip: pip install arize-phoenix",
  // ... add more
};

// ==========================================
// NEW: FULL EVALUATION FUNCTION
// ==========================================

async function evaluateRAGResponse(query: string, response: string, context: string, spanId: string) {
  console.log(`\n🔍 Running all 15 evaluators for span: ${spanId}\n`);
  
  // Run ALL 15 evaluators
  const evaluations = await runAllEvaluations({
    query,
    response,
    context,
    expected: context,
    keywords: ["Phoenix", "Arize", "observability", "open-source"],
    pattern: "Phoenix.*Arize|observability.*platform",
    predicted: response.toLowerCase().split(/\s+/).filter(w => w.length > 3),
    actual: context.toLowerCase().split(/\s+/).filter(w => w.length > 3),
    available_tools: ["search", "retrieve", "calculate"],
    selected_tool: "retrieve",
    min: 20,
    max: 500
  });
  const evaluations
  // Log to Phoenix
  await logEvaluationsToPhoenix(spanId, evaluations, "default");
  
  // Display results
  console.log("\n📊 Evaluation Results:");
  evaluations.forEach(evalResult => {
    const icon = evalResult.score >= 0.8 ? "✅" : evalResult.score >= 0.5 ? "⚠️" : "❌";
    console.log(`${icon} ${evalResult.evaluator}: ${evalResult.score.toFixed(2)} (${evalResult.label})`);
  });
  
  return evaluations;
}

// ==========================================
// YOUR EXISTING RAG FUNCTION (modified)
// ==========================================

async function runRAG(query: string) {
  // Start a span for tracing
  return await tracer.startActiveSpan("rag-query", async (span) => {
    const spanId = span.spanContext().spanId;
    
    console.log(`\n📝 Query: ${query}`);
    
    // Your existing RAG logic
    const context = KNOWLEDGE_BASE[query] || "No relevant information found.";
    
    // Generate response using Gemini
    const prompt = `Answer the question based on the context.
Context: ${context}
Question: ${query}
Answer:`;
    
    const result = await model.generateContent(prompt);
    const response = result.response.text().trim();
    
    console.log(`💬 Response: ${response}`);
    
    // ==========================================
    // ADD EVALUATIONS HERE
    // ==========================================
    const evaluations = await evaluateRAGResponse(query, response, context, spanId);
    
    // Add evaluations to span attributes
    evaluations.forEach(evalResult => {
      span.setAttribute(`eval.${evalResult.evaluator}.score`, evalResult.score);
      span.setAttribute(`eval.${evalResult.evaluator}.label`, evalResult.label);
    });
    
    span.end();
    
    return { query, response, context, evaluations };
  });
}

// ==========================================
// MAIN - Run multiple test queries
// ==========================================

async function main() {
  console.log("╔════════════════════════════════════════════════════════╗");
  console.log("║     RAG SYSTEM WITH FULL PHOENIX EVALUATIONS          ║");
  console.log("╚════════════════════════════════════════════════════════╝\n");
  
  const testQueries = [
    "What is Phoenix?",
    "How to install Phoenix?",
    // Add more test queries
  ];
  
  for (const query of testQueries) {
    await runRAG(query);
  }
  
  console.log("\n✅ All evaluations complete!");
  console.log("🌐 View results at: http://localhost:6006/projects/default");
}

main().catch(console.error);