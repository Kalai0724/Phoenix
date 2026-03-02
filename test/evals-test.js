// evals-test.ts - Updated with all evaluation types
import { trace } from "@opentelemetry/api";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { llmEvaluators, codeEvaluators, sentimentEvaluator, qualityEvaluator, evaluateAndLog, runAllEvaluations, phoenixClient } from "./evaluators";
// Your existing configuration
const CONFIG = {
    model: "gemini-2.5-flash",
    datasetName: `rag-evaluation-${Date.now()}`,
    experimentName: `rag-experiment-${Date.now()}`,
};
// Initialize Gemini (your existing code)
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: CONFIG.model });
// Get tracer (your existing code)
const tracer = trace.getTracer("rag-application");
// Your existing knowledge base
const KNOWLEDGE_BASE = {
    "What is Phoenix?": "Phoenix is an open-source AI observability platform by Arize AI.",
    // ... add more
};
// ==========================================
// EXAMPLE: Evaluate a single response
// ==========================================
async function evaluateSingleResponse() {
    const query = "What is Phoenix?";
    const response = "Phoenix is an AI observability platform created by Arize AI for monitoring LLM applications.";
    const context = KNOWLEDGE_BASE[query];
    console.log("=== Running All Evaluations ===\n");
    // 1. LLM Evaluations
    console.log("1. FAITHFULNESS (checks if response matches context):");
    const faithResult = await llmEvaluators.faithfulness.evaluate({
        input: query,
        output: response,
        context: context
    });
    console.log(`   Score: ${faithResult.score}, Label: ${faithResult.label}`);
    console.log(`   Explanation: ${faithResult.explanation}\n`);
    console.log("2. CORRECTNESS (checks factual accuracy):");
    const correctResult = await llmEvaluators.correctness.evaluate({
        input: query,
        output: response
    });
    console.log(`   Score: ${correctResult.score}, Label: ${correctResult.label}`);
    console.log(`   Explanation: ${correctResult.explanation}\n`);
    // 2. Code Evaluations
    console.log("3. EXACT MATCH:");
    const exactResult = codeEvaluators.exactMatch.evaluate({
        output: response,
        expected: context
    });
    console.log(`   Score: ${exactResult.score}\n`);
    console.log("4. LENGTH CHECK:");
    const lenResult = codeEvaluators.lengthCheck.evaluate({
        output: response,
        min: 10,
        max: 200
    });
    console.log(`   Score: ${lenResult.score}\n`);
    console.log("5. CONTAINS KEYWORDS:");
    const kwResult = codeEvaluators.containsKeywords.evaluate({
        output: response,
        keywords: ["Phoenix", "Arize", "observability"]
    });
    console.log(`   Score: ${kwResult.score}\n`);
    // 3. Custom Evaluations
    console.log("6. SENTIMENT ANALYSIS:");
    const sentResult = await sentimentEvaluator.evaluate({
        text: response
    });
    console.log(`   Score: ${sentResult.score}, Label: ${sentResult.label}\n`);
    console.log("7. QUALITY RATING:");
    const qualResult = await qualityEvaluator.evaluate({
        query: query,
        response: response
    });
    console.log(`   Score: ${qualResult.score}, Label: ${qualResult.label}\n`);
    // 4. Run all at once
    console.log("=== Running All Evaluations at Once ===");
    const allResults = await runAllEvaluations({
        input: query,
        output: response,
        context: context,
        expected: context
    });
    console.log("\nAll Results:", JSON.stringify(allResults, null, 2));
}
// ==========================================
// EXAMPLE: Evaluate traces from Phoenix
// ==========================================
async function evaluateExistingTraces(projectName = "default") {
    console.log(`\n=== Evaluating Traces from Project: ${projectName} ===\n`);
    // Get traces from Phoenix
    const traces = await phoenixClient.spans.getSpans({
        projectName: projectName,
        limit: 10
    });
    for (const span of traces) {
        const input = span.attributes?.["llm.input_messages"];
        const output = span.attributes?.["llm.output_messages"];
        const context = span.attributes?.["retrieval.documents"];
        if (input && output) {
            console.log(`Evaluating span ${span.span_id}...`);
            const results = await evaluateAndLog(span.span_id, input, output, context, projectName);
            console.log("Results:", results);
        }
    }
}
// ==========================================
// EXAMPLE: Run experiment with all evaluators
// ==========================================
import { runExperiment } from "@arizeai/phoenix-client/experiments";
async function runFullExperiment() {
    console.log("\n=== Running Full Experiment ===\n");
    // Create dataset
    const dataset = await phoenixClient.datasets.createDataset({
        name: CONFIG.datasetName,
        description: "RAG evaluation dataset with all metrics",
        examples: [
            {
                input: { query: "What is Phoenix?" },
                output: {
                    response: "Phoenix is an AI observability platform.",
                    context: KNOWLEDGE_BASE["What is Phoenix?"]
                },
                expected: { response: "Phoenix is an open-source AI observability platform by Arize AI." }
            },
            // Add more examples...
        ]
    });
    // Task function
    const task = async (example) => {
        const result = await model.generateContent(example.input.query);
        return {
            response: result.response.text(),
            context: example.output.context
        };
    };
    // Run experiment with ALL evaluators
    const experiment = await runExperiment({
        dataset: dataset,
        task: task,
        evaluators: [
            // LLM Evaluators
            llmEvaluators.faithfulness,
            llmEvaluators.correctness,
            llmEvaluators.documentRelevance,
            // Code Evaluators
            codeEvaluators.exactMatch,
            codeEvaluators.jsonValid,
            codeEvaluators.lengthCheck,
            codeEvaluators.containsKeywords,
            // Custom Evaluators
            sentimentEvaluator,
            qualityEvaluator,
        ],
        experimentName: CONFIG.experimentName,
        projectName: "default"
    });
    console.log("Experiment completed:", experiment.id);
    return experiment;
}
// ==========================================
// MAIN - Run everything
// ==========================================
async function main() {
    try {
        // 1. Single response evaluation
        await evaluateSingleResponse();
        // 2. Evaluate existing traces (uncomment if you have traces)
        // await evaluateExistingTraces("default");
        // 3. Run full experiment (uncomment to run)
        // await runFullExperiment();
    }
    catch (error) {
        console.error("Error:", error);
    }
}
main();
