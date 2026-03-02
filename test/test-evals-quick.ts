// Quick test to verify all evaluators work
import { 
  llmEvaluators, 
  codeEvaluators, 
  sentimentEvaluator, 
  qualityEvaluator 
} from "./evaluators";

async function quickTest() {
  console.log("Testing all evaluators...\n");

  const testData = {
    query: "What is Phoenix?",
    response: "Phoenix is an open-source AI observability platform by Arize AI.",
    context: "Phoenix is an open-source AI observability platform by Arize AI."
  };

  // Test LLM evaluators
  console.log("✓ Testing Faithfulness...");
  const f = await llmEvaluators.faithfulness.evaluate({
    input: testData.query,
    output: testData.response,
    context: testData.context
  });
  console.log(`  Result: ${f.label} (${f.score})\n`);

  console.log("✓ Testing Correctness...");
  const c = await llmEvaluators.correctness.evaluate({
    input: testData.query,
    output: testData.response
  });
  console.log(`  Result: ${c.label} (${c.score})\n`);

  // Test Code evaluators
  console.log("✓ Testing Exact Match...");
  const em = codeEvaluators.exactMatch.evaluate({
    output: "hello",
    expected: "hello"
  });
  console.log(`  Result: ${em.score}\n`);

  console.log("✓ Testing JSON Valid...");
  const jv = codeEvaluators.jsonValid.evaluate({
    output: '{"test": true}'
  });
  console.log(`  Result: ${jv.score}\n`);

  // Test Custom evaluators
  console.log("✓ Testing Sentiment...");
  const s = await sentimentEvaluator.evaluate({
    text: "This is amazing!"
  });
  console.log(`  Result: ${s.label} (${s.score})\n`);

  console.log("✓ All tests passed!");
}

quickTest().catch(console.error);