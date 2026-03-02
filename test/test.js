import { testAllEvaluators } from "./evaluators.js";

console.log("Starting Phoenix Evaluation Suite...\n");
testAllEvaluators().catch(console.error);