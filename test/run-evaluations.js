import { testAllEvaluators, allEvaluators, runAllEvaluations, logEvaluationsToPhoenix } from "./phoenix-evaluations.js";

// Run full test suite
console.log("Starting Phoenix Evaluation Suite...\n");
testAllEvaluators().catch(console.error);