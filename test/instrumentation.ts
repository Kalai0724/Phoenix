import { NodeSDK } from "@opentelemetry/sdk-node";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { ATTR_SERVICE_NAME } from "@opentelemetry/semantic-conventions";

const sdk = new NodeSDK({
  resource: resourceFromAttributes({
    [ATTR_SERVICE_NAME]: "rag-evaluation",
  }),
  traceExporter: new OTLPTraceExporter({
    url: "http://localhost:6006/v1/traces",
  }),
});

sdk.start();

console.log("✅ OpenTelemetry instrumentation started");