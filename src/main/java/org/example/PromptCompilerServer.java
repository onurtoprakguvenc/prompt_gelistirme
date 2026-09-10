package org.example;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.json.JsonReadFeature;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.json.JsonMapper;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.*;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public class PromptCompilerServer {

    // ── Configuration ────────────────────────────────────────────────────
    private static final int    PORT           = Integer.parseInt(System.getenv().getOrDefault("PORT", "8081"));
    private static final String GEMINI_API_KEY = resolveApiKey();
    private static final String GEMINI_MODEL   = System.getenv().getOrDefault("GEMINI_MODEL", "gemini-3.6-flash");
    private static final String GEMINI_URL     = "https://generativelanguage.googleapis.com/v1beta/models/"
            + GEMINI_MODEL + ":generateContent?key=" + GEMINI_API_KEY;
    private static final int    MAX_TOKENS     = 8192;

    private static final ObjectMapper JSON = JsonMapper.builder()
            .enable(JsonReadFeature.ALLOW_BACKSLASH_ESCAPING_ANY_CHARACTER)
            .enable(JsonReadFeature.ALLOW_UNESCAPED_CONTROL_CHARS)
            .disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
            .serializationInclusion(JsonInclude.Include.NON_NULL)
            .build();

    private static final HttpClient HTTP = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(15))
            .build();

    // ══════════════════════════════════════════════════════════════════════
    //  DATA RECORDS
    // ══════════════════════════════════════════════════════════════════════

    // ── Stage 1 ──────────────────────────────────────────────────────────

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record StructuralPayload(List<String> requiredInputs, List<String> outputContracts) {}

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record PromptIntentDigest(
            String targetObjective,
            List<String> functionalInvariants,
            List<String> negativeConstraints,
            StructuralPayload structuralPayload,
            List<String> failureModes
    ) {}

    // ── Stage 2 ──────────────────────────────────────────────────────────

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record CompiledPromptSpec(
            String systemDirective,
            String behavioralConstraints,
            String inputSchema,
            String outputSchema,
            String edgeCaseHandling,
            String assembledPrompt
    ) {}

    // ── Stage 3 ──────────────────────────────────────────────────────────

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record PromptVerificationRequest(
            String compiledPrompt,
            String inputSchema,
            String samplePayload,
            List<String> requiredInputs
    ) {}

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record PromptVerificationReport(
            Double structuralIntegrityScore,
            String syntheticTestInput,
            String simulatedOutputDigest,
            List<String> boundaryBreachRisks,
            List<String> hardeningPatches,
            String deploymentManifest
    ) {}

    // ── Shared ───────────────────────────────────────────────────────────

    public record Stage1Request(String rawConcept) {}
    public record ErrorResponse(String error, int status) {}

    // ══════════════════════════════════════════════════════════════════════
    //  SYSTEM INSTRUCTIONS
    // ══════════════════════════════════════════════════════════════════════

    private static final String STAGE1_SYSTEM = """
            You are a deterministic prompt decomposition engine.
            Your sole function: extract load-bearing requirements, boundaries, invariants, \
            and edge conditions from raw user concepts.
            
            RULES — ABSOLUTE:
            1. Strip ALL conversational packaging, polite phrasing, persona fluff, empty descriptors.
            2. Formulate functional requirements — never descriptive suggestions.
            3. Define non-negotiable boundaries and explicit anti-patterns.
            4. Identify every plausible failure mode and ambiguity zone.
            5. Output ONLY the JSON object matching the provided schema. No prose, no markdown, \
               no preamble, no explanation.
            """;

    private static final String STAGE2_SYSTEM = """
            You are a prompt specification compiler targeting high-parameter reasoning models \
            (Claude Opus tier, advanced reasoning tiers).
            
            INPUT: A structured PromptIntentDigest JSON containing extracted objectives, invariants, \
            constraints, payload schemas, and failure modes from a prior decomposition stage.
            
            YOUR TASK: Compile a production-grade prompt specification from the digest. The output \
            must maximize the model's reasoning capacity by:
            1. systemDirective — A dense, imperative system block defining the model's exact role \
               and operational mode. No persona fluff, no "You are an expert" patterns. Pure \
               functional directives.
            2. behavioralConstraints — Merged invariants and negative constraints as enforceable \
               rules with explicit violation consequences.
            3. inputSchema — Formal description of what dynamic inputs the prompt accepts, with \
               types and validation expectations.
            4. outputSchema — Exact structural contract the model must satisfy, including format, \
               field names, and data types.
            5. edgeCaseHandling — Explicit instructions for each identified failure mode: what to \
               do when ambiguity, missing data, or conflicting requirements are encountered.
            6. assembledPrompt — The final, fully assembled prompt text ready for deployment. This \
               is the concatenation of all above blocks into a single coherent prompt with clear \
               section delimiters. Must be copy-paste ready.
            
            RULES — ABSOLUTE:
            - Output ONLY the JSON object matching the provided schema.
            - No markdown fences, no preamble, no meta-commentary.
            - Every sentence must be an enforceable directive or a structural definition.
            - Prefer imperative mood. Ban hedging words: "consider", "might", "perhaps", "try to".
            """;

    private static final String STAGE3_SYSTEM = """
            You are a prompt verification and hardening engine. You perform adversarial structural \
            analysis on compiled prompt specifications before deployment.
            
            INPUT: A PromptVerificationRequest containing:
            - compiledPrompt: the full assembled prompt from a prior compilation stage.
            - inputSchema: the expected dynamic input contract.
            - samplePayload: optional user-supplied test payload. CRITICAL: If samplePayload is \
              null, empty, or blank, you MUST evaluate the compiled prompt strictly by feeding \
              your own generated syntheticTestInput into the simulation to produce simulatedOutputDigest. \
              Never skip simulation due to absent sample data.
            - requiredInputs: list of named input variables extracted during Stage 1 decomposition \
              (e.g., "source_text", "target_language", "schema_definition"). Use these to generate \
              semantically accurate dynamic placeholder tokens in the deployment manifest.
            
            YOUR TASK — execute ALL of the following analyses:
            
            1. structuralIntegrityScore (0.0 to 1.0):
               Evaluate using POSITIVE-STEERING criteria. Score additively across four axes, each \
               contributing 0.25 to the maximum 1.0:
               A) DIRECTIVE CLARITY (0.0–0.25): Every instruction uses imperative mood with \
                  unambiguous verbs. Directives specify exact behaviors, not aspirational goals. \
                  Absence of hedging language ("try to", "consider", "might", "perhaps") is a \
                  natural consequence, not a checklist item.
               B) BOUNDARY ENFORCEABILITY (0.0–0.25): Constraints are testable and binary \
                  (pass/fail), not gradient. Delimiter tags isolate system instructions from \
                  dynamic input zones. Violation consequences are stated.
               C) OUTPUT CONTRACT DETERMINISM (0.0–0.25): The output schema is fully specified \
                  with field names, types, and structural constraints. A conformance test can be \
                  written against the schema without interpretation.
               D) FAILURE-MODE COVERAGE (0.0–0.25): Each identified edge case has an explicit \
                  handling directive. Missing-data, malformed-input, and conflicting-parameter \
                  scenarios are addressed with concrete fallback behaviors.
               Sum the four axis scores. The result MUST be a decimal between 0.0 and 1.0.
            
            2. syntheticTestInput:
               Generate ONE adversarial edge-case input that conforms to the inputSchema but is \
               designed to probe boundary conditions: missing optional fields, maximum-length values, \
               contradictory parameters, type coercion traps, or injection attempts.
            
            3. simulatedOutputDigest:
               Given the compiledPrompt, simulate what a high-tier reasoning model would produce \
               when given the test input. Use samplePayload as the test input if it is non-empty; \
               otherwise use syntheticTestInput. Verify the simulated output conforms to the output \
               schema defined in the prompt. Provide the simulated output as a compact string.
            
            4. boundaryBreachRisks:
               List every concrete risk where a target model could drift from the prompt's intent: \
               attention budget exhaustion on long inputs, schema field omission under token pressure, \
               persona bleed-through, soft constraint erosion over multi-turn conversations, \
               delimiter confusion from user-injected content.
            
            5. hardeningPatches:
               For each identified breach risk, provide ONE concrete imperative directive that patches \
               the vulnerability. Each patch must be a copy-paste-ready sentence that can be appended \
               to the prompt's system block. No explanations — directives only. Imperative mood, \
               maximum 2 sentences each.
            
            6. deploymentManifest:
               Produce the final production-ready prompt wrapper using XML-style delimiter tags. \
               Derive dynamic placeholder tokens from the requiredInputs list using this \
               deterministic normalization: strip all characters except [a-zA-Z0-9_ ], replace \
               spaces and hyphens with underscores, collapse consecutive underscores, convert to \
               UPPER_SNAKE_CASE, wrap in double curly braces. Examples: "source text" → \
               {{SOURCE_TEXT}}, "user-query" → {{USER_QUERY}}, "schema.def" → {{SCHEMADEF}}. \
               Every token inside <dynamic_input> must match the regex {{[A-Z0-9_]+}}. \
               If requiredInputs is empty or null, use a single {{INPUT}} fallback.
               
               Structure:
               <system_instruction>
               [The hardened system directive with all hardeningPatches integrated]
               </system_instruction>
               <payload_contract>
               [Input/output schema block]
               </payload_contract>
               <dynamic_input>
               [One placeholder token per required input, each on its own labeled line]
               </dynamic_input>
            
            RULES — ABSOLUTE:
            - Output ONLY the JSON matching the provided schema.
            - No markdown, no preamble, no meta-commentary.
            - The structuralIntegrityScore MUST be a decimal number between 0.0 and 1.0.
            - The deploymentManifest MUST use XML-style delimiter tags exactly as specified.
            """;

    // ══════════════════════════════════════════════════════════════════════
    //  GEMINI RESPONSE SCHEMAS
    // ══════════════════════════════════════════════════════════════════════

    private static final String STAGE1_RESPONSE_SCHEMA = """
            {
              "type": "OBJECT",
              "properties": {
                "targetObjective":      { "type": "STRING" },
                "functionalInvariants": { "type": "ARRAY", "items": { "type": "STRING" } },
                "negativeConstraints":  { "type": "ARRAY", "items": { "type": "STRING" } },
                "structuralPayload": {
                  "type": "OBJECT",
                  "properties": {
                    "requiredInputs":  { "type": "ARRAY", "items": { "type": "STRING" } },
                    "outputContracts": { "type": "ARRAY", "items": { "type": "STRING" } }
                  },
                  "required": ["requiredInputs", "outputContracts"]
                },
                "failureModes": { "type": "ARRAY", "items": { "type": "STRING" } }
              },
              "required": ["targetObjective","functionalInvariants","negativeConstraints","structuralPayload","failureModes"]
            }
            """;

    private static final String STAGE2_RESPONSE_SCHEMA = """
            {
              "type": "OBJECT",
              "properties": {
                "systemDirective":       { "type": "STRING" },
                "behavioralConstraints": { "type": "STRING" },
                "inputSchema":           { "type": "STRING" },
                "outputSchema":          { "type": "STRING" },
                "edgeCaseHandling":      { "type": "STRING" },
                "assembledPrompt":       { "type": "STRING" }
              },
              "required": ["systemDirective","behavioralConstraints","inputSchema","outputSchema","edgeCaseHandling","assembledPrompt"]
            }
            """;

    private static final String STAGE3_RESPONSE_SCHEMA = """
            {
              "type": "OBJECT",
              "properties": {
                "structuralIntegrityScore": { "type": "NUMBER" },
                "syntheticTestInput":       { "type": "STRING" },
                "simulatedOutputDigest":    { "type": "STRING" },
                "boundaryBreachRisks":      { "type": "ARRAY", "items": { "type": "STRING" } },
                "hardeningPatches":         { "type": "ARRAY", "items": { "type": "STRING" } },
                "deploymentManifest":       { "type": "STRING" }
              },
              "required": ["structuralIntegrityScore","syntheticTestInput","simulatedOutputDigest","boundaryBreachRisks","hardeningPatches","deploymentManifest"]
            }
            """;

    // ══════════════════════════════════════════════════════════════════════
    //  ENTRY POINT
    // ══════════════════════════════════════════════════════════════════════

    public static void main(String[] args) throws IOException {
        if (GEMINI_API_KEY.isBlank()) {
            System.err.println("FATAL: Set GEMINI_API_KEY environment variable.");
            System.exit(1);
        }

        HttpServer server = HttpServer.create(new InetSocketAddress(PORT), 0);
        server.createContext("/",                   new StaticHandler());
        server.createContext("/api/compile/stage1", new Stage1Handler());
        server.createContext("/api/compile/stage2", new Stage2Handler());
        server.createContext("/api/compile/stage3", new Stage3Handler());
        server.setExecutor(java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor());
        server.start();

        System.out.printf("Prompt Compiler [3-stage] running → http://localhost:%d%n", PORT);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  HTTP HANDLERS
    // ══════════════════════════════════════════════════════════════════════

    /** GET / — serves index.html from classpath. */
    static class StaticHandler implements HttpHandler {
        private final byte[] indexHtml;

        StaticHandler() {
            try (InputStream is = Objects.requireNonNull(
                    getClass().getResourceAsStream("/index.html"),
                    "index.html not found on classpath")) {
                indexHtml = is.readAllBytes();
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        @Override
        public void handle(HttpExchange ex) throws IOException {
            if ("GET".equals(ex.getRequestMethod())) {
                ex.getResponseHeaders().add("Content-Type", "text/html; charset=utf-8");
                ex.sendResponseHeaders(200, indexHtml.length);
                ex.getResponseBody().write(indexHtml);
            } else {
                ex.sendResponseHeaders(405, -1);
            }
            ex.close();
        }
    }

    /** POST /api/compile/stage1 — raw concept → PromptIntentDigest */
    static class Stage1Handler implements HttpHandler {
        @Override
        public void handle(HttpExchange ex) throws IOException {
            setCors(ex);
            if (preflight(ex)) return;
            if (!requirePost(ex)) return;

            try {
                String body = readBody(ex);
                Stage1Request req = JSON.readValue(body, Stage1Request.class);
                if (req.rawConcept() == null || req.rawConcept().isBlank()) {
                    sendError(ex, 400, "Field 'rawConcept' is required and must be non-empty.");
                    return;
                }

                String geminiPayload = buildGeminiPayload(STAGE1_SYSTEM, req.rawConcept(), STAGE1_RESPONSE_SCHEMA);
                String rawResponse   = callGemini(geminiPayload);
                String extracted     = extractText(rawResponse);
                String sanitized     = sanitizeJsonString(extracted);

                PromptIntentDigest digest = JSON.readValue(sanitized, PromptIntentDigest.class);
                sendJson(ex, 200, digest);
            } catch (JsonProcessingException e) {
                sendError(ex, 502, "Model returned unparseable JSON: " + e.getOriginalMessage());
            } catch (RateLimitException e) {
                sendError(ex, 429, e.getMessage());
            } catch (GeminiException e) {
                sendError(ex, 502, "Model call failed: " + e.getMessage());
            } catch (Exception e) {
                sendError(ex, 500, "Internal error: " + e.getMessage());
            }
        }
    }

    /** POST /api/compile/stage2 — PromptIntentDigest → CompiledPromptSpec */
    static class Stage2Handler implements HttpHandler {
        @Override
        public void handle(HttpExchange ex) throws IOException {
            setCors(ex);
            if (preflight(ex)) return;
            if (!requirePost(ex)) return;

            try {
                String body = readBody(ex);
                PromptIntentDigest digest = JSON.readValue(body, PromptIntentDigest.class);
                if (digest.targetObjective() == null || digest.targetObjective().isBlank()) {
                    sendError(ex, 400, "Invalid PromptIntentDigest: targetObjective is required.");
                    return;
                }

                String digestJson    = JSON.writeValueAsString(digest);
                String userMessage   = "Compile the following PromptIntentDigest into a production prompt specification:\n\n" + digestJson;
                String geminiPayload = buildGeminiPayload(STAGE2_SYSTEM, userMessage, STAGE2_RESPONSE_SCHEMA);
                String rawResponse   = callGemini(geminiPayload);
                String extracted     = extractText(rawResponse);
                String sanitized     = sanitizeJsonString(extracted);

                CompiledPromptSpec spec = JSON.readValue(sanitized, CompiledPromptSpec.class);
                sendJson(ex, 200, spec);
            } catch (JsonProcessingException e) {
                sendError(ex, 502, "Model returned unparseable JSON: " + e.getOriginalMessage());
            } catch (RateLimitException e) {
                sendError(ex, 429, e.getMessage());
            } catch (GeminiException e) {
                sendError(ex, 502, "Model call failed: " + e.getMessage());
            } catch (Exception e) {
                sendError(ex, 500, "Internal error: " + e.getMessage());
            }
        }
    }

    /** POST /api/compile/stage3 — CompiledPromptSpec → PromptVerificationReport */
    static class Stage3Handler implements HttpHandler {
        @Override
        public void handle(HttpExchange ex) throws IOException {
            setCors(ex);
            if (preflight(ex)) return;
            if (!requirePost(ex)) return;

            try {
                String body = readBody(ex);
                PromptVerificationRequest req = JSON.readValue(body, PromptVerificationRequest.class);

                if (req.compiledPrompt() == null || req.compiledPrompt().isBlank()) {
                    sendError(ex, 400, "Field 'compiledPrompt' is required and must be non-empty.");
                    return;
                }
                if (req.inputSchema() == null || req.inputSchema().isBlank()) {
                    sendError(ex, 400, "Field 'inputSchema' is required and must be non-empty.");
                    return;
                }

                String reqJson     = JSON.writeValueAsString(req);
                String userMessage = "Verify and harden the following compiled prompt specification:\n\n" + reqJson;
                String geminiPayload = buildGeminiPayload(STAGE3_SYSTEM, userMessage, STAGE3_RESPONSE_SCHEMA);
                String rawResponse   = callGemini(geminiPayload);
                String extracted     = extractText(rawResponse);
                String sanitized     = sanitizeJsonString(extracted);

                PromptVerificationReport report = JSON.readValue(sanitized, PromptVerificationReport.class);

                // Clamp score to [0.0, 1.0] in case the model drifted
                if (report.structuralIntegrityScore() != null) {
                    double clamped = Math.max(0.0, Math.min(1.0, report.structuralIntegrityScore()));
                    if (clamped != report.structuralIntegrityScore()) {
                        report = new PromptVerificationReport(
                                clamped,
                                report.syntheticTestInput(),
                                report.simulatedOutputDigest(),
                                report.boundaryBreachRisks(),
                                report.hardeningPatches(),
                                report.deploymentManifest()
                        );
                    }
                }

                sendJson(ex, 200, report);
            } catch (JsonProcessingException e) {
                sendError(ex, 502, "Model returned unparseable JSON: " + e.getOriginalMessage());
            } catch (RateLimitException e) {
                sendError(ex, 429, e.getMessage());
            } catch (GeminiException e) {
                sendError(ex, 502, "Model call failed: " + e.getMessage());
            } catch (Exception e) {
                sendError(ex, 500, "Internal error: " + e.getMessage());
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    //  GEMINI INTEGRATION
    // ══════════════════════════════════════════════════════════════════════

    private static String buildGeminiPayload(String systemInstruction, String userMessage, String responseSchema) {
        try {
            Object schemaObj = JSON.readValue(responseSchema, Object.class);
            Map<String, Object> payload = Map.of(
                    "systemInstruction", Map.of("parts", List.of(Map.of("text", systemInstruction))),
                    "contents",          List.of(Map.of(
                            "role", "user",
                            "parts", List.of(Map.of("text", userMessage))
                    )),
                    "generationConfig",  Map.of(
                            "responseMimeType", "application/json",
                            "responseSchema",   schemaObj,
                            "maxOutputTokens",  MAX_TOKENS,
                            "temperature",      0.1
                    )
            );
            return JSON.writeValueAsString(payload);
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Failed to build Gemini payload", e);
        }
    }

    private static String callGemini(String jsonPayload) throws GeminiException {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(GEMINI_URL))
                    .header("Content-Type", "application/json")
                    .timeout(Duration.ofSeconds(90))
                    .POST(HttpRequest.BodyPublishers.ofString(jsonPayload))
                    .build();

            HttpResponse<String> resp = HTTP.send(request, HttpResponse.BodyHandlers.ofString());

            if (resp.statusCode() == 429) {
                throw new RateLimitException(
                        "Model quota exhausted or rate limit hit. Wait 15-30 seconds before retrying.");
            }
            if (resp.statusCode() != 200) {
                throw new GeminiException("HTTP " + resp.statusCode() + ": " + truncate(resp.body(), 400));
            }
            return resp.body();
        } catch (GeminiException e) {
            throw e;
        } catch (Exception e) {
            throw new GeminiException("Network error: " + e.getMessage());
        }
    }

    @SuppressWarnings("unchecked")
    private static String extractText(String geminiResponse) throws GeminiException {
        try {
            Map<String, Object> root = JSON.readValue(geminiResponse, Map.class);
            List<Map<String, Object>> candidates = (List<Map<String, Object>>) root.get("candidates");
            if (candidates == null || candidates.isEmpty()) {
                throw new GeminiException("No candidates in response");
            }
            Map<String, Object> content = (Map<String, Object>) candidates.get(0).get("content");
            if (content == null) {
                // Check for blocked content
                Object finishReason = candidates.get(0).get("finishReason");
                throw new GeminiException("No content in candidate (finishReason: " + finishReason + ")");
            }
            List<Map<String, Object>> parts = (List<Map<String, Object>>) content.get("parts");
            if (parts == null || parts.isEmpty()) {
                throw new GeminiException("No parts in candidate content");
            }
            Object text = parts.get(0).get("text");
            if (text == null) {
                throw new GeminiException("Empty text in response part");
            }
            return text.toString();
        } catch (GeminiException e) {
            throw e;
        } catch (Exception e) {
            throw new GeminiException("Failed to parse Gemini envelope: " + e.getMessage());
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    //  UTILITIES
    // ══════════════════════════════════════════════════════════════════════

    private static String resolveApiKey() {
        String key = System.getenv("GEMINI_API_KEY");
        return (key != null) ? key : "";
    }

    /** Strips markdown fences and unescaped control characters from model JSON output. */
    private static String sanitizeJsonString(String raw) {
        if (raw == null) return "{}";
        String s = raw.strip();
        if (s.startsWith("```")) {
            s = s.replaceFirst("^```[a-zA-Z]*\\n?", "");
            s = s.replaceFirst("\\n?```$", "");
            s = s.strip();
        }
        StringBuilder sb = new StringBuilder(s.length());
        boolean inString = false;
        boolean escaped = false;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (escaped) {
                sb.append(c);
                escaped = false;
                continue;
            }
            if (c == '\\' && inString) {
                sb.append(c);
                escaped = true;
                continue;
            }
            if (c == '"') {
                inString = !inString;
            }
            if (inString && c < 0x20 && c != '\n' && c != '\r' && c != '\t') {
                sb.append(' ');
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    private static String readBody(HttpExchange ex) throws IOException {
        try (InputStream is = ex.getRequestBody()) {
            return new String(is.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    private static void sendJson(HttpExchange ex, int code, Object obj) throws IOException {
        byte[] bytes = JSON.writeValueAsBytes(obj);
        ex.getResponseHeaders().add("Content-Type", "application/json; charset=utf-8");
        ex.sendResponseHeaders(code, bytes.length);
        ex.getResponseBody().write(bytes);
        ex.close();
    }

    private static void sendError(HttpExchange ex, int code, String message) throws IOException {
        sendJson(ex, code, new ErrorResponse(message, code));
    }

    private static void setCors(HttpExchange ex) {
        ex.getResponseHeaders().add("Access-Control-Allow-Origin", "*");
        ex.getResponseHeaders().add("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
        ex.getResponseHeaders().add("Access-Control-Allow-Headers", "Content-Type");
    }

    /** Handles OPTIONS preflight. Returns true if handled (caller should return). */
    private static boolean preflight(HttpExchange ex) throws IOException {
        if ("OPTIONS".equals(ex.getRequestMethod())) {
            ex.sendResponseHeaders(204, -1);
            ex.close();
            return true;
        }
        return false;
    }

    /** Validates POST method. Returns false if rejected (caller should return). */
    private static boolean requirePost(HttpExchange ex) throws IOException {
        if (!"POST".equals(ex.getRequestMethod())) {
            sendError(ex, 405, "Method not allowed");
            return false;
        }
        return true;
    }

    private static String truncate(String s, int max) {
        return (s != null && s.length() > max) ? s.substring(0, max) + "…" : s;
    }

    static class GeminiException extends Exception {
        GeminiException(String msg) { super(msg); }
    }

    static class RateLimitException extends GeminiException {
        RateLimitException(String msg) { super(msg); }
    }
}