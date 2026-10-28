import { expect, test } from "bun:test";

import {
  buildTrustedTokensModel,
  cacheV2Set,
  modalityPlan,
  resolveTrustedTokensBaseModel,
  trustedtokens,
  type TrustedTokensModel,
} from "../src/sync/providers/trustedtokens.js";

function model(overrides: Partial<TrustedTokensModel> = {}): TrustedTokensModel {
  return {
    id: "zai-org/GLM-5.3",
    pricing: {
      prompt: "0.0000015",
      completion: "0.0000045",
      input_cache_read: "0.0000003",
    },
    architecture: {
      input_modalities: ["text"],
      output_modalities: ["text"],
    },
    context_length: 230_000,
    supported_parameters: ["temperature", "tools", "response_format"],
    reasoning: { supported_efforts: ["max", "high", "low"] },
    ...overrides,
  };
}

test("maps TrustedTokens lab namespaces onto catalog labs", () => {
  expect(resolveTrustedTokensBaseModel("Qwen/Qwen3.6-35B-A3B-FP8")).toBe("alibaba/qwen3.6-35b-a3b");
  expect(resolveTrustedTokensBaseModel("Qwen/Qwen3.8-27B")).toBe("alibaba/qwen3.8-27b");
  expect(resolveTrustedTokensBaseModel("zai-org/GLM-5.3")).toBe("zhipuai/glm-5.3");
  expect(resolveTrustedTokensBaseModel("deepseek-ai/DeepSeek-V4.1-Flash-1017"))
    .toBe("deepseek/deepseek-v4.1-flash-1017");
  expect(resolveTrustedTokensBaseModel("nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B"))
    .toBe("nvidia/nemotron-3.5-super-vl-120b-a12b");
});

test("leaves models without a lab entry and route variants unresolved", () => {
  expect(resolveTrustedTokensBaseModel("tngtech/DeepSeek-TNG-R1T2-Chimera")).toBeUndefined();
  expect(resolveTrustedTokensBaseModel("acme/not-in-the-catalog")).toBeUndefined();
  expect(resolveTrustedTokensBaseModel("tng/zai-org/GLM-5.3")).toBeUndefined();
});

test("converts the catalog's EUR per-token prices to USD per million tokens", () => {
  const built = buildTrustedTokensModel(model(), undefined, "zhipuai/glm-5.3");

  expect(built).toMatchObject({
    base_model: "zhipuai/glm-5.3",
    // EUR 1.50 / 4.50 / 0.30 per 1M tokens at 1.1150 USD/EUR.
    cost: { input: 1.6725, output: 5.0175, cache_read: 0.3345 },
  });
});

test("publishes the lower of catalog and lab context, and keeps output inherited", () => {
  const built = buildTrustedTokensModel(model(), undefined, "zhipuai/glm-5.3");

  expect(built).toMatchObject({ limit: { context: 230_000 } });
  expect(built).not.toHaveProperty("limit.output");
});

test("never raises context above the lab entry", () => {
  const built = buildTrustedTokensModel(
    model({ id: "Qwen/Qwen3.8-27B", context_length: 1_000_000 }),
    undefined,
    "alibaba/qwen3.8-27b",
  );

  expect(built).not.toHaveProperty("limit");
});

test("sorts reasoning efforts low to high and keeps none", () => {
  const built = buildTrustedTokensModel(
    model({
      id: "Qwen/Qwen3.8-27B",
      reasoning: { supported_efforts: ["xhigh", "none", "medium", "low"] },
    }),
    undefined,
    "alibaba/qwen3.8-27b",
  );

  expect(built).toMatchObject({
    reasoning_options: [{ type: "effort", values: ["none", "low", "medium", "xhigh"] }],
  });
});

test("adds the measured off state to DeepSeek's catalog ladder", () => {
  const built = buildTrustedTokensModel(
    model({
      id: "deepseek-ai/DeepSeek-V4.1-Flash-1017",
      reasoning: { supported_efforts: ["max", "high", "low"] },
    }),
    undefined,
    "deepseek/deepseek-v4.1-flash-1017",
  );

  expect(built).toMatchObject({
    reasoning_options: [{ type: "effort", values: ["none", "low", "high", "max"] }],
  });
});

test("uses the Kilo peer's none/high pair for the toggle-only labs", () => {
  const qwen = buildTrustedTokensModel(
    model({
      id: "Qwen/Qwen3.6-35B-A3B-FP8",
      pricing: { prompt: "0.00000025", completion: "0.0000015", input_cache_read: "0.00000005" },
      architecture: { input_modalities: ["text", "image"], output_modalities: ["text"] },
      context_length: 262_144,
      reasoning: { supported_efforts: ["medium", "none"] },
    }),
    undefined,
    "alibaba/qwen3.6-35b-a3b",
  );
  expect(qwen).toMatchObject({
    reasoning_options: [{ type: "effort", values: ["none", "high"] }],
  });

  const gemma = buildTrustedTokensModel(
    model({
      id: "google/gemma-4-26B-A4B-it",
      pricing: { prompt: "0.0000003", completion: "0.0000008", input_cache_read: "0.00000005" },
      architecture: { input_modalities: ["text", "image"], output_modalities: ["text"] },
      context_length: 262_144,
      reasoning: { supported_efforts: ["medium", "none"] },
    }),
    undefined,
    "google/gemma-4-26b-a4b-it",
  );
  expect(gemma).toMatchObject({
    reasoning_options: [{ type: "effort", values: ["none", "high"] }],
  });

  const nemotron = buildTrustedTokensModel(
    model({
      id: "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B",
      pricing: { prompt: "0.00000015", completion: "0.0000005", input_cache_read: "0.00000002" },
      architecture: { input_modalities: ["text", "image"], output_modalities: ["text"] },
      context_length: 262_144,
      reasoning: { supported_efforts: ["medium", "none"] },
    }),
    undefined,
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
  );
  expect(nemotron).toMatchObject({
    reasoning_options: [{ type: "effort", values: ["none", "high"] }],
  });
});

test("copies the NVIDIA peer ladder for DiffusionGemma", () => {
  const built = buildTrustedTokensModel(
    model({
      id: "google/diffusiongemma-26B-A4B-it",
      pricing: { prompt: "0.0000003", completion: "0.0000008", input_cache_read: "0.00000005" },
      architecture: { input_modalities: ["text", "image"], output_modalities: ["text"] },
      context_length: 262_144,
      supported_parameters: ["top_p", "top_k", "max_tokens", "response_format", "tools", "reasoning_effort"],
      reasoning: { supported_efforts: ["medium", "none"] },
    }),
    undefined,
    "google/diffusiongemma-26b-a4b-it",
  );

  expect(built).toMatchObject({
    reasoning_options: [{
      type: "effort",
      values: ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
    }],
  });
});

test("does not invent reasoning controls when the catalog has none", () => {
  const built = buildTrustedTokensModel(
    model({ reasoning: { supported_efforts: null } }),
    undefined,
    "zhipuai/glm-5.3",
  );

  expect(built).toMatchObject({ reasoning_options: [] });
});

test("records the reasoning side channel for every reasoner", () => {
  const built = buildTrustedTokensModel(model(), undefined, "zhipuai/glm-5.3");

  expect(built).toMatchObject({ interleaved: { field: "reasoning_content" } });
});

test("marks DiffusionGemma's measured temperature and structured-output rejections", () => {
  const built = buildTrustedTokensModel(
    model({
      id: "google/diffusiongemma-26B-A4B-it",
      pricing: { prompt: "0.0000003", completion: "0.0000008", input_cache_read: "0.00000005" },
      architecture: { input_modalities: ["text", "image"], output_modalities: ["text"] },
      context_length: 262_144,
      supported_parameters: ["top_p", "top_k", "max_tokens", "response_format", "tools"],
      reasoning: { supported_efforts: ["medium", "none"] },
    }),
    undefined,
    "google/diffusiongemma-26b-a4b-it",
  );

  expect(built).toMatchObject({ temperature: false, structured_output: false });
});

test("keeps the lab's image formats but drops video and PDF the host rejects", () => {
  const qwenModel = model({
    id: "Qwen/Qwen3.6-35B-A3B-FP8",
    pricing: { prompt: "0.00000025", completion: "0.0000015", input_cache_read: "0.00000005" },
    architecture: { input_modalities: ["text", "image"], output_modalities: ["text"] },
    context_length: 262_144,
    reasoning: { supported_efforts: ["medium", "none"] },
  });
  const qwenPlan = modalityPlan(qwenModel, undefined, "alibaba/qwen3.6-35b-a3b");
  const qwen = buildTrustedTokensModel(qwenModel, undefined, "alibaba/qwen3.6-35b-a3b", qwenPlan);
  expect(qwen).toMatchObject({ modalities: { input: ["text", "image"] } });
  expect(qwenPlan.v2).toEqual(["text", "image/bmp", "image/jpeg", "image/png", "image/tiff", "image/webp", "image/heic"]);
  expect(JSON.stringify(qwenPlan)).not.toContain("video/");

  const flashModel = model({
    id: "zai-org/GLM-5.3-Flash",
    pricing: { prompt: "0.0000003", completion: "0.0000006", input_cache_read: "0.00000006" },
    architecture: { input_modalities: ["text", "image"], output_modalities: ["text"] },
    context_length: 1_048_576,
  });
  const flashPlan = modalityPlan(flashModel, undefined, "zhipuai/glm-5.3-flash");
  const flash = buildTrustedTokensModel(flashModel, undefined, "zhipuai/glm-5.3-flash", flashPlan);
  expect(flash).toMatchObject({ modalities: { input: ["text", "image"] } });
  expect(flashPlan.v2).toEqual(["text", "image/jpeg", "image/png"]);
});

test("inherits modalities unchanged when the lab has no rejected formats", () => {
  const deepseekModel = model({
    id: "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    pricing: { prompt: "0.0000003", completion: "0.0000006", input_cache_read: "0.00000006" },
    architecture: { input_modalities: ["text", "image"], output_modalities: ["text"] },
    context_length: 1_048_576,
    reasoning: { supported_efforts: ["max", "high", "low"] },
  });
  expect(modalityPlan(deepseekModel, undefined, "deepseek/deepseek-v4.1-flash-1017")).toEqual({});
  const deepseek = buildTrustedTokensModel(deepseekModel, undefined, "deepseek/deepseek-v4.1-flash-1017");
  expect(deepseek).not.toHaveProperty("modalities");

  const glm = buildTrustedTokensModel(model(), undefined, "zhipuai/glm-5.3");
  expect(glm).not.toHaveProperty("modalities");
});

test("adds implicit caching only when the file has no authored cache kind", () => {
  expect(cacheV2Set(0.3, undefined)).toEqual({ "capabilities.cache": { implicit: true } });
  expect(cacheV2Set(0.3, { capabilities: { cache: { explicit: {} } } })).toBeUndefined();
  expect(cacheV2Set(undefined, {})).toBeUndefined();
});

test("ignores TNG's own model until its lab metadata exists", () => {
  expect(trustedtokens.missingModelID!(model({ id: "tngtech/DeepSeek-TNG-R1T2-Chimera" })))
    .toBeUndefined();
  expect(trustedtokens.missingModelID!(model({ id: "acme/not-in-the-catalog" })))
    .toBe("acme/not-in-the-catalog");
  expect(trustedtokens.missingModelID!(model({ id: "tng/zai-org/GLM-5.3" }))).toBeUndefined();

  expect(trustedtokens.translateModel(
    model({ id: "tngtech/DeepSeek-TNG-R1T2-Chimera" }),
    { existing: () => undefined, authored: () => undefined, v2: () => undefined },
  )).toBeUndefined();
});

test("reports the intentionally skipped model", () => {
  const notices = trustedtokens.sourceNotices!([
    model({ id: "tngtech/DeepSeek-TNG-R1T2-Chimera" }),
    model({ id: "tng/zai-org/GLM-5.3" }),
  ]);

  expect(notices.join("\n")).toContain("intentionally not published");
  expect(notices.join("\n")).toContain("provider-route variants");
});

test("refuses a truncated catalog", () => {
  const few = { data: Array.from({ length: 4 }, (_, index) => ({ id: `vendor/model-${index}` })) };
  expect(() => trustedtokens.parseModels(few)).toThrow("truncated catalog");

  const enough = { data: Array.from({ length: 5 }, (_, index) => ({ id: `vendor/model-${index}` })) };
  expect(trustedtokens.parseModels(enough)).toHaveLength(5);
});
