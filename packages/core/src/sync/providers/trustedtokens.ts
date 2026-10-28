import { readFileSync } from "node:fs";
import path from "node:path";
import { z } from "zod";

import type { ExistingModel, SyncProvider, SyncedFullModel, SyncedModel } from "../index.js";
import { factorBaseModel, modelMetadata, resolveModelMetadataBaseModel } from "./openrouter.js";

// TrustedTokens (TNG Technology Consulting) serves models from German
// infrastructure through an OpenAI-compatible gateway. `/v1/models` requires
// the account key and returns the OpenRouter consumer schema plus a
// `reasoning` object:
// https://trustedtokens.eu/docs (accessed 2026-10-28)
const API_ENDPOINT = "https://api.trustedtokens.eu/v1/models";

// The catalog is account-scoped, so a handful of rows means a truncated
// response rather than a real retirement wave: the live catalog lists 11
// models and every plan advertises all of them.
const MIN_MODELS = 5;

// TNG's own DeepSeek R1T2 merge is served but has no `models/` lab entry yet.
// Leave it out until that metadata is authored; a `[missing-model]` issue would
// otherwise be opened on every sync.
const IGNORED_IDS = new Set(["tngtech/DeepSeek-TNG-R1T2-Chimera"]);

// Measured 2026-10-28: `reasoning_effort="none"` returns no reasoning_content
// on DeepSeek V4.1, so the host has a real off state that its own catalog
// omits. The rest of the ladder matches the lab/OpenRouter baseline.
const REASONING_OVERRIDES: Record<string, ReasoningEffort[] | undefined> = {
  "deepseek-ai/DeepSeek-V4.1-Flash-1017": ["none", "low", "high", "max"],
};

// Measured 2026-10-28: DiffusionGemma rejects `response_format` with 422
// ("Structured outputs are not yet supported for diffusion language models")
// even though the catalog advertises `response_format`.
const STRUCTURED_OUTPUT_OVERRIDES: Record<string, boolean | undefined> = {
  "google/diffusiongemma-26B-A4B-it": false,
};

const HEADER = `# Costs and catalog: ${API_ENDPOINT} (accessed 2026-10-28).
# Reasoning, cache, modalities, tools and structured output were measured on 2026-10-28; see sync.md → TrustedTokens Notes.
# Reasoning side channel: message.reasoning_content.
`;

const Price = z.union([z.string(), z.number()]);

const ReasoningEffort = ["none", "minimal", "low", "medium", "high", "xhigh", "max"] as const;
type ReasoningEffort = (typeof ReasoningEffort)[number];

export const TrustedTokensModel = z.object({
  id: z.string().min(1),
  name: z.string().min(1).optional(),
  description: z.string().min(1).optional(),
  created: z.number().int().nonnegative().optional(),
  architecture: z.object({
    modality: z.string().optional(),
    input_modalities: z.array(z.string()).optional(),
    output_modalities: z.array(z.string()).optional(),
  }).passthrough().optional(),
  top_provider: z.object({
    is_moderated: z.boolean().optional(),
    context_length: z.number().int().positive().optional(),
  }).passthrough().optional(),
  pricing: z.object({
    prompt: Price,
    completion: Price,
    input_cache_read: Price.optional(),
  }).passthrough().optional(),
  supported_parameters: z.array(z.string()).optional(),
  context_length: z.number().int().positive().optional(),
  reasoning: z.object({
    mandatory: z.boolean().nullish(),
    default_enabled: z.boolean().nullish(),
    supported_efforts: z.array(z.string()).nullish(),
    default_effort: z.string().nullish(),
  }).passthrough().optional(),
  hugging_face_id: z.string().optional(),
}).passthrough();

export const TrustedTokensResponse = z.object({
  data: z.array(TrustedTokensModel),
}).passthrough();

export type TrustedTokensModel = z.infer<typeof TrustedTokensModel>;

export const trustedtokens = {
  id: "trustedtokens",
  name: "TrustedTokens",
  modelsDir: "providers/trustedtokens/models",
  deleteMissing: true,
  missingModelID(model) {
    if (IGNORED_IDS.has(model.id)) return undefined;
    // A `provider/lab/model` route variant is not a canonical model; its route
    // entry has to be authored by hand instead of opening an issue per variant.
    return isRouteVariant(model.id) ? undefined : model.id;
  },
  sourceNotices(models) {
    const notices: string[] = [];
    const ignored = models.filter((model) => IGNORED_IDS.has(model.id));
    if (ignored.length > 0) {
      notices.push(
        `${ignored.length} TrustedTokens model(s) are intentionally not published until a models/ lab entry exists: ${ignored.map((model) => `\`${model.id}\``).join(", ")}.`,
      );
    }
    const variants = models.filter((model) => isRouteVariant(model.id));
    if (variants.length > 0) {
      notices.push(
        `${variants.length} TrustedTokens provider-route variants (\`provider/lab/model\`) are not synced.`,
        `Skipped route IDs: ${variants.map((model) => `\`${model.id}\``).join(", ")}`,
      );
    }
    return notices;
  },
  async fetchModels() {
    const key = process.env.TRUSTEDTOKENS_API_KEY;
    if (key === undefined || key === "") {
      throw new Error("TrustedTokens sync requires TRUSTEDTOKENS_API_KEY");
    }
    return fetchTrustedTokensModels(key);
  },
  parseModels(raw) {
    const { data } = TrustedTokensResponse.parse(raw);
    if (data.length < MIN_MODELS) {
      throw new Error(
        `TrustedTokens catalog returned ${data.length} models; refusing to sync a truncated catalog`,
      );
    }
    return data;
  },
  translateModel(model, context) {
    if (IGNORED_IDS.has(model.id)) return undefined;
    const existing = context.existing(model.id);
    const baseModel = existing?.base_model ?? resolveTrustedTokensBaseModel(model.id);
    if (baseModel === undefined) return undefined;
    const plan = modalityPlan(model, existing, baseModel);
    const built = buildTrustedTokensModel(model, existing, baseModel, plan);
    if (built === undefined) return undefined;
    return {
      id: model.id,
      model: built,
      header: HEADER,
      v2Set: {
        ...cacheV2Set(built.cost?.cache_read, context.v2?.(model.id)),
        ...(plan.v2 === undefined ? {} : { modalities: { input: plan.v2, output: ["text"] } }),
      },
    };
  },
} satisfies SyncProvider<TrustedTokensModel>;

export async function fetchTrustedTokensModels(key: string, fetcher: typeof fetch = fetch) {
  const endpoint = process.env.TRUSTEDTOKENS_MODELS_URL ?? API_ENDPOINT;
  const response = await fetcher(endpoint, {
    headers: {
      Accept: "application/json",
      Authorization: `Bearer ${key}`,
    },
  });
  if (!response.ok) {
    throw new Error(`TrustedTokens models request failed: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

// TrustedTokens namespaces models by upstream lab (`Qwen`, `zai-org`,
// `deepseek-ai`) and keeps quantization or vendor markers in a few slugs.
const ORG_ALIASES: Record<string, string | undefined> = {
  qwen: "alibaba",
  "deepseek-ai": "deepseek",
  "zai-org": "zhipuai",
};

const QUANTIZATION_SUFFIX = /-(?:fp8|fp4|bf16|awq|gptq|gguf|int4|int8)$/i;

export function resolveTrustedTokensBaseModel(modelID: string) {
  const segments = modelID.split("/");
  if (segments.length !== 2) return undefined;
  const [org, slug] = segments as [string, string];
  const lab = ORG_ALIASES[org.toLowerCase()] ?? org;
  const candidates = new Set<string>([slug]);
  const unquantized = slug.replace(QUANTIZATION_SUFFIX, "");
  candidates.add(unquantized);
  // TrustedTokens prefixes NVIDIA's own models with the vendor name.
  candidates.add(unquantized.replace(/^nvidia-/i, ""));

  for (const candidate of candidates) {
    const resolved = resolveModelMetadataBaseModel(`${lab}/${candidate}`);
    if (resolved !== undefined) return resolved;
  }
  return undefined;
}

export function buildTrustedTokensModel(
  model: TrustedTokensModel,
  existing: ExistingModel | undefined,
  baseModel: string,
  plan = modalityPlan(model, existing, baseModel),
): SyncedModel | undefined {
  const input = perMillion(model.pricing?.prompt);
  const output = perMillion(model.pricing?.completion);
  if (input === undefined || output === undefined) return undefined;

  const cacheRead = perMillion(model.pricing?.input_cache_read);
  const labLimit = modelMetadata(baseModel).limit as { context?: number; output?: number } | undefined;
  const sourceContext = model.context_length ?? model.top_provider?.context_length;
  const context = servedContext(sourceContext, labLimit?.context) ?? existing?.limit?.context;
  if (context === undefined) return undefined;

  const reasoning = model.reasoning === undefined ? existing?.reasoning : true;
  const limit = {
    context,
    // TrustedTokens publishes no completion cap, so the lab's value survives.
    output: existing?.limit?.output,
  };
  const cost: SyncedFullModel["cost"] = {
    ...existing?.cost,
    input,
    output,
    cache_read: cacheRead,
  };

  return factorBaseModel(
    baseModel,
    {
      name: existing?.name,
      description: existing?.description,
      attachment: plan.v1 === undefined
        ? existing?.attachment
        : plan.v1.some((value) => value !== "text"),
      reasoning,
      reasoning_options: reasoningOptions(model, existing, reasoning),
      // Measured: every reasoner returns its trace in `reasoning_content`.
      interleaved: reasoning === true
        ? { field: "reasoning_content" }
        : existing?.interleaved,
      tool_call: supportedParameter(model, "tools") ?? existing?.tool_call,
      // DiffusionGemma rejects `temperature` with 422; every other row accepts it.
      temperature: supportedParameter(model, "temperature") ?? existing?.temperature,
      structured_output: STRUCTURED_OUTPUT_OVERRIDES[model.id]
        ?? supportedParameter(model, "response_format")
        ?? existing?.structured_output,
      status: existing?.status,
      provider: existing?.provider,
      experimental: existing?.experimental,
      cost,
      limit,
      modalities: plan.v1 === undefined
        ? undefined
        : { input: plan.v1, output: ["text"] },
    },
    limit,
    existing?.base_model === baseModel ? existing?.base_model_omit : undefined,
  );
}

// Only the default provider's bare ids are canonical models. Every other route
// is listed as `provider/lab/model`, which needs a hand-authored catalog entry.
function isRouteVariant(modelID: string) {
  return modelID.split("/").length > 2;
}

function reasoningOptions(
  model: TrustedTokensModel,
  existing: ExistingModel | undefined,
  reasoning: boolean | undefined,
): SyncedFullModel["reasoning_options"] {
  if (reasoning !== true) return undefined;
  const curated = REASONING_OVERRIDES[model.id];
  if (curated !== undefined) return [{ type: "effort", values: curated }];
  if (model.reasoning === undefined) return existing?.reasoning_options;
  const present = new Set((model.reasoning.supported_efforts ?? []).map((value) => value.toLowerCase()));
  const efforts: ReasoningEffort[] = ReasoningEffort.filter((value) => present.has(value));
  // An empty list means "no caller control", not an invented toggle or budget.
  return efforts.length === 0 ? [] : [{ type: "effort", values: efforts }];
}

function supportedParameter(model: TrustedTokensModel, parameter: string) {
  if (model.supported_parameters === undefined) return undefined;
  return model.supported_parameters.includes(parameter);
}

// This host advertises images generically and rejects video and PDF parts with
// 422. Inherit the lab's image MIME list (minus those rejected formats) instead
// of replacing a precise list with a generic `image`. v1 can only carry the
// generic categories, so the precise list is written through `v2Set`.
export interface ModalityPlan {
  v1?: SyncedFullModel["modalities"]["input"];
  v2?: string[];
}

export function modalityPlan(
  model: TrustedTokensModel,
  existing: ExistingModel | undefined,
  baseModel: string,
): ModalityPlan {
  const advertised = model.architecture?.input_modalities;
  if (advertised === undefined) return { v1: existing?.modalities?.input };

  const acceptsImage = advertised.some((value) => /^images?$/i.test(value));
  const labInput = (modelMetadata(baseModel).modalities as { input?: string[] } | undefined)?.input
    ?? existing?.modalities?.input
    ?? ["text"];
  const labHasVisual = labInput.some(isVisualModality);

  if (!acceptsImage) {
    return labHasVisual ? { v1: ["text"], v2: ["text"] } : {};
  }
  if (!labHasVisual) return { v1: ["text", "image"], v2: ["text", "image"] };
  if (!labInput.some(isVideoOrPdf)) return {};

  const inherited = (rawLabInput(baseModel) ?? labInput)
    .filter((value) => value === "text" || value === "image" || value.startsWith("image/"));
  return inherited.length > 0
    ? { v1: ["text", "image"], v2: [...new Set(inherited)] }
    : { v1: ["text", "image"], v2: ["text", "image"] };
}

const LABS_DIR = path.join(import.meta.dirname, "..", "..", "..", "..", "..", "models");

function rawLabInput(baseModel: string) {
  try {
    const doc = Bun.TOML.parse(readFileSync(path.join(LABS_DIR, `${baseModel}.toml`), "utf8")) as {
      modalities?: { input?: unknown };
    };
    const input = doc.modalities?.input;
    return Array.isArray(input)
      ? input.filter((value): value is string => typeof value === "string")
      : undefined;
  } catch {
    return undefined;
  }
}

function isVisualModality(value: string) {
  return value === "image" || value.startsWith("image/")
    || value === "video" || value.startsWith("video/")
    || value === "pdf" || value === "application/pdf";
}

function isVideoOrPdf(value: string) {
  return value === "video" || value.startsWith("video/")
    || value === "pdf" || value === "application/pdf";
}

function cachedCapabilities(current: Record<string, unknown> | undefined) {
  const capabilities = current?.capabilities;
  return typeof capabilities === "object" && capabilities !== null && !Array.isArray(capabilities)
    ? capabilities as Record<string, unknown>
    : undefined;
}

// The catalog prices cached input for every row, and repeated prompts report
// cached_tokens after a cold first request, so a new entry gets implicit
// caching. An authored cache kind is never replaced.
export function cacheV2Set(
  cacheRead: number | undefined,
  current: Record<string, unknown> | undefined,
) {
  if (cachedCapabilities(current)?.cache !== undefined) return undefined;
  if (cacheRead === undefined) return undefined;
  return { "capabilities.cache": { implicit: true } };
}

// TrustedTokens may advertise more context than the lab model documents (e.g.
// 1,048,576 for a 1M model). Publish the lower value so clients never send a
// prompt the upstream model cannot serve.
function servedContext(source: number | undefined, lab: number | undefined) {
  if (source === undefined) return undefined;
  return lab === undefined ? source : Math.min(source, lab);
}

function perMillion(value: string | number | undefined) {
  const parsed = typeof value === "string" ? Number(value) : value;
  if (parsed === undefined || !Number.isFinite(parsed) || parsed < 0) return undefined;
  return Math.round(parsed * 1_000_000 * 1_000_000) / 1_000_000;
}
