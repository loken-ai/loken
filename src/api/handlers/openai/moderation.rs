//! `/v1/moderations` over a local safety classifier. Five families are served, each
//! with the protocol its model card describes: Llama Guard 3 answers `safe` or `unsafe`
//! with hazard codes in one pass; Shieldstral, ShieldGemma and Granite Guardian answer
//! one yes/no question per policy, the probability read from the first token; gpt-oss
//! Safeguard reasons over a written policy and names the violated categories. Every
//! family lands in the Moderation API's thirteen categories.

use super::*;
use axum::response::{IntoResponse, Response};
use serde_json::{json, Value};
use std::collections::BTreeMap;

const MODERATION_CATEGORIES: [&str; 13] = [
    "harassment",
    "harassment/threatening",
    "hate",
    "hate/threatening",
    "illicit",
    "illicit/violent",
    "self-harm",
    "self-harm/intent",
    "self-harm/instructions",
    "sexual",
    "sexual/minors",
    "violence",
    "violence/graphic",
];

/// The Moderation API's own definition of each category, the policy text a
/// policy-adaptive classifier is asked about.
const CATEGORY_DEFINITIONS: [(&str, &str); 13] = [
    ("harassment", "expresses, incites, or promotes harassing language towards any target"),
    ("harassment/threatening", "is harassing and also includes violence or serious harm towards any target"),
    ("hate", "expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste"),
    ("hate/threatening", "is hateful and also includes violence or serious harm towards the targeted group"),
    ("illicit", "gives advice or instruction on how to commit illicit acts, such as how to shoplift"),
    ("illicit/violent", "gives advice or instruction on illicit acts that involve violence or procuring a weapon"),
    ("self-harm", "promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders"),
    ("self-harm/intent", "has the speaker express that they are engaging or intend to engage in self-harm, such as suicide, cutting, and eating disorders"),
    ("self-harm/instructions", "encourages performing acts of self-harm or gives instructions or advice on how to commit them"),
    ("sexual", "is meant to arouse sexual excitement, such as the description of sexual activity, or promotes sexual services, excluding sex education and wellness"),
    ("sexual/minors", "is sexual content that includes an individual who is under 18 years old"),
    ("violence", "depicts death, violence, or physical injury"),
    ("violence/graphic", "depicts death, violence, or physical injury in graphic detail"),
];

/// The classifier families, told apart by the model's name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    LlamaGuard,
    Shieldstral,
    ShieldGemma,
    GraniteGuardian,
    GptOssSafeguard,
}

impl Family {
    fn of(model: &str) -> Option<Self> {
        let m = model.to_ascii_lowercase();
        if m.contains("llama-guard") || m.contains("llamaguard") {
            Some(Self::LlamaGuard)
        } else if m.contains("shieldstral") {
            Some(Self::Shieldstral)
        } else if m.contains("shieldgemma") {
            Some(Self::ShieldGemma)
        } else if m.contains("guardian") {
            Some(Self::GraniteGuardian)
        } else if m.contains("safeguard") {
            Some(Self::GptOssSafeguard)
        } else {
            None
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::LlamaGuard => "llama-guard",
            Self::Shieldstral => "shieldstral",
            Self::ShieldGemma => "shieldgemma",
            Self::GraniteGuardian => "granite-guardian",
            Self::GptOssSafeguard => "gpt-oss-safeguard",
        }
    }
}

/// What a classifier said about one input, in the API's terms.
#[derive(Debug, Default)]
struct Judgement {
    flagged: bool,
    /// Per category, the confidence the classifier put on it.
    scores: BTreeMap<&'static str, f64>,
    /// The classifier's own labels, for families whose taxonomy has more than the API.
    hazards: Vec<String>,
    /// The classifier's overall probability that the input is unsafe.
    unsafe_probability: f64,
}

/// A judgement threshold: the API flags a category above one half, and the model cards
/// read their yes/no scores the same way.
const FLAG_THRESHOLD: f64 = 0.5;

/// Alternatives the classifier is asked to report for its first token; the model cards
/// read yes/no probabilities from the top twenty.
const FIRST_TOKEN_ALTERNATIVES: usize = 20;

// ------------------------------------------------------------------ asking the model

/// Runs a raw prompt through `/v1/completions` and returns the text and the
/// alternatives reported for the first generated token.
async fn ask_raw(
    state: &APIServer,
    model: &str,
    prompt: String,
    max_tokens: usize,
) -> Result<(String, BTreeMap<String, f64>), ApiError> {
    let req = json!({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "logprobs": FIRST_TOKEN_ALTERNATIVES,
    });
    let resp = Box::pin(text_completions(State(state.clone()), Json(req))).await?;
    let v = body_json(resp).await?;
    let text = v
        .pointer("/choices/0/text")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let top = v
        .pointer("/choices/0/logprobs/top_logprobs/0")
        .and_then(Value::as_object)
        .map(|m| {
            m.iter()
                .filter_map(|(t, lp)| lp.as_f64().map(|l| (t.clone(), l)))
                .collect()
        })
        .unwrap_or_default();
    Ok((text, top))
}

/// Runs a chat exchange through `/v1/chat/completions`, with the model's own template,
/// and returns the answer and the alternatives reported for its first token.
async fn ask_chat(
    state: &APIServer,
    model: &str,
    system: &str,
    user: &str,
    max_tokens: usize,
    reasoning_effort: Option<&str>,
) -> Result<(String, BTreeMap<String, f64>), ApiError> {
    let mut req = json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_completion_tokens": max_tokens,
        "temperature": 0,
        "logprobs": true,
        "top_logprobs": FIRST_TOKEN_ALTERNATIVES,
    });
    if let Some(effort) = reasoning_effort {
        req["reasoning_effort"] = json!(effort);
    }
    let req: ChatCompletionRequest =
        serde_json::from_value(req).map_err(|e| ApiError::Internal(format!("moderation: {e}")))?;
    let resp = Box::pin(chat_completion(State(state.clone()), OpenAIJson(req))).await?;
    let v = body_json(resp).await?;
    let text = v
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let top = v
        .pointer("/choices/0/logprobs/content/0/top_logprobs")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|e| {
                    Some((
                        e.get("token")?.as_str()?.to_string(),
                        e.get("logprob")?.as_f64()?,
                    ))
                })
                .collect()
        })
        .unwrap_or_default();
    Ok((text, top))
}

async fn body_json(resp: Response) -> Result<Value, ApiError> {
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .map_err(|e| ApiError::Internal(format!("moderation: {e}")))?;
    Ok(serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// The probability the classifier put on `positive` against `negative` as its first
/// token, from the alternatives reported for it: the mass on each word's spellings,
/// renormalised between the two, as the model cards compute their scores.
fn first_token_probability(top: &BTreeMap<String, f64>, positive: &str, negative: &str) -> f64 {
    let mass = |word: &str| {
        top.iter()
            .filter(|(t, _)| {
                t.trim()
                    .trim_matches(|c: char| c == '"' || c == '\'' || c == '.' || c == ',')
                    .eq_ignore_ascii_case(word)
            })
            .map(|(_, lp)| lp.exp())
            .sum::<f64>()
    };
    let (p, n) = (mass(positive), mass(negative));
    if p + n > 0.0 {
        p / (p + n)
    } else {
        0.0
    }
}

// --------------------------------------------------------------- Llama Guard 3

/// The hazard taxonomy Llama Guard 3 classifies against, as its model card lists it,
/// and the categories of the Moderation API each hazard lands in. Hazards the API has
/// no category for still flag the input and are reported under `hazards`.
const GUARD_HAZARDS: [(&str, &str, &[&str]); 13] = [
    ("S1", "Violent Crimes", &["violence", "illicit/violent"]),
    ("S2", "Non-Violent Crimes", &["illicit"]),
    ("S3", "Sex Crimes", &["sexual", "illicit"]),
    ("S4", "Child Exploitation", &["sexual/minors"]),
    ("S5", "Defamation", &["harassment"]),
    ("S6", "Specialized Advice", &[]),
    ("S7", "Privacy", &[]),
    ("S8", "Intellectual Property", &[]),
    (
        "S9",
        "Indiscriminate Weapons",
        &["violence", "illicit/violent"],
    ),
    ("S10", "Hate", &["hate"]),
    ("S11", "Self-Harm", &["self-harm"]),
    ("S12", "Sexual Content", &["sexual"]),
    ("S13", "Elections", &[]),
];

/// The prompt Llama Guard 3 was trained on, for one user message. Sent raw; the
/// model's own template says the same with constructs the template engine does not
/// render.
fn guard_prompt(text: &str) -> String {
    let policy: String = GUARD_HAZARDS
        .iter()
        .map(|(code, name, _)| format!("{code}: {name}."))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n\
         Task: Check if there is unsafe content in 'User' messages in conversations according \
         our safety policy with the below categories.\n\n\
         <BEGIN UNSAFE CONTENT CATEGORIES>\n{policy}\n<END UNSAFE CONTENT CATEGORIES>\n\n\
         <BEGIN CONVERSATION>\n\nUser: {text}\n\n<END CONVERSATION>\n\n\
         Provide your safety assessment for ONLY THE LAST User message in the above conversation:\n\
         \x20- First line must read 'safe' or 'unsafe'.\n\
         \x20- If unsafe, a second line must include a comma-separated list of violated categories.\
         <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
}

/// Reads Llama Guard's answer: `safe`, or `unsafe` then the violated hazard codes.
fn parse_guard(answer: &str) -> (bool, Vec<String>) {
    let mut lines = answer.trim().lines();
    let verdict = lines.next().unwrap_or("").trim().to_ascii_lowercase();
    if verdict != "unsafe" {
        return (false, Vec::new());
    }
    let codes = lines
        .next()
        .unwrap_or("")
        .split(',')
        .map(|c| c.trim().to_ascii_uppercase())
        .filter(|c| GUARD_HAZARDS.iter().any(|(code, _, _)| *code == c))
        .collect();
    (true, codes)
}

async fn judge_llama_guard(
    state: &APIServer,
    model: &str,
    text: &str,
) -> Result<Judgement, ApiError> {
    let (answer, top) = ask_raw(state, model, guard_prompt(text), 16).await?;
    let (flagged, codes) = parse_guard(&answer);
    let p_unsafe = if top.is_empty() {
        if flagged {
            1.0
        } else {
            0.0
        }
    } else {
        first_token_probability(&top, "unsafe", "safe")
    };
    let mut j = Judgement {
        flagged,
        unsafe_probability: p_unsafe,
        ..Default::default()
    };
    for code in &codes {
        if let Some((_, _, cats)) = GUARD_HAZARDS.iter().find(|(c, _, _)| c == code) {
            for c in cats.iter() {
                j.scores.insert(c, p_unsafe);
            }
        }
    }
    j.hazards = codes;
    Ok(j)
}

// ------------------------------------------------------------------- Shieldstral

const SHIELDSTRAL_SYSTEM: &str = "Judge whether the Document meets the requirements based on the Query and the Instruction provided. Note that the answer can only be \"yes\" or \"no\".";
const SHIELDSTRAL_INSTRUCT: &str = "You are a strict safety moderator reviewing potentially harmful content. Apply a low tolerance threshold.";

/// Shieldstral's exchange in the Mistral-3 instruct framing, sent raw: the community
/// GGUF carries no chat template, and the model's card describes exactly this system
/// sentence and the three-tag user message.
fn shieldstral_prompt(query: &str, text: &str) -> String {
    format!(
        "<s>[SYSTEM_PROMPT]{SHIELDSTRAL_SYSTEM}[/SYSTEM_PROMPT][INST]<Instruct>: {SHIELDSTRAL_INSTRUCT}\n\n<Query>: {query}\n\n<Document>: {text}[/INST]"
    )
}

/// One gate question over every category, then one question per category when the gate
/// says yes: a safe input costs one pass instead of fourteen.
async fn judge_shieldstral(
    state: &APIServer,
    model: &str,
    text: &str,
) -> Result<Judgement, ApiError> {
    let all: String = CATEGORY_DEFINITIONS
        .iter()
        .map(|(_, d)| format!("- {d}"))
        .collect::<Vec<_>>()
        .join("\n");
    let gate = format!("Does this content do any of the following?\n{all}");
    let (_, top) = ask_raw(state, model, shieldstral_prompt(&gate, text), 1).await?;
    let p_any = first_token_probability(&top, "yes", "no");
    let mut j = Judgement {
        flagged: p_any >= FLAG_THRESHOLD,
        unsafe_probability: p_any,
        ..Default::default()
    };
    if !j.flagged {
        return Ok(j);
    }
    for (cat, def) in CATEGORY_DEFINITIONS {
        let q = format!("Does this content {def}?");
        let (_, top) = ask_raw(state, model, shieldstral_prompt(&q, text), 1).await?;
        j.scores
            .insert(cat, first_token_probability(&top, "yes", "no"));
    }
    Ok(j)
}

// ------------------------------------------------------------------- ShieldGemma

/// ShieldGemma's frame for a user prompt, as its Ollama template renders it, with one
/// principle. The four principles the model was trained on come first, as a gate; the
/// API's categories are then asked one at a time in the same words.
fn shieldgemma_prompt(text: &str, principles: &str) -> String {
    format!(
        "You are a policy expert trying to help determine whether a user prompt is in violation of the defined safety policies.\n\n\
         <start_of_turn>\nHuman Question: {text}\n<end_of_turn>\n\n\
         Our safety principle is defined in the below:\n\n{principles}\n\n===\n\n\
         Does the human question violate the above principle? Your answer must start with 'Yes' or 'No'. And then walk through step by step to be sure we answer correctly.\n\n"
    )
}

/// The four principles ShieldGemma was trained on, in its own words, and the API
/// categories each lands in. Asked together as a gate, then one at a time: the model
/// reads principles it knows; ones written for it in other words it answers too freely.
const SHIELDGEMMA_PRINCIPLES: [(&str, &str, &[&str]); 4] = [
    ("No Dangerous Content", "The prompt shall not contain or seek generation of content that harming oneself and/or others (e.g., accessing or building firearms and explosive devices, promotion of terrorism, instructions for suicide).", &["illicit/violent", "violence", "self-harm"]),
    ("No Harassment", "The prompt shall not contain or seek generation of content that is malicious, intimidating, bullying, or abusive content targeting another individual (e.g., physical threats, denial of tragic events, disparaging victims of violence).", &["harassment", "harassment/threatening"]),
    ("No Hate Speech", "The prompt shall not contain or seek generation of content that targeting identity and/or protected attributes (e.g., racial slurs, promotion of discrimination, calls to violence against protected groups) or dehumanizing/belittling/vilifying on the bases of race, ethnicity, religion, disability, age, nationality, veteran status, sexual orientation, gender, gender identity, caste, or any other protected status. We do allow content that includes a reference to a protected individual or group in a positive manner.", &["hate", "hate/threatening"]),
    ("No Sexually Explicit Information", "The prompt shall not contain or seek generation of content that contains references to sexual acts or other lewd content (e.g., sexually graphic descriptions, content aimed at causing arousal). Medical or scientific terms related to human anatomy or sex education are permitted.", &["sexual"]),
];

fn shieldgemma_principle(name: &str, text: &str) -> String {
    format!("* \"{name}\": {text}")
}

async fn judge_shieldgemma(
    state: &APIServer,
    model: &str,
    text: &str,
) -> Result<Judgement, ApiError> {
    let all: String = SHIELDGEMMA_PRINCIPLES
        .iter()
        .map(|(n, t, _)| shieldgemma_principle(n, t))
        .collect::<Vec<_>>()
        .join("\n");
    let (_, top) = ask_raw(state, model, shieldgemma_prompt(text, &all), 1).await?;
    let p_any = first_token_probability(&top, "yes", "no");
    let mut j = Judgement {
        flagged: p_any >= FLAG_THRESHOLD,
        unsafe_probability: p_any,
        ..Default::default()
    };
    if !j.flagged {
        return Ok(j);
    }
    for (name, principle, cats) in SHIELDGEMMA_PRINCIPLES {
        let (_, top) = ask_raw(
            state,
            model,
            shieldgemma_prompt(text, &shieldgemma_principle(name, principle)),
            1,
        )
        .await?;
        let p = first_token_probability(&top, "yes", "no");
        if p >= FLAG_THRESHOLD {
            j.hazards.push(name.to_string());
        }
        for c in cats.iter() {
            j.scores.insert(c, p);
        }
    }
    Ok(j)
}

// ------------------------------------------------------------- Granite Guardian

/// Granite Guardian's risks for a user message, their definitions as the model's
/// template carries them, and the API categories each lands in. `harm` is the gate.
const GRANITE_RISKS: [(&str, &str, &[&str]); 7] = [
    ("harm", "The 'User' message is harmful by common-sense.", &[]),
    ("violence", "The 'User' message contains content promoting or describing physical harm to individuals or groups. This includes depictions of assault, self-harm, or creation of a hostile environment through threats or intimidation.", &["violence"]),
    ("social_bias", "The 'User' message contains systemic prejudice against groups based on shared identity or characteristics, often stemming from stereotypes or cultural influences. This can manifest in thoughts, attitudes, or behaviors that unfairly favor or disfavor certain groups over others.", &["hate"]),
    ("sexual_content", "The 'User' message contains material explicitly related to sexual activities, anatomy, or desires. This ranges from discussions of sexual practices to graphic depictions of sexual acts, excluding content related to general relationships or non-sexual intimacy.", &["sexual"]),
    ("unethical_behavior", "The 'User' message contains actions that violate moral or professional standards, focusing on exploitation for personal gain or disregard for others' well-being. This category excludes physical violence or explicit bias, instead covering activities like fraud, theft, or abuse of power.", &["illicit"]),
    ("profanity", "The 'User' message contains the use of language considered offensive or socially unacceptable in a given context. This primarily refers to curse words, swear words, and crude expressions, excluding slurs or derogatory terms targeting specific groups.", &["harassment"]),
    ("jailbreak", "The 'User' message contains the deliberate circumvention of AI systems' built-in safeguards or ethical guidelines. This involves crafting specific prompts or scenarios designed to manipulate the AI into generating restricted or inappropriate content.", &[]),
];

fn granite_prompt(text: &str, definition: &str) -> String {
    format!(
        "<|start_of_role|>user<|end_of_role|>You are a safety agent trying to help determine whether the user message is harmful according to the safety risk definition below.\n\n\
         <start_of_turn>\nUser Message: {text}\n<end_of_turn>\n\n\
         Our safety risk definition is defined below:\n<start_of_risk_definition>\n* {definition}\n<end_of_risk_definition>\n\n\
         Is the user message harmful based on the risk definition? Your answer must be either 'Yes' or 'No'.<|end_of_text|>\n\
         <|start_of_role|>assistant<|end_of_role|>"
    )
}

async fn judge_granite(state: &APIServer, model: &str, text: &str) -> Result<Judgement, ApiError> {
    let (_, top) = ask_raw(state, model, granite_prompt(text, GRANITE_RISKS[0].1), 1).await?;
    let p_harm = first_token_probability(&top, "yes", "no");
    let mut j = Judgement {
        flagged: p_harm >= FLAG_THRESHOLD,
        unsafe_probability: p_harm,
        ..Default::default()
    };
    if !j.flagged {
        return Ok(j);
    }
    for (risk, def, cats) in GRANITE_RISKS.iter().skip(1) {
        let (_, top) = ask_raw(state, model, granite_prompt(text, def), 1).await?;
        let p = first_token_probability(&top, "yes", "no");
        if p >= FLAG_THRESHOLD {
            j.hazards.push(risk.to_string());
        }
        for c in cats.iter() {
            j.scores.insert(c, p);
        }
    }
    Ok(j)
}

// ------------------------------------------------------------ gpt-oss Safeguard

/// The policy gpt-oss Safeguard reasons over: the API's categories as its rules, and the
/// answer as a JSON object naming the violated ones, which the cookbook's policy format
/// allows. Reasoning stays low: the categories are plain and the input is one text.
fn safeguard_policy() -> String {
    let rules: String = CATEGORY_DEFINITIONS
        .iter()
        .map(|(cat, def)| format!("- \"{cat}\": content that {def}."))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "# Instruction\nYou are a content moderation classifier. Decide which of the categories below the user's content falls into. \
         Respond with exactly one JSON object and nothing else: {{\"violations\": [<category names>]}}. \
         Use an empty list when none applies.\n\n# Categories\n{rules}\n\n# Criteria\n\
         Judge the content itself, not the intent of the person asking about it. Educational, medical and news content that describes without promoting does not violate. \
         Quoting a slur to condemn it does not violate. Fiction violates when it depicts the category in earnest.\n"
    )
}

/// The verdict is the last JSON object of the answer: a reasoning model may think aloud
/// before it, and that thinking can itself quote an object.
fn parse_violations(answer: &str) -> Vec<String> {
    let Some(end) = answer.rfind('}') else {
        return Vec::new();
    };
    let head = &answer[..=end];
    let Some(start) = head.rfind('{') else {
        return Vec::new();
    };
    let Ok(v) = serde_json::from_str::<Value>(&head[start..]) else {
        return Vec::new();
    };
    v.get("violations")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str())
                .map(str::to_ascii_lowercase)
                .filter(|c| MODERATION_CATEGORIES.contains(&c.as_str()))
                .collect()
        })
        .unwrap_or_default()
}

async fn judge_safeguard(
    state: &APIServer,
    model: &str,
    text: &str,
) -> Result<Judgement, ApiError> {
    let (answer, _) = ask_chat(state, model, &safeguard_policy(), text, 256, Some("low")).await?;
    let hits = parse_violations(&answer);
    let mut j = Judgement {
        flagged: !hits.is_empty(),
        unsafe_probability: if hits.is_empty() { 0.0 } else { 1.0 },
        ..Default::default()
    };
    for h in &hits {
        if let Some(c) = MODERATION_CATEGORIES.iter().find(|c| **c == h.as_str()) {
            j.scores.insert(c, 1.0);
        }
    }
    Ok(j)
}

// ---------------------------------------------------------------------- endpoint

/// `POST /v1/moderations`: each input judged by the configured classifier, in the
/// protocol of its family. Without a configured model, 501 and the key to set.
pub(crate) async fn moderations(
    State(state): State<APIServer>,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    let Some(model) = state.default_inference_config.moderation_model.clone() else {
        return Ok(not_implemented(
            "no moderation model is configured; set `[inference] moderation_model` to a safety classifier: a Llama Guard 3, Shieldstral, ShieldGemma, Granite Guardian or gpt-oss Safeguard model",
        ));
    };
    let Some(family) = Family::of(&model) else {
        return Ok(not_implemented(&format!(
            "moderation model '{model}' is not of a family this server knows: llama-guard, shieldstral, shieldgemma, granite guardian, gpt-oss-safeguard"
        )));
    };
    let inputs: Vec<String> = match body.get("input") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        _ => {
            return Err(ApiError::Validation(
                "`input` must be a string or an array of strings".into(),
            ))
        }
    };
    let mut results = Vec::new();
    for text in inputs {
        let j = match family {
            Family::LlamaGuard => judge_llama_guard(&state, &model, &text).await?,
            Family::Shieldstral => judge_shieldstral(&state, &model, &text).await?,
            Family::ShieldGemma => judge_shieldgemma(&state, &model, &text).await?,
            Family::GraniteGuardian => judge_granite(&state, &model, &text).await?,
            Family::GptOssSafeguard => judge_safeguard(&state, &model, &text).await?,
        };
        let categories: serde_json::Map<String, Value> = MODERATION_CATEGORIES
            .iter()
            .map(|c| {
                (
                    c.to_string(),
                    json!(j.scores.get(c).is_some_and(|p| *p >= FLAG_THRESHOLD)),
                )
            })
            .collect();
        let scores: serde_json::Map<String, Value> = MODERATION_CATEGORIES
            .iter()
            .map(|c| {
                (
                    c.to_string(),
                    json!(j.scores.get(c).copied().unwrap_or(0.0)),
                )
            })
            .collect();
        let flagged = j.flagged || categories.values().any(|v| v.as_bool().unwrap_or(false));
        results.push(json!({
            "flagged": flagged,
            "categories": categories,
            "category_scores": scores,
            "hazards": j.hazards,
            "unsafe_probability": j.unsafe_probability,
        }));
    }
    Ok(Json(json!({
        "id": super::files::new_id("modr"),
        "model": model,
        "family": family.name(),
        "results": results,
    }))
    .into_response())
}

fn not_implemented(message: &str) -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(openai_error_body(
            StatusCode::NOT_IMPLEMENTED,
            message.to_string(),
        )),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn families_are_told_apart_by_name() {
        assert_eq!(Family::of("llama-guard3:8b"), Some(Family::LlamaGuard));
        assert_eq!(Family::of("shieldstral-q4_k_m"), Some(Family::Shieldstral));
        assert_eq!(Family::of("shieldgemma:2b"), Some(Family::ShieldGemma));
        assert_eq!(
            Family::of("granite3-guardian:2b"),
            Some(Family::GraniteGuardian)
        );
        assert_eq!(
            Family::of("gpt-oss-safeguard:20b"),
            Some(Family::GptOssSafeguard)
        );
        assert_eq!(Family::of("qwen3:8b"), None);
    }

    #[test]
    fn llama_guard_answers_are_read() {
        assert_eq!(parse_guard("safe"), (false, vec![]));
        assert_eq!(
            parse_guard("unsafe\nS1,S10"),
            (true, vec!["S1".into(), "S10".into()])
        );
        assert_eq!(parse_guard("unsafe\nS99, s2 "), (true, vec!["S2".into()]));
        for (_, _, cs) in GUARD_HAZARDS {
            for c in cs.iter() {
                assert!(MODERATION_CATEGORIES.contains(c), "{c}");
            }
        }
    }

    #[test]
    fn yes_no_probability_comes_from_the_first_token() {
        let mut top = BTreeMap::new();
        top.insert("Yes".to_string(), -0.05f64);
        top.insert(" no".to_string(), -3.0f64);
        top.insert("\"yes\"".to_string(), -6.0f64);
        let p = first_token_probability(&top, "yes", "no");
        assert!(p > 0.9 && p < 1.0, "{p}");
        assert_eq!(first_token_probability(&BTreeMap::new(), "yes", "no"), 0.0);
    }

    #[test]
    fn safeguard_violations_are_read_from_json_only() {
        assert_eq!(
            parse_violations("{\"violations\": [\"hate\", \"nonsense\"]}"),
            vec!["hate".to_string()]
        );
        assert!(parse_violations("I think this is fine.").is_empty());
        assert_eq!(
            parse_violations("Thus {\"violations\": [\"hate\"]}. Check again.assistantfinal {\"violations\": [\"violence\"]}"),
            vec!["violence".to_string()]
        );
        assert!(parse_violations("{\"violations\": []}").is_empty());
    }

    #[test]
    fn every_policy_prompt_carries_the_text() {
        for p in [
            guard_prompt("needle"),
            shieldstral_prompt("q", "needle"),
            shieldgemma_prompt("needle", "* principle"),
            granite_prompt("needle", GRANITE_RISKS[0].1),
        ] {
            assert!(p.contains("needle"));
        }
        assert!(safeguard_policy().contains("\"sexual/minors\""));
        for (_, _, cats) in GRANITE_RISKS.iter().chain(SHIELDGEMMA_PRINCIPLES.iter()) {
            for c in cats.iter() {
                assert!(MODERATION_CATEGORIES.contains(c), "{c}");
            }
        }
    }
}
