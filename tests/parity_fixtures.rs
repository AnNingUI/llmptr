use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::str::FromStr;

use serde::Deserialize;
use serde_json::Value;
use llmptr::{
    Format, translate_non_stream, translate_request, translate_stream, translate_token_count,
};

#[derive(Debug, Deserialize)]
struct Fixture {
    name: String,
    kind: String,
    from: String,
    to: String,
    model: String,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    compare_go: bool,
    input: Value,
    #[serde(default)]
    inputs: Vec<Value>,
    #[serde(default)]
    original_request: Value,
    #[serde(default)]
    translated_request: Value,
    #[serde(default)]
    count: i64,
    expected: Option<Value>,
}

#[test]
fn parity_request_fixtures() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/parity");
    let mut files: Vec<_> = fs::read_dir(&root)
        .unwrap_or_else(|err| panic!("failed to read fixture dir {}: {err}", root.display()))
        .map(|entry| entry.expect("fixture dir entry").path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("json"))
        .collect();
    files.sort();

    assert!(!files.is_empty(), "expected at least one parity fixture");

    for path in files {
        let raw = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("failed to read fixture {}: {err}", path.display()));
        let fixture: Fixture = serde_json::from_str(&raw)
            .unwrap_or_else(|err| panic!("failed to parse fixture {}: {err}", path.display()));

        let from = Format::from_str(&fixture.from)
            .unwrap_or_else(|_| panic!("invalid from format in {}", fixture.name));
        let to = Format::from_str(&fixture.to)
            .unwrap_or_else(|_| panic!("invalid to format in {}", fixture.name));

        let mut actual = match fixture.kind.as_str() {
            "request" => translate_request(from, to, &fixture.model, fixture.input, fixture.stream),
            "response_stream" => {
                let mut param: Box<dyn std::any::Any> = Box::new(());
                let stream_inputs = if fixture.inputs.is_empty() {
                    vec![fixture.input]
                } else {
                    fixture.inputs
                };
                let mut chunks = Vec::new();
                for input in stream_inputs {
                    let input = normalize_stream_input_for_rust(input);
                    chunks.extend(translate_stream(
                        from,
                        to,
                        &fixture.model,
                        &fixture.original_request,
                        &fixture.translated_request,
                        input,
                        Some(&mut param),
                    ));
                }
                Value::Array(
                    chunks
                        .into_iter()
                        .map(|chunk| match chunk {
                            Value::String(s) => Value::String(s),
                            other => Value::String(serde_json::to_string(&other).unwrap()),
                        })
                        .collect(),
                )
            }
            "response_non_stream" => translate_non_stream(
                from,
                to,
                &fixture.model,
                &fixture.original_request,
                &fixture.translated_request,
                fixture.input,
                None,
            ),
            "token_count" => translate_token_count(from, to, fixture.count, fixture.input),
            other => panic!("unsupported fixture kind {other:?} in {}", fixture.name),
        };
        let mut expected = if fixture.compare_go {
            go_golden(&path)
        } else {
            fixture
                .expected
                .unwrap_or_else(|| panic!("fixture {} missing expected", fixture.name))
        };
        normalize_unstable_fields(&mut actual);
        normalize_unstable_fields(&mut expected);
        assert_eq!(actual, expected, "fixture {}", fixture.name);
    }
}

fn normalize_stream_input_for_rust(input: Value) -> Value {
    let Some(raw) = input.as_str() else {
        return input;
    };
    if raw == "[DONE]" {
        return Value::String(raw.to_string());
    }
    let trimmed = raw.trim();
    let payload = trimmed
        .strip_prefix("data:")
        .map(str::trim)
        .unwrap_or(trimmed);
    serde_json::from_str(payload).unwrap_or(input)
}

fn normalize_unstable_fields(value: &mut Value) {
    normalize_sse_event_strings(value);
    let mut generated_call_ids = HashMap::new();
    let mut generated_function_ids = HashMap::new();
    normalize_generated_ids(value, &mut generated_call_ids, &mut generated_function_ids);
    if let Some(user_id) = value.pointer_mut("/metadata/user_id")
        && user_id.is_string()
    {
        *user_id = Value::String("__normalized_user_id__".to_string());
    }
    normalize_create_time(value);
    normalize_created_timestamp(value);
    normalize_created_at_timestamp(value);
}

fn normalize_generated_ids(
    value: &mut Value,
    call_ids: &mut HashMap<String, String>,
    function_ids: &mut HashMap<String, String>,
) {
    match value {
        Value::Object(obj) => {
            for child in obj.values_mut() {
                normalize_generated_ids(child, call_ids, function_ids);
            }
        }
        Value::Array(items) => {
            for item in items {
                normalize_generated_ids(item, call_ids, function_ids);
            }
        }
        Value::String(s) => {
            if is_generated_call_id(s) {
                let next = call_ids.len();
                let normalized = call_ids
                    .entry(s.clone())
                    .or_insert_with(|| format!("__normalized_call_id_{next}__"))
                    .clone();
                *s = normalized;
            } else if let Some(call_id) =
                s.strip_prefix("fc_").filter(|id| is_generated_call_id(id))
            {
                let next = call_ids.len();
                let normalized = call_ids
                    .entry(call_id.to_string())
                    .or_insert_with(|| format!("__normalized_call_id_{next}__"))
                    .clone();
                *s = format!("fc_{normalized}");
            } else if is_generated_function_call_id(s) {
                let next = function_ids.len();
                let normalized = function_ids
                    .entry(s.clone())
                    .or_insert_with(|| format!("__normalized_function_id_{next}__"))
                    .clone();
                *s = normalized;
            }
        }
        _ => {}
    }
}

fn is_generated_call_id(s: &str) -> bool {
    let Some(suffix) = s.strip_prefix("call_") else {
        return false;
    };
    if suffix.len() == 24 && suffix.bytes().all(|b| b.is_ascii_alphanumeric()) {
        return true;
    }
    let mut parts = suffix.rsplitn(2, '_');
    let Some(counter) = parts.next() else {
        return false;
    };
    let Some(timestamp) = parts.next() else {
        return false;
    };
    !timestamp.is_empty()
        && timestamp.bytes().all(|b| b.is_ascii_hexdigit())
        && counter.bytes().all(|b| b.is_ascii_digit())
}

fn is_generated_function_call_id(s: &str) -> bool {
    let mut parts = s.rsplitn(3, '-');
    let Some(counter) = parts.next() else {
        return false;
    };
    let Some(timestamp) = parts.next() else {
        return false;
    };
    let Some(name) = parts.next() else {
        return false;
    };
    !name.is_empty()
        && counter.bytes().all(|b| b.is_ascii_digit())
        && timestamp.len() >= 10
        && timestamp.bytes().all(|b| b.is_ascii_digit())
}

fn normalize_sse_event_strings(value: &mut Value) {
    match value {
        Value::Object(obj) => {
            for child in obj.values_mut() {
                normalize_sse_event_strings(child);
            }
        }
        Value::Array(items) => {
            for item in items {
                normalize_sse_event_strings(item);
            }
        }
        Value::String(s) => {
            if let Some((event, mut data)) = parse_sse_event_string(s) {
                normalize_unstable_fields(&mut data);
                *s = format!(
                    "event: {event}\ndata: {}",
                    serde_json::to_string(&data).unwrap()
                );
            } else if let Ok(mut parsed) = serde_json::from_str::<Value>(s)
                && (parsed.is_object() || parsed.is_array())
            {
                normalize_unstable_fields(&mut parsed);
                *s = serde_json::to_string(&parsed).unwrap();
            }
        }
        _ => {}
    }
}

fn parse_sse_event_string(s: &str) -> Option<(String, Value)> {
    let mut event = None;
    let mut data = None;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("event: ") {
            event = Some(rest.to_string());
        } else if let Some(rest) = line.strip_prefix("data: ") {
            data = Some(rest);
        }
    }
    let data = serde_json::from_str::<Value>(data?).ok()?;
    Some((event.unwrap_or_default(), data))
}

fn normalize_create_time(value: &mut Value) {
    match value {
        Value::Object(obj) => {
            if let Some(create_time) = obj.get_mut("createTime")
                && create_time.is_string()
            {
                *create_time = Value::String("__normalized_create_time__".to_string());
            }
            for child in obj.values_mut() {
                normalize_create_time(child);
            }
        }
        Value::Array(items) => {
            for item in items {
                normalize_create_time(item);
            }
        }
        Value::String(s) => {
            if s.contains("\"createTime\":")
                && let Ok(mut parsed) = serde_json::from_str::<Value>(s)
            {
                normalize_create_time(&mut parsed);
                *s = serde_json::to_string(&parsed).unwrap();
            }
        }
        _ => {}
    }
}

fn normalize_created_timestamp(value: &mut Value) {
    match value {
        Value::Object(obj) => {
            if let Some(created) = obj.get_mut("created")
                && created.is_number()
            {
                *created = Value::String("__normalized_created__".to_string());
            }
            for child in obj.values_mut() {
                normalize_created_timestamp(child);
            }
        }
        Value::Array(items) => {
            for item in items {
                normalize_created_timestamp(item);
            }
        }
        Value::String(s) => {
            if s.contains("\"created\":")
                && let Ok(mut parsed) = serde_json::from_str::<Value>(s)
            {
                normalize_created_timestamp(&mut parsed);
                *s = serde_json::to_string(&parsed).unwrap();
            }
        }
        _ => {}
    }
}

fn normalize_created_at_timestamp(value: &mut Value) {
    match value {
        Value::Object(obj) => {
            if let Some(created_at) = obj.get_mut("created_at")
                && created_at.is_number()
            {
                *created_at = Value::String("__normalized_created_at__".to_string());
            }
            for child in obj.values_mut() {
                normalize_created_at_timestamp(child);
            }
        }
        Value::Array(items) => {
            for item in items {
                normalize_created_at_timestamp(item);
            }
        }
        Value::String(s) => {
            if s.contains("\"created_at\":")
                && let Ok(mut parsed) = serde_json::from_str::<Value>(s)
            {
                normalize_created_at_timestamp(&mut parsed);
                *s = serde_json::to_string(&parsed).unwrap();
            }
        }
        _ => {}
    }
}

fn go_golden(fixture_path: &Path) -> Value {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .expect("llmptr should have a parent workspace");
    let cli_proxy = workspace.join("CLIProxyAPI");
    let go_cache = manifest_dir.join("target/go-build-cache");
    fs::create_dir_all(&go_cache)
        .unwrap_or_else(|err| panic!("failed to create Go build cache: {err}"));

    let output = Command::new("go")
        .arg("run")
        .arg("./cmd/translator-golden")
        .arg(fixture_path)
        .current_dir(&cli_proxy)
        .env("GOCACHE", &go_cache)
        .output()
        .unwrap_or_else(|err| panic!("failed to run Go golden helper: {err}"));

    if !output.status.success() {
        panic!(
            "Go golden helper failed for {}:\nstdout:\n{}\nstderr:\n{}",
            fixture_path.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    serde_json::from_slice(&output.stdout).unwrap_or_else(|err| {
        panic!(
            "failed to parse Go golden output for {}: {err}\n{}",
            fixture_path.display(),
            String::from_utf8_lossy(&output.stdout)
        )
    })
}
