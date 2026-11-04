//! Selected native tool definitions and the publisher's single-call envelope.
//!
//! This is a codec, not a JSON-schema validator or an execution registry.
//! Faculty argument validation still owns operation-specific constraints.
//! `Value`/`Map` is the explicit MCP compatibility boundary for one selected
//! schema or invocation, not a retained database or lookup catalogue. Tools
//! are an immutable vector; lookup borrows it for one parse/render operation.
//! Downstream canonical persistence can use json_tree, while the exact call
//! body must remain separate from any typed-value reserialization.

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value, json};

/// One explicitly selected schema. Its rendered definition is part of the prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

impl ToolDefinition {
    pub fn exec() -> Self {
        Self {
            name: "exec".into(),
            description: "Propose a shell command.".into(),
            input_schema: json!({"type":"object","properties":{"command":{
                "type":"string","description":"Shell command text."}},"required":["command"]}),
        }
    }
}

/// Immutable selection installed before prompt preparation and token admission.
/// Changing definitions requires preparing a new codec/history; token prefix
/// equality, not a registry label, determines inference-state reuse.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tools {
    definitions: Vec<ToolDefinition>,
}

impl Tools {
    pub fn new(definitions: Vec<ToolDefinition>) -> Result<Self> {
        for (index, tool) in definitions.iter().enumerate() {
            valid_name(&tool.name)?;
            ensure!(
                !definitions[..index]
                    .iter()
                    .any(|prior| prior.name == tool.name),
                "duplicate registered function name"
            );
            let schema = tool
                .input_schema
                .as_object()
                .context("tool schema must be an object")?;
            ensure!(
                schema.get("type").and_then(Value::as_str) == Some("object"),
                "tool arguments must have object schema"
            );
            if let Some(properties) = schema.get("properties") {
                for (name, shape) in properties
                    .as_object()
                    .context("tool properties must be an object")?
                {
                    valid_name(name)?;
                    ensure!(shape.is_object(), "parameter schema must be an object");
                }
            }
            if let Some(required) = schema.get("required") {
                for name in required.as_array().context("required must be an array")? {
                    let name = name.as_str().context("required parameter must be a name")?;
                    ensure!(
                        schema
                            .get("properties")
                            .and_then(Value::as_object)
                            .is_some_and(|properties| properties.contains_key(name)),
                        "required parameter has no declared schema"
                    );
                }
            }
        }
        Ok(Self { definitions })
    }

    /// The existing standalone exec-only behavior; ordinary owners explicitly
    /// replace this selection with their actual registered definitions.
    pub fn exec_only() -> Self {
        Self {
            definitions: vec![ToolDefinition::exec()],
        }
    }

    pub fn definitions(&self) -> &[ToolDefinition] {
        &self.definitions
    }

    fn definition(&self, name: &str) -> Result<&ToolDefinition> {
        self.definitions
            .iter()
            .find(|tool| tool.name == name)
            .context("unregistered function name")
    }

    pub(super) fn render(&self) -> Result<String> {
        let mut out =
            String::from("# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for tool in &self.definitions {
            out.push_str("\n<function>\n<name>");
            out.push_str(&tool.name);
            out.push_str("</name>\n<description>");
            out.push_str(tool.description.trim());
            out.push_str("</description>\n<parameters>");
            if let Some(properties) = tool
                .input_schema
                .get("properties")
                .and_then(Value::as_object)
            {
                for (name, shape) in properties {
                    out.push_str("\n<parameter>\n<name>");
                    out.push_str(name);
                    out.push_str("</name>");
                    for key in ["type", "description", "enum"] {
                        if let Some(value) = shape.get(key) {
                            let value = if key == "description" {
                                value
                                    .as_str()
                                    .context("parameter description must be text")?
                                    .trim()
                                    .to_owned()
                            } else if key == "enum" {
                                json_text(value)?
                            } else {
                                value
                                    .as_str()
                                    .context("unsupported parameter type representation")?
                                    .to_owned()
                            };
                            field(&mut out, key, &value)?;
                        }
                    }
                    extra_fields(&mut out, shape, &["name", "type", "description", "enum"])?;
                    out.push_str("\n</parameter>");
                }
            }
            extra_fields(
                &mut out,
                &tool.input_schema,
                &["type", "properties", "required"],
            )?;
            if let Some(required) = tool.input_schema.get("required") {
                field(&mut out, "required", &json_text(required)?)?;
            }
            out.push_str("\n</parameters>\n</function>");
        }
        out.push_str("\n</tools>");
        Ok(out)
    }

    /// Decode one complete body after lossless ByteLevel decoding. Delimiter
    /// framing is exact: remove one template LF, never trim string arguments.
    pub fn parse_call(&self, body: &str) -> Result<Invocation> {
        let rest = body
            .strip_prefix("\n<function=")
            .context("expected function opener")?;
        let (name, mut rest) = rest
            .split_once(">\n")
            .context("incomplete function opener")?;
        valid_name(name)?;
        let tool = self.definition(name)?;
        // Ephemeral ordered parameter text for this one decode only.
        let mut raw: Vec<(&str, &str)> = Vec::new();
        while let Some(next) = rest.strip_prefix("<parameter=") {
            let (parameter, next) = next
                .split_once(">\n")
                .context("incomplete parameter opener")?;
            valid_name(parameter)?;
            ensure!(
                !raw.iter().any(|(prior, _)| *prior == parameter),
                "duplicate parameter name"
            );
            let (value, next) = next
                .split_once("\n</parameter>\n")
                .context("incomplete parameter body")?;
            valid_value(value)?;
            raw.push((parameter, value));
            rest = next;
        }
        ensure!(
            rest == "</function>\n",
            "expected exactly one complete function body"
        );
        let properties = tool
            .input_schema
            .get("properties")
            .and_then(Value::as_object);
        let mut arguments = Map::new();
        for (parameter, value) in &raw {
            let shape = properties
                .and_then(|p| p.get(*parameter))
                .context("undeclared parameter")?;
            if shape.get("type").is_some() {
                arguments.insert((*parameter).to_owned(), typed_value(value, shape)?);
            }
        }
        // Only the observed discriminator shape: an untyped parameter may use
        // exactly one root oneOf branch selected by already decoded consts.
        // No unions, refs, recursive evaluation or general schema validation.
        let selected = raw
            .iter()
            .filter(|(parameter, _)| !arguments.contains_key(*parameter))
            .map(|(parameter, value)| {
                let shape = discriminated_shape(&tool.input_schema, parameter, &arguments)?;
                Ok(((*parameter).to_owned(), typed_value(value, shape)?))
            })
            .collect::<Result<Vec<_>>>()?;
        arguments.extend(selected);
        if let Some(required) = tool.input_schema.get("required").and_then(Value::as_array) {
            for name in required {
                ensure!(
                    arguments.contains_key(name.as_str().context("invalid required name")?),
                    "missing required parameter"
                );
            }
        }
        if name == "exec" {
            let command = arguments
                .get("command")
                .and_then(Value::as_str)
                .context("exec requires string command")?;
            ensure!(
                arguments.len() == 1,
                "exec requires exactly one command parameter"
            );
            super::valid_command(command)?;
        }
        Ok(Invocation {
            name: name.to_owned(),
            arguments,
            body: body.to_owned(),
        })
    }
}

/// A complete, typed, selected invocation plus the exact original call body.
/// It is not authority to execute: only an EOS-complete parser outcome releases
/// a generated invocation. Partial output never contains this value.
///
/// `arguments` uses serde_json's integer/f64 representation, not arbitrary
/// precision. Integer parameters reject out-of-range/floating representations;
/// number and nested free-form JSON may round to finite f64. `body` preserves
/// every original numeric lexeme, whitespace and parameter order. Persist it
/// as the original; a json_tree made from `arguments` is the typed projection,
/// not a claim of lossless original JSON structure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Invocation {
    name: String,
    arguments: Map<String, Value>,
    body: String,
}

impl Invocation {
    /// Canonical publisher rendering for host-created history, such as wake
    /// recalls. Generated history must retain the original parser Invocation.
    pub fn new(
        name: impl Into<String>,
        arguments: Map<String, Value>,
        tools: &Tools,
    ) -> Result<Self> {
        let name = name.into();
        valid_name(&name)?;
        let mut body = format!("\n<function={name}>\n");
        for (parameter, value) in &arguments {
            valid_name(parameter)?;
            let value = scalar_text(value)?;
            valid_value(&value)?;
            body.push_str(&format!("<parameter={parameter}>\n{value}\n</parameter>\n"));
        }
        body.push_str("</function>\n");
        let invocation = tools.parse_call(&body)?;
        ensure!(
            invocation.arguments == arguments,
            "tool schema does not preserve argument types"
        );
        Ok(invocation)
    }

    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn arguments(&self) -> &Map<String, Value> {
        &self.arguments
    }
    pub fn body(&self) -> &str {
        &self.body
    }

    pub(super) fn validate(&self, tools: &Tools) -> Result<()> {
        ensure!(
            &tools.parse_call(&self.body)? == self,
            "history invocation differs from selected schema"
        );
        Ok(())
    }
}

fn valid_name(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty()
            && name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b)),
        "unsupported function/parameter/schema-key spelling"
    );
    Ok(())
}

fn valid_value(value: &str) -> Result<()> {
    for delimiter in [
        "</parameter>",
        "</function>",
        "</tool_call>",
        "<parameter=",
        "<function=",
        "<tool_call>",
    ] {
        ensure!(!value.contains(delimiter), "ambiguous parameter delimiter");
    }
    Ok(())
}

fn typed_value(text: &str, shape: &Value) -> Result<Value> {
    let kind = shape
        .get("type")
        .and_then(Value::as_str)
        .context("unsupported or ambiguous parameter type")?;
    if kind == "string" {
        return Ok(Value::String(text.to_owned()));
    }
    // Jinja's scalar string filter emits Python spellings; nested containers
    // use JSON. JSON bool/null spellings are also unambiguous under these types.
    let value = match (kind, text) {
        ("boolean", "True") => Value::Bool(true),
        ("boolean", "False") => Value::Bool(false),
        ("null", "None") => Value::Null,
        _ => serde_json::from_str(text).context("invalid typed parameter value")?,
    };
    let matches = match kind {
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        // serde_json falls back to f64 for overflowing integer lexemes. Do not
        // accept that fallback (or decimal/exponent representations) here.
        "integer" => value.is_i64() || value.is_u64(),
        "number" => value.as_f64().is_some_and(f64::is_finite),
        "object" => value.is_object(),
        "array" => value.is_array(),
        _ => bail!("unsupported parameter type"),
    };
    ensure!(matches, "parameter value has wrong type");
    Ok(value)
}

fn discriminated_shape<'a>(
    schema: &'a Value,
    parameter: &str,
    decoded: &Map<String, Value>,
) -> Result<&'a Value> {
    let branches = schema
        .get("oneOf")
        .and_then(Value::as_array)
        .context("untyped parameter has no supported discriminator")?;
    let mut selected = None;
    for branch in branches {
        let Some(properties) = branch.get("properties").and_then(Value::as_object) else {
            continue;
        };
        let mut has_const = false;
        let matches = properties.iter().all(|(name, shape)| {
            let Some(expected) = shape.get("const") else {
                return true;
            };
            has_const = true;
            decoded.get(name) == Some(expected)
        });
        if has_const && matches {
            ensure!(selected.is_none(), "ambiguous parameter discriminator");
            selected = Some(properties);
        }
    }
    selected
        .context("no matching parameter discriminator")?
        .get(parameter)
        .context("selected discriminator has no parameter type")
}

fn field(out: &mut String, name: &str, value: &str) -> Result<()> {
    valid_name(name)?;
    out.push_str(&format!("\n<{name}>{value}</{name}>"));
    Ok(())
}

fn extra_fields(out: &mut String, shape: &Value, handled: &[&str]) -> Result<()> {
    for (key, value) in shape.as_object().context("schema must be an object")? {
        if !handled.contains(&key.as_str()) {
            field(out, key, &scalar_text(value)?)?;
        }
    }
    Ok(())
}

fn scalar_text(value: &Value) -> Result<String> {
    Ok(match value {
        Value::String(value) => value.clone(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Null => "None".into(),
        _ => json_text(value)?,
    })
}

// The retained publisher reference uses json.dumps(ensure_ascii=False): spaces
// after separators, strings losslessly quoted, and JSON booleans inside arrays
// and objects. No debug formatting or Python expression parser is involved.
fn json_text(value: &Value) -> Result<String> {
    Ok(match value {
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(json_text)
                .collect::<Result<Vec<_>>>()?
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| {
                    Ok(format!(
                        "{}: {}",
                        serde_json::to_string(key)?,
                        json_text(value)?
                    ))
                })
                .collect::<Result<Vec<_>>>()?
                .join(", ")
        ),
        _ => serde_json::to_string(value)?,
    })
}
