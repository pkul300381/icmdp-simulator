/**
 * api.js
 * LLM Intent Parser fφ   (Paper Definition 4.1, §4.2)
 *
 * The LLM is outside the trusted enforcement boundary.
 * It PROPOSES a raw structured specification g_raw.
 * Deterministic validation (compiler.js) accepts or rejects it.
 *
 * Paper guarantee: "The parser proposes a structured interpretation,
 * but the validator performs authoritative grounding against the action
 * vocabulary, field schema, enumerated values, and unit system."
 *
 * API: Anthropic /v1/messages (claude-sonnet-4-6)
 */

'use strict';

// ---------------------------------------------------------------------------
// System prompt — grounds the LLM to the declared schema and grammar G
// This is the grounding context x ∈ X from Paper Definition 4.1
// ---------------------------------------------------------------------------
function buildSystemPrompt(schema, actions) {
  const schemaDesc = Object.entries(schema).map(([field, def]) => {
    if (def.type === 'enum') return `  - ${field} (enum): ${def.values.join(', ')}`;
    if (def.type === 'number') return `  - ${field} (number, ${def.unit}): ${def.min}–${def.max}`;
    if (def.type === 'integer') return `  - ${field} (integer): ${def.min}–${def.max}`;
    return `  - ${field} (${def.type})`;
  }).join('\n');

  return `You are an intent parser (fφ) in an Intent-Constrained MDP system.

Your ONLY job is to translate a natural-language instruction into a structured JSON specification.
You do NOT enforce constraints. You do NOT select actions. You PROPOSE a specification.
A deterministic validator will accept or reject your output. Do not try to be clever — be precise.

AVAILABLE ACTIONS: ${actions.join(', ')}

STATE FIELDS:
${schemaDesc}

GRAMMAR: Your output must be a JSON object with a "prohibitions" array.
Each prohibition has:
  - "action": one of the available actions (exact spelling)
  - "condition": a condition that, when TRUE, causes the action to be prohibited

Conditions can be:
  - String atom: "field op value"  e.g. "severity = sev1"
  - Object with "not": { "not": condition }
  - Object with "and": { "and": [condition, condition, ...] }
  - Object with "or":  { "or":  [condition, condition, ...] }

OPERATORS by type:
  - enum fields:    = !=
  - number/integer: = != < <= > >=

EXAMPLES:
Instruction: "Do not deploy while a Severity-1 production incident is active"
Output: {"prohibitions": [{"action": "Deploy", "condition": "severity = sev1"}]}

Instruction: "Never scale in when CPU is above 70 percent or latency exceeds 800ms"
Output: {"prohibitions": [{"action": "ScaleIn", "condition": {"or": ["cpu > 70", "latency > 800"]}}]}

Instruction: "Freeze all deployments until further notice"
Output: {"prohibitions": [{"action": "Deploy", "condition": "severity = none"}, {"action": "Deploy", "condition": "severity = sev2"}, {"action": "Deploy", "condition": "severity = sev1"}]}

CRITICAL RULES:
1. Output ONLY valid JSON. No preamble, no explanation, no markdown.
2. Use exact action names and field names as listed above.
3. If the instruction is ambiguous or refers to unknown fields/actions, output: {"error": "reason"}
4. Do not invent fields or actions not in the schema.
5. A prohibition's condition being TRUE means the action is EXCLUDED.`;
}

// ---------------------------------------------------------------------------
// LLM parser call — fφ : L × X → G_raw   (Paper Definition 4.1)
//
// Returns: { success: true, raw: parsed JSON } | { success: false, error }
// ---------------------------------------------------------------------------
async function parseIntent(instruction, schema, actions, apiKey, model = 'claude-3-7-sonnet-20250219') {
  if (!apiKey) {
    return localParseIntent(instruction, schema, actions);
  }

  const systemPrompt = buildSystemPrompt(schema, actions);

  try {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
        'anthropic-dangerous-direct-browser-access': 'true',
      },
      body: JSON.stringify({
        model: model || 'claude-3-7-sonnet-20250219',
        max_tokens: 1000,
        system: systemPrompt,
        messages: [
          {
            role: 'user',
            content: `Instruction: "${instruction}"\n\nOutput the JSON specification:`,
          },
        ],
      }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      return { success: false, error: `API error ${response.status}: ${err.error?.message || response.statusText}` };
    }

    const data = await response.json();
    const text = data.content?.find(b => b.type === 'text')?.text?.trim();

    if (!text) {
      return { success: false, error: 'Empty response from LLM parser.' };
    }

    // Strip any accidental markdown fences
    const cleaned = text.replace(/```json|```/g, '').trim();

    try {
      const parsed = JSON.parse(cleaned);
      if (parsed.error) {
        return { success: false, error: `Parser rejected instruction: ${parsed.error}` };
      }
      return { success: true, raw: parsed };
    } catch (e) {
      return { success: false, error: `Parser output was not valid JSON: ${cleaned.slice(0, 200)}` };
    }
  } catch (err) {
    // If network fails (e.g. invalid key or CORS), fallback or report error
    return { success: false, error: `Network error connecting to Anthropic API: ${err.message}` };
  }
}

// ---------------------------------------------------------------------------
// Local Deterministic Parser (offline/demo mode)
// Translates canonical intents without requiring active external API keys.
// ---------------------------------------------------------------------------
function localParseIntent(instruction, schema, actions) {
  const norm = instruction.toLowerCase().trim();

  // Failure Mode: Under-specification
  if (norm.includes('faster') || norm.includes('better') || norm.includes('optimize') || norm.includes('be safe') || norm.length < 5) {
    return { success: false, error: 'Instruction is under-specified. Please specify an action to prohibit and state conditions.' };
  }

  // Example 1: Sev1 deploy
  if (norm.includes('sev1') || (norm.includes('severity') && (norm.includes('1') || norm.includes('one')))) {
    if (norm.includes('deploy')) {
      return {
        success: true,
        raw: {
          prohibitions: [
            { action: 'Deploy', condition: 'severity = sev1' }
          ]
        }
      };
    }
  }

  // Example 2: Freeze deploy
  if (norm.includes('freeze') && norm.includes('deploy')) {
    return {
      success: true,
      raw: {
        prohibitions: [
          { action: 'Deploy', condition: { or: ['severity = none', 'severity = sev2', 'severity = sev1'] } }
        ]
      }
    };
  }

  // Example 3: Never scale in when CPU is above 70 percent or latency exceeds 800ms
  if (norm.includes('scale') && (norm.includes('in') || norm.includes('down')) && (norm.includes('cpu') || norm.includes('latency'))) {
    return {
      success: true,
      raw: {
        prohibitions: [
          { action: 'ScaleIn', condition: { or: ['cpu > 70', 'latency > 800'] } }
        ]
      }
    };
  }

  // Failure Mode: Infeasible intent / over-tightening (prohibit all actions)
  if (norm.includes('prohibit all') || norm.includes('freeze all actions') || norm.includes('block everything') || norm.includes('infeasible')) {
    return {
      success: true,
      raw: {
        prohibitions: actions.map(a => ({ action: a, condition: 'severity = none' }))
      }
    };
  }

  // Pattern matching for general prohibition: e.g. "Do not <action> when/while/if <field> <op> <val>"
  for (const action of actions) {
    if (norm.includes(action.toLowerCase())) {
      for (const field of Object.keys(schema)) {
        if (norm.includes(field)) {
          const fieldDef = schema[field];
          if (fieldDef.type === 'enum') {
            for (const val of fieldDef.values) {
              if (norm.includes(val)) {
                return {
                  success: true,
                  raw: {
                    prohibitions: [{ action, condition: `${field} = ${val}` }]
                  }
                };
              }
            }
          } else {
            const numMatch = norm.match(/(\d+)/);
            if (numMatch) {
              let op = '=';
              if (norm.includes('above') || norm.includes('exceed') || norm.includes('greater') || norm.includes('>')) op = '>';
              else if (norm.includes('below') || norm.includes('under') || norm.includes('less') || norm.includes('<')) op = '<';
              return {
                success: true,
                raw: {
                  prohibitions: [{ action, condition: `${field} ${op} ${numMatch[1]}` }]
                }
              };
            }
          }
        }
      }
    }
  }

  return {
    success: false,
    error: 'Could not parse instruction locally. Enter an Anthropic API key in Settings for full LLM parser support, or use one of the preset examples.'
  };
}

if (typeof module !== 'undefined') {
  module.exports = { parseIntent, localParseIntent, buildSystemPrompt };
}
