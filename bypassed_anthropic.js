import https from 'https';
import zlib from 'zlib';
import crypto from 'crypto';
import { Transform } from 'stream';


export const claudeModels = [
  "claude-opus-4-7",
  "claude-sonnet-4-6",
  "claude-opus-4-6",
  "claude-haiku-4-5",
  "claude-opus-4-5",
  "claude-sonnet-4-5",
  "claude-opus-4-1",
  "claude-opus-4-0",
  "claude-sonnet-4-0",
  "claude-3-haiku-20240307",
  "claude-3-7-sonnet-20250219",
];

export const claudeAliases = ["opus", "sonnet", "haiku"];

export const freemodelModels = [
  "gpt-5.5",
  "gpt-5.4",
  "gpt-5.4-mini",
  "gpt-5.3-codex",
];

const freemodelModelMap = {
  'claude-opus-4-7': 'gpt-5.5',
  'claude-sonnet-4-6': 'gpt-5.4',
  'claude-haiku-4-5': 'gpt-5.4-mini',
  'claude-opus-4-6': 'gpt-5.5',
  'claude-opus-4-5': 'gpt-5.5',
  'claude-sonnet-4-5': 'gpt-5.4',
  'claude-sonnet-4-0': 'gpt-5.4',
  'claude-opus-4-1': 'gpt-5.5',
  'claude-opus-4-0': 'gpt-5.5',
  'gpt-5.5': 'gpt-5.5',
  'gpt-5.4': 'gpt-5.4',
  'gpt-5.4-mini': 'gpt-5.4-mini',
  'gpt-5.3-codex': 'gpt-5.3-codex',
};

const freemodelAliasMap = {
  opus: 'gpt-5.5',
  sonnet: 'gpt-5.4',
  haiku: 'gpt-5.4-mini',
};

const keepAliveAgent = new https.Agent({
  keepAlive: true,
  maxSockets: 128,
  keepAliveMsecs: 15000,
  timeout: 60000
});

function openaiToAnthropicSSE() {
  let started = false;
  let messageId = 'msg_' + crypto.randomBytes(12).toString('hex');
  return new Transform({
    readableObjectMode: false,
    writableObjectMode: false,
    transform(chunk, encoding, callback) {
      const str = chunk.toString('utf8');
      const lines = str.split('\n');
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (!raw || raw === '[DONE]') continue;
        try {
          const p = JSON.parse(raw);
          const choices = p.choices || [];
          for (const c of choices) {
            const delta = c.delta || {};
            const finish = c.finish_reason;
            if (delta.role === 'assistant' && !started) {
              started = true;
              this.push(`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n`);
            }
            if (delta.content) {
              this.push(`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":${JSON.stringify(delta.content)}}}\n\n`);
            }
            if (finish === 'stop') {
              this.push(`data: {"type":"content_block_stop","index":0}\n\n`);
              this.push(`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}}\n\n`);
            }
          }
        } catch (e) {
          // skip parse errors
        }
      }
      callback();
    }
  });
}

export class BypassedClaudeClient {
  constructor(options = {}) {
    this.apiKey = (options.apiKey?.toString().trim() || process.env.FREEMODEL_API_KEY?.toString().trim()) || null;
    if (!this.apiKey) {
      throw new Error('Missing API key. Set FREEMODEL_API_KEY or pass `freemodelapi` in the request body.');
    }
    this.model = options.model || "claude-opus-4-7";
    this.maxTokens = options.maxTokens || 64000;
    this.temperature = options.temperature;
  }

  async create(params = {}) {
    const {
      messages = [],
      system: clientSystem,
      model = this.model,
      max_tokens = this.maxTokens,
      temperature = this.temperature,
      stream: clientStream = false,
      thinking,
      tools,
      onToken
    } = params;

    const keyStr = this.apiKey.trim();
    const isFreemodelApi = keyStr.startsWith('fe_oa_');
    const isDirectAnthropic = !isFreemodelApi && keyStr.startsWith('sk-ant-');
    const targetHost = isDirectAnthropic ? 'api.anthropic.com' : isFreemodelApi ? 'api.freemodel.dev' : 'cc.freemodel.dev';

    let resolvedModel = model || this.model;
    const lowerModel = resolvedModel.toLowerCase();

    if (isFreemodelApi) {
      return this._createFreemodel({ messages, clientSystem, resolvedModel, lowerModel, max_tokens, temperature, clientStream, onToken });
    }

    if (lowerModel === 'opus') resolvedModel = 'claude-opus-4-7';
    else if (lowerModel === 'sonnet') resolvedModel = 'claude-sonnet-4-6';
    else if (lowerModel === 'haiku') resolvedModel = 'claude-haiku-4-5';

    let targetModel = resolvedModel;
    if (isDirectAnthropic) {
      const m = resolvedModel.toLowerCase();
      if (m.includes('opus')) {
        targetModel = 'claude-3-opus-20240229';
      } else if (m.includes('sonnet')) {
        targetModel = 'claude-3-7-sonnet-20250219';
      } else if (m.includes('haiku')) {
        if (m.includes('20240307')) {
          targetModel = 'claude-3-haiku-20240307';
        } else {
          targetModel = 'claude-3-5-haiku-20241022';
        }
      } else {
        targetModel = 'claude-3-7-sonnet-20250219';
      }
    }

    const sessionId = crypto.randomUUID();
    const deviceId = crypto.randomBytes(16).toString('hex');

    const systemArray = [
      {
        type: 'text',
        text: 'x-anthropic-billing-header: cc_version=2.1.152.8a5; cc_entrypoint=cli; cch=40f45;'
      },
      {
        type: 'text',
        text: "You are Claude Code, Anthropic's official CLI for Claude.",
        cache_control: { type: 'ephemeral' }
      }
    ];

    if (clientSystem) {
      if (typeof clientSystem === 'string') {
        systemArray.push({ type: 'text', text: clientSystem });
      } else if (Array.isArray(clientSystem)) {
        systemArray.push(...clientSystem);
      }
    }

    const requestBody = {
      model: targetModel,
      max_tokens,
      stream: clientStream,
      messages,
      system: systemArray,
      metadata: {
        user_id: JSON.stringify({
          device_id: deviceId,
          account_uuid: '',
          session_id: sessionId
        })
      }
    };
    if (tools && tools.length) {
      requestBody.tools = tools;
    }

    const isSonnet = targetModel.toLowerCase().includes('sonnet');
    if (isSonnet && isDirectAnthropic) {
      if (thinking === true) {
        requestBody.thinking = {
          type: 'enabled',
          budget_tokens: 1024
        };
        if (!max_tokens || max_tokens <= 2048) {
          requestBody.max_tokens = 4096;
        }
      } else if (thinking && typeof thinking === 'object') {
        requestBody.thinking = thinking;
      }
    }

    if (temperature !== undefined && !requestBody.thinking) {
      requestBody.temperature = temperature;
    }

    if (!this.apiKey) {
      throw new Error('API key missing: cannot create request without a valid FREEMODEL_API_KEY or freemodelapi value.');
    }

    const bodyStr = JSON.stringify(requestBody);

    const headers = {
      'accept': clientStream ? 'text/event-stream' : 'application/json',
      'content-type': 'application/json',
      'user-agent': 'claude-cli/2.1.152 (external, cli)',
      'x-claude-code-session-id': sessionId,
      'x-stainless-arch': 'x64',
      'x-stainless-lang': 'js',
      'x-stainless-os': 'Linux',
      'x-stainless-package-version': '0.94.0',
      'x-stainless-retry-count': '0',
      'x-stainless-runtime': 'node',
      'x-stainless-runtime-version': 'v24.3.0',
      'x-stainless-timeout': '600',
      'anthropic-beta': 'claude-code-20250219,context-1m-2025-08-07,interleaved-thinking-2025-05-14,redact-thinking-2026-02-12,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,advisor-tool-2026-03-01,effort-2025-11-24',
      'anthropic-dangerous-direct-browser-access': 'true',
      'anthropic-version': '2023-06-01',
      'x-app': 'cli',
      'content-length': Buffer.byteLength(bodyStr)
    };

    if (isDirectAnthropic) {
      headers['x-api-key'] = this.apiKey.trim();
    } else {
      headers['authorization'] = `Bearer ${this.apiKey.trim()}`;
    }

    return new Promise((resolve, reject) => {
      const req = https.request({
        hostname: targetHost,
        port: 443,
        path: '/v1/messages?beta=true',
        method: 'POST',
        headers,
        agent: keepAliveAgent
      }, (res) => {
        if (res.statusCode !== 200) {
          let errorBody = '';
          res.on('data', (chunk) => {
            errorBody += chunk.toString('utf8');
          });
          res.on('end', () => {
            const cleaned = errorBody.trim();
            let msg = `Upstream returned status ${res.statusCode}: ${cleaned || res.statusMessage}`;
            if (!isDirectAnthropic && res.statusCode === 400) {
              msg = `[Freemodel Gateway] ${msg}\nThe cc.freemodel.dev gateway is deprecated. If you have a fe_oa_... API key, it works with api.freemodel.dev (already supported).\n\nSolutions:\n1. Use an Anthropic API key starting with 'sk-ant-' for direct routing\n2. Use a fe_oa_... key (auto-routes to api.freemodel.dev)`;
            }
            reject(new Error(msg));
          });
          return;
        }

        let stream = res;
        const encoding = res.headers['content-encoding'];
        if (encoding === 'br') {
          stream = res.pipe(zlib.createBrotliDecompress());
        } else if (encoding === 'gzip') {
          stream = res.pipe(zlib.createGunzip());
        } else if (encoding === 'deflate') {
          stream = res.pipe(zlib.createInflate());
        }

        if (clientStream) {
          resolve(stream);
          return;
        }

        let accumulatedData = '';
        stream.on('data', (chunk) => {
          accumulatedData += chunk.toString('utf8');
        });

        stream.on('end', () => {
          if (!clientStream) {
            try {
              const cleaned = accumulatedData.trim();
              if (cleaned.startsWith('{') && cleaned.endsWith('}')) {
                const parsed = JSON.parse(cleaned);
                if (parsed.content) {
                  resolve(parsed);
                  return;
                }
              }
            } catch (e) {
            }
          }

          let fullText = '';
          let messageId = 'msg_' + crypto.randomBytes(12).toString('hex');
          let stopReason = 'end_turn';
          let inputTokens = 0;
          let outputTokens = 0;

          const lines = accumulatedData.split('\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const dataStr = line.slice(6).trim();
              if (!dataStr) continue;
              try {
                const parsed = JSON.parse(dataStr);
                if (parsed.type === 'message_start' && parsed.message) {
                  messageId = parsed.message.id || messageId;
                  if (parsed.message.usage) {
                    inputTokens = parsed.message.usage.input_tokens || inputTokens;
                  }
                } else if (parsed.type === 'content_block_delta' && parsed.delta && parsed.delta.text) {
                  fullText += parsed.delta.text;
                  if (onToken) onToken(parsed.delta.text);
                } else if (parsed.type === 'message_delta' && parsed.delta) {
                  if (parsed.delta.stop_reason) {
                    stopReason = parsed.delta.stop_reason;
                  }
                  if (parsed.usage) {
                    outputTokens = parsed.usage.output_tokens || outputTokens;
                  }
                }
              } catch (e) {
              }
            }
          }

          resolve({
            id: messageId,
            type: "message",
            role: "assistant",
            content: [{ type: "text", text: fullText }],
            model: targetModel,
            stop_reason: stopReason,
            stop_sequence: null,
            usage: {
              input_tokens: inputTokens,
              output_tokens: outputTokens
            }
          });
        });

        stream.on('error', (err) => reject(err));
      });

      req.on('error', (err) => reject(err));
      req.write(bodyStr);
      req.end();
    });
  }

  async _createFreemodel({ messages, clientSystem, resolvedModel, lowerModel, max_tokens, temperature, clientStream, onToken }) {
    const aliasMap = {
      opus: 'gpt-5.5',
      sonnet: 'gpt-5.4',
      haiku: 'gpt-5.4-mini',
    };

    let gptModel = freemodelModelMap[resolvedModel] || aliasMap[lowerModel] || 'gpt-5.5';
    if (!gptModel) gptModel = 'gpt-5.5';

    const openaiMessages = [];
    if (clientSystem) {
      openaiMessages.push({ role: 'system', content: typeof clientSystem === 'string' ? clientSystem : JSON.stringify(clientSystem) });
    }
    for (const m of messages) {
      const role = m.role || 'user';
      let content = m.content;
      if (Array.isArray(content)) {
        const converted = [];
        for (const block of content) {
          if (block.type === 'text') {
            converted.push({ type: 'text', text: block.text || '' });
          } else if (block.type === 'image' && block.source?.type === 'base64') {
            converted.push({
              type: 'image_url',
              image_url: { url: `data:${block.source.media_type};base64,${block.source.data}` }
            });
          } else if (block.type === 'document' && block.source?.type === 'base64') {
            converted.push({
              type: 'text',
              text: `[Attached file: ${block.source.media_type}, size: ${(block.source.data.length * 0.75).toFixed(0)} bytes]`
            });
          }
        }
        content = converted.length ? converted : (typeof m.content === 'string' ? m.content : '');
      }
      openaiMessages.push({ role, content });
    }

    const requestBody = {
      model: gptModel,
      messages: openaiMessages,
      stream: clientStream,
    };
    if (max_tokens) requestBody.max_tokens = max_tokens;
    if (temperature !== undefined) requestBody.temperature = temperature;

    if (!this.apiKey) {
      throw new Error('API key missing');
    }

    const bodyStr = JSON.stringify(requestBody);

    const headers = {
      'accept': clientStream ? 'text/event-stream' : 'application/json',
      'content-type': 'application/json',
      'authorization': `Bearer ${this.apiKey.trim()}`,
      'content-length': Buffer.byteLength(bodyStr)
    };

    return new Promise((resolve, reject) => {
      const req = https.request({
        hostname: 'api.freemodel.dev',
        port: 443,
        path: '/v1/chat/completions',
        method: 'POST',
        headers,
        agent: keepAliveAgent
      }, (res) => {
        if (res.statusCode !== 200) {
          let errorBody = '';
          res.on('data', (chunk) => {
            errorBody += chunk.toString('utf8');
          });
          res.on('end', () => {
            const cleaned = errorBody.trim();
            reject(new Error(`Freemodel API returned status ${res.statusCode}: ${cleaned || res.statusMessage}`));
          });
          return;
        }

        let stream = res;
        const encoding = res.headers['content-encoding'];
        if (encoding === 'br') {
          stream = res.pipe(zlib.createBrotliDecompress());
        } else if (encoding === 'gzip') {
          stream = res.pipe(zlib.createGunzip());
        } else if (encoding === 'deflate') {
          stream = res.pipe(zlib.createInflate());
        }

        if (clientStream) {
          const converter = openaiToAnthropicSSE();
          stream.pipe(converter);
          resolve(converter);
          return;
        }

        let accumulatedData = '';
        stream.on('data', (chunk) => {
          accumulatedData += chunk.toString('utf8');
        });

        stream.on('end', () => {
          let content = '';
          let finishReason = 'stop';
          let inputTokens = 0;
          let outputTokens = 0;

          try {
            const parsed = JSON.parse(accumulatedData.trim());
            if (parsed.choices && parsed.choices.length > 0) {
              content = parsed.choices[0].message?.content || '';
              finishReason = parsed.choices[0].finish_reason || 'stop';
            }
            if (parsed.usage) {
              inputTokens = parsed.usage.prompt_tokens || 0;
              outputTokens = parsed.usage.completion_tokens || 0;
            }
          } catch (e) {
            const lines = accumulatedData.split('\n');
            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const raw = line.slice(6).trim();
                if (!raw || raw === '[DONE]') continue;
                try {
                  const p = JSON.parse(raw);
                  const choices = p.choices || [];
                  for (const c of choices) {
                    if (c.delta?.content) content += c.delta.content;
                    if (c.finish_reason) finishReason = c.finish_reason;
                  }
                } catch (e2) {}
              }
            }
          }

          const anthropicResponse = {
            id: 'msg_' + crypto.randomBytes(12).toString('hex'),
            type: "message",
            role: "assistant",
            content: [{ type: "text", text: content }],
            model: gptModel,
            stop_reason: finishReason === 'stop' ? 'end_turn' : finishReason,
            stop_sequence: null,
            usage: {
              input_tokens: inputTokens,
              output_tokens: outputTokens
            }
          };

          resolve(anthropicResponse);
        });

        stream.on('error', (err) => reject(err));
      });

      req.on('error', (err) => reject(err));
      req.write(bodyStr);
      req.end();
    });
  }
}


export class BypassedChatAnthropic {
  static getLangChainClass(BaseChatModelClass) {
    return class extends BaseChatModelClass {
      constructor(fields = {}) {
        super(fields);
        this.client = new BypassedClaudeClient({
          apiKey: fields.apiKey,
          model: fields.modelName || fields.model,
          maxTokens: fields.maxTokens,
          temperature: fields.temperature
        });
      }

      _llmType() {
        return "bypassed-anthropic-chat";
      }

      _convertMessages(messages) {
        const formatted = [];
        let systemPrompt = null;
        for (const msg of messages) {
          const type = msg._getType();
          if (type === "system") {
            systemPrompt = msg.content;
          } else if (type === "human") {
            formatted.push({ role: "user", content: msg.content });
          } else if (type === "ai") {
            formatted.push({ role: "assistant", content: msg.content });
          } else {
            formatted.push({ role: msg.role || "user", content: msg.content });
          }
        }
        return { formatted, systemPrompt };
      }

      async _generate(messages, options, runManager) {
        const { formatted, systemPrompt } = this._convertMessages(messages);
        const onToken = (token) => {
          if (runManager) runManager.handleLLMNewToken(token);
        };

        const res = await this.client.create({
          messages: formatted,
          system: systemPrompt,
          onToken
        });

        const chatGeneration = {
          message: {
            content: res.content[0].text,
            _getType: () => "ai"
          }
        };
        return { generations: [chatGeneration] };
      }
    };
  }
}
