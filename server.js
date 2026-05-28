import 'dotenv/config';
import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { BypassedClaudeClient, claudeModels, claudeAliases, freemodelModels } from './bypassed_anthropic.js';
import { tavily } from '@tavily/core';
import { ChatOpenAI } from '@langchain/openai';
import { DynamicTool } from '@langchain/core/tools';
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { HumanMessage, AIMessage } from '@langchain/core/messages';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import morgan from 'morgan';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 8080;
const PROD_URL = process.env.PROD_URL || 'https://free-anthropic.vercel.app';

app.use(morgan('dev'));

app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});

app.use(express.json({ limit: '100mb' }));
app.use(express.static(path.join(__dirname, 'public')));

function getApiKey(req) {
  const fromAuth = (req.headers.authorization || '').replace('Bearer ', '').trim();
  const fromBody = (req.body.freemodelapi || '').toString().trim();
  const envKey = (process.env.FREEMODEL_API_KEY || '').toString().trim();
  return fromAuth || fromBody || envKey;
}

function ensureApiKey(key) {
  if (!key) throw new Error('Missing API key. Provide `freemodelapi`, Authorization header, or set FREEMODEL_API_KEY env.');
  return key;
}

function makeClient(apiKey) {
  return new BypassedClaudeClient({ apiKey: ensureApiKey(apiKey) });
}

// ── Standard chat bypass ──
app.post('/api/chat', async (req, res) => {
  try {
    const { thinking = false, streaming = false, stream: reqStream, model, messages = [], system, max_tokens, temperature } = req.body;
    const useStreaming = streaming || reqStream;
    const apiKey = getApiKey(req);
    const client = makeClient(apiKey);

    if (useStreaming) {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no'
      });
      try {
        const stream = await client.create({ model: model || "claude-opus-4-7", messages, system, max_tokens, temperature, stream: true, thinking });
        stream.pipe(res);
      } catch (streamErr) {
        console.error('[Stream error]', streamErr);
        res.write(`data: ${JSON.stringify({ error: streamErr.message })}\n\n`);
        res.end();
      }
    } else {
      const result = await client.create({ model, messages, system, max_tokens, temperature, stream: false, thinking });
      res.status(200).json(result);
    }
  } catch (err) {
    console.error('[Chat error]', err);
    res.status(500).json({ error: err.message });
  }
});

// ── OpenAI-compatible endpoint (for LangChain ChatOpenAI) ──
app.post('/api/v1/chat/completions', async (req, res) => {
  try {
    const { model = 'claude-sonnet-4-20250514', messages = [], temperature = 0.7, max_tokens, stream = false, tools, thinking = false } = req.body;
    const apiKey = getApiKey(req);
    const client = makeClient(apiKey);

    const systemMsgs = messages.filter(m => m.role === 'system');
    const others = messages.filter(m => m.role !== 'system');
    const system = systemMsgs.map(m => m.content).join('\n') || undefined;

    const bypassMessages = others.map(m => {
      if (m.role === 'user') {
        const text = Array.isArray(m.content) ? m.content.filter(p => p.type === 'text').map(p => p.text).join('\n') : m.content;
        return { role: 'user', content: String(text) };
      }
      if (m.role === 'assistant' && m.tool_calls) {
        return { role: 'assistant', content: m.content || '', tool_calls: m.tool_calls.map(tc => ({ id: tc.id, type: 'tool_use', name: tc.function.name, input: JSON.parse(tc.function.arguments || '{}') })) };
      }
      if (m.role === 'assistant') {
        return { role: 'assistant', content: String(m.content || '') };
      }
      if (m.role === 'tool') {
        return { role: 'user', content: `[Tool result for ${m.tool_call_id}]:\n${m.content}` };
      }
      return { role: 'user', content: String(m.content || '') };
    });

    let anthropicTools;
    if (tools && tools.length) {
      anthropicTools = tools.map(t => ({
        name: t.function?.name || t.name,
        description: t.function?.description || t.description || '',
        input_schema: t.function?.parameters || t.input_schema || { type: 'object', properties: {} }
      }));
    }

    const result = await client.create({
      model, messages: bypassMessages, system, temperature, max_tokens, stream: false, thinking,
      tools: anthropicTools
    });

    const content = result.content || [];
    const textBlock = content.find(c => c.type === 'text');
    const toolUseBlocks = content.filter(c => c.type === 'tool_use');

    const msg = { role: 'assistant', content: textBlock?.text || '' };
    if (toolUseBlocks.length) {
      msg.tool_calls = toolUseBlocks.map(tc => ({
        id: tc.id || 'call_' + Date.now(),
        type: 'function',
        function: { name: tc.name, arguments: JSON.stringify(tc.input) }
      }));
    }

    res.json({
      id: 'chatcmpl-' + Date.now(),
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model,
      choices: [{ index: 0, message: msg, finish_reason: toolUseBlocks.length ? 'tool_calls' : 'stop' }],
      usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
    });
  } catch (err) {
    console.error('[OpenAI endpoint error]', err);
    res.status(500).json({ error: err.message, content: [{ type: 'text', text: '' }] });
  }
});

// ── LangChain Agent: Web Search with Tavily ──
app.post('/api/chat-with-search', async (req, res) => {
  try {
    const { model, messages = [], temperature = 0.7 } = req.body;
    const apiKey = getApiKey(req);
    ensureApiKey(apiKey);

    const lastUser = [...messages].reverse().find(m => m.role === 'user');
    if (!lastUser) return res.status(400).json({ error: 'No user message', content: [{ type: 'text', text: '' }] });

    let query = '';
    if (typeof lastUser.content === 'string') {
      query = lastUser.content;
    } else if (Array.isArray(lastUser.content)) {
      query = lastUser.content.filter(c => c.type === 'text').map(c => c.text).filter(Boolean).join('\n');
    }
    const baseUrl = process.env.VERCEL ? PROD_URL : `http://localhost:${PORT}`;

    // ── LangChain ChatOpenAI pointing to our own OpenAI endpoint ──
    const llm = new ChatOpenAI({
      model: model || 'claude-sonnet-4-20250514',
      temperature: temperature || 0.7,
      apiKey,
      configuration: { baseURL: `${baseUrl}/api/v1` }
    });

    // ── Tavily search tool ──
    const searchTool = new DynamicTool({
      name: 'web_search',
      description: 'Search the web for current information. Input should be a search query string.',
      func: async (input) => {
        try {
          const tvly = tavily({ apiKey: process.env.TAVILY_API_KEY });
          const results = await tvly.search(input, { searchDepth: 'advanced', maxResults: 5 });
          if (!results.results?.length) return 'No results found.';
          return results.results.map(r =>
            `[${r.title || 'Untitled'}](${r.url})\n${r.content}`
          ).join('\n\n');
        } catch (e) {
          return `Search error: ${e.message}`;
        }
      }
    });

    // ── Build agent with proper system prompt via messageModifier ──
    const agentTools = [searchTool];
    const agentLlm = llm.bindTools(agentTools);
    const toolNode = new ToolNode(agentTools);

    const agent = createReactAgent({
      llm: agentLlm,
      tools: toolNode,
      messageModifier: `You are a helpful AI assistant. You have access to a web_search tool.

RULES for using web_search:
- Only search the web when the user asks about current events, recent news, real-time data, or specific facts you are unsure about.
- For simple greetings, introductions, or general knowledge questions, answer directly WITHOUT searching.
- Examples of when to search: "what happened today", "latest AI news", "weather in Tokyo", "stock price of Apple"
- Examples of when NOT to search: "hi", "who are you", "what is 2+2", "tell me a joke", "how does this work"

When you do search:
- Use a concise search query (just the key terms, not the full question).
- Cite sources inline using [Source](url).
- Be conversational and accurate.`
    });

    // ── Execute with proper message separation ──
    const result = await agent.invoke({
      messages: [new HumanMessage(query)]
    });

    const lastMsg = result.messages[result.messages.length - 1];
    const text = lastMsg?.content || '';

    res.json({
      content: [{ type: 'text', text: String(text) }],
      model: model || 'claude-sonnet-4-20250514',
      role: 'assistant'
    });
  } catch (err) {
    console.error('[Web Search agent error]', err);
    res.status(200).json({ content: [{ type: 'text', text: '⚠️ Error: ' + err.message }] });
  }
});

// ── Models endpoint ──
app.get('/api/models', (req, res) => {
  res.json({
    models: claudeModels,
    aliases: claudeAliases,
    freemodel_models: freemodelModels,
    key_hint: "For freemodel.dev API, use fe_oa_... keys"
  });
});

app.listen(PORT, (err) => {
  if (err) {
    if (err.code === 'EADDRINUSE') {
      console.error(`\n❌ Error: Port ${PORT} is already in use!`);
      console.error(`Please kill the process using this port, run with a different PORT env, or check for orphaned processes.\n`);
    } else {
      console.error(`\n❌ Error starting server:`, err.message);
    }
    process.exit(1);
  }
  console.log(`\n=============================================================`);
  console.log(`🤖 Claude Bypass Server running at:`);
  console.log(`👉 http://localhost:${PORT}`);
  console.log(`👉 OpenAI endpoint: http://localhost:${PORT}/api/v1/chat/completions`);
  console.log(`=============================================================\n`);
});
